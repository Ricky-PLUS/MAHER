import os
import cv2
import base64
import logging
import traceback
import numpy as np
import torch
import json
import tempfile
from flask import Flask, request, jsonify, Response
import argparse

# 导入 MoGe 与 WildDet3D
from moge.model.v2 import MoGeModel 
from wilddet3d import build_model, preprocess
from wilddet3d.vis.visualize import draw_3d_boxes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

moge_model = None
wild_model = None
device = None
USE_CPU_OFFLOAD = False

def load_models(args):
    global moge_model, wild_model, device, USE_CPU_OFFLOAD
    try:
        USE_CPU_OFFLOAD = args.cpu_offload
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        target_device = 'cpu' if USE_CPU_OFFLOAD else device
        logger.info(f"System has GPU: {device}. Target model holding device: {target_device} (CPU Offload: {USE_CPU_OFFLOAD})")

        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        logger.info(f"Loading MoGe Model to {target_device}...")
        moge_model = MoGeModel.from_pretrained(args.moge_ckpt).to(target_device)
        moge_model.eval()

        logger.info(f"Loading WildDet3D Model and sub-components to {target_device}...")
        wild_model = build_model(
            checkpoint=args.wilddet3d_ckpt,
            sam3_checkpoint=args.sam3_ckpt,
            lingbot_checkpoint=args.lingbot_ckpt,
            score_threshold=args.score_threshold,
            skip_pretrained=True,
            use_depth_input_test=True, 
        )
        wild_model.to(target_device)
        wild_model.eval()
        
        logger.info(f"All Models loaded successfully! Mode: {'RAM Save VRAM' if USE_CPU_OFFLOAD else 'VRAM Fast Performance'}")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {traceback.format_exc()}")
        return False

@app.route('/infer', methods=['POST'])
def infer():
    global moge_model, wild_model, device, USE_CPU_OFFLOAD
    if moge_model is None or wild_model is None: 
        return jsonify({"error": "Models not fully loaded"}), 500
    
    try:
        data = request.get_json()
        if 'image' not in data: 
            return jsonify({"error": "Missing image"}), 400
            
        boundingbox = data.get('boundingbox')
        
        if not boundingbox and 'boundingboxes' in data:
            boxes = data.get('boundingboxes')
            if boxes and isinstance(boxes[0], list):
                boundingbox = boxes[0]  
            elif boxes and isinstance(boxes[0], (int, float)):
                boundingbox = boxes   

        if not boundingbox: 
            return jsonify({"error": "Provide a boundingbox"}), 400

        if len(boundingbox) != 4 or not all(isinstance(x, (int, float)) for x in boundingbox): 
            return jsonify({"error": "Invalid format. Provide a single box like: [x_min, y_min, x_max, y_max]"}), 400
            
        return_vis = data.get('return_vis', False)
        
        try:
            image_bytes = base64.b64decode(data['image'])
            image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            H, W = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return jsonify({"error": f"Image decoding failed: {e}"}), 400
            
        logger.info(f"Processing image shape: {H}x{W}, single boundingbox: {boundingbox}")

        with torch.inference_mode():
            if USE_CPU_OFFLOAD:
                moge_model.to(device)
            
            input_image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)    
            moge_output = moge_model.infer(input_image_tensor)

            depth_moge = moge_output['depth'].cpu().numpy().astype(np.float32)
            intrinsics_raw = moge_output['intrinsics'].cpu().numpy()
            
            if USE_CPU_OFFLOAD:
                moge_model.to('cpu')
                del input_image_tensor, moge_output
                torch.cuda.empty_cache()

            if depth_moge.ndim == 3:
                depth_moge = depth_moge.squeeze()
            scale_matrix = np.array([[W, 0, W], [0, H, H], [0, 0, 1]], dtype=np.float32)
            intrinsics_pixel = (intrinsics_raw * scale_matrix).astype(np.float32)

            image_np = image_rgb.astype(np.float32)
            prep_data = preprocess(image_np, intrinsics_pixel, depth=depth_moge)

            if USE_CPU_OFFLOAD:
                wild_model.to(device) 
            
            results = wild_model(
                images=prep_data["images"].cuda(),
                intrinsics=prep_data["intrinsics"].cuda()[None],
                input_hw=[prep_data["input_hw"]],
                original_hw=[prep_data["original_hw"]],
                padding=[prep_data["padding"]],
                input_boxes=[boundingbox],  
                prompt_text="geometric",
                depth_gt=prep_data["depth_gt"].cuda(),  
            )
            
            if USE_CPU_OFFLOAD:
                wild_model.to('cpu') 
                torch.cuda.empty_cache()

        boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results

        boxes_out = boxes[0].cpu().numpy().tolist() if len(boxes) > 0 else []
        boxes3d_out = boxes3d[0].cpu().numpy().tolist() if len(boxes3d) > 0 else []
        scores_3d_out = scores_3d[0].cpu().numpy().tolist() if len(scores_3d) > 0 else []

        response_data = {
            "success": True, 
            "boxes": boxes_out,
            "boxes3d": boxes3d_out,
            "scores_3d": scores_3d_out
        }

        if return_vis and len(boxes3d_out) > 0:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                
            try:
                draw_3d_boxes(
                    image=image_np.astype(np.uint8),
                    boxes3d=boxes3d[0],
                    intrinsics=intrinsics_pixel,
                    scores_2d=scores_2d[0] if len(scores_2d) > 0 else None,
                    scores_3d=scores_3d[0] if len(scores_3d) > 0 else None,
                    class_ids=class_ids[0] if len(class_ids) > 0 else None,
                    save_path=temp_path,
                )
                
                vis_img = cv2.imread(temp_path)
                if vis_img is not None:
                    _, buffer = cv2.imencode('.png', vis_img)
                    response_data["vis_image"] = base64.b64encode(buffer.tobytes()).decode('utf-8')
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return Response(json.dumps(response_data, sort_keys=False), mimetype='application/json')
        
    except Exception as e:
        logger.error(f"Inference failed: {traceback.format_exc()}")
        return jsonify({"error": f"Inference crashed: {str(e)}"}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WildDet3D Server with MoGe')
    parser.add_argument('--moge_ckpt', type=str, default='')
    parser.add_argument('--wilddet3d_ckpt', type=str, default='')
    parser.add_argument('--sam3_ckpt', type=str, default='')
    parser.add_argument('--lingbot_ckpt', type=str, default='')
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument('--port', type=int, default=20027)
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offloading to save GPU memory at the cost of inference speed.')
    
    args = parser.parse_args()
    
    if not load_models(args):
        logger.error("System exiting due to model loading failure.")
        exit(1)
        
    app.run(host='0.0.0.0', port=args.port, debug=False)