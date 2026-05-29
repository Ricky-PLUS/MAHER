import os
import cv2
import base64
import logging
import traceback
import numpy as np
import torch
from PIL import Image
import json
from flask import Flask, request, jsonify, Response
import argparse

from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox

try:
    from pycocotools import mask as mask_utils
except ImportError:  
    os.system('pip install pycocotools -i https://mirrors.aliyun.com/pypi/simple/')
    from pycocotools import mask as mask_utils

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

model = None
processor = None 
device = None


def get_pole_of_inaccessibility_fast(binary_mask):
    if binary_mask.dtype != np.uint8: binary_mask = binary_mask.astype(np.uint8)
    rows, cols = np.nonzero(binary_mask)
    if len(rows) == 0: return None

    y_min, y_max, x_min, x_max = rows.min(), rows.max(), cols.min(), cols.max()
    roi = binary_mask[y_min:y_max+1, x_min:x_max+1]
    if roi.max() == 1: roi = roi * 255

    dist_map = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
    _, _, _, max_loc = cv2.minMaxLoc(dist_map)
    return (int(x_min + max_loc[0]), int(y_min + max_loc[1]))



def get_oriented_extreme_points(binary_mask, margin_ratio=0.05, max_margin_px=8.0):

    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
        
    binary_mask = np.where(binary_mask > 0, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    
    cx, cy = rect[0]
    v1 = box[1] - box[0]
    v2 = box[2] - box[1]
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
        
    u1 = v1 / norm1  
    u2 = v2 / norm2  

    y_coords, x_coords = np.nonzero(binary_mask)
    if len(y_coords) == 0:
        return None
        
    pts = np.column_stack((x_coords, y_coords))
    c_pts = pts - np.array([cx, cy])

    proj1 = c_pts.dot(u1)
    proj2 = c_pts.dot(u2)

    bins2 = np.round(proj2).astype(int) 
    unique_bins2 = np.unique(bins2)
    max_span1 = -1
    best_endpoints1 = None
    
    for b in unique_bins2:
        mask_b = (bins2 == b)
        p1_vals = proj1[mask_b]
        actual_pts = pts[mask_b] 
        
        if len(p1_vals) == 0:
            continue
            
        min_p1, max_p1 = np.min(p1_vals), np.max(p1_vals)
        span = max_p1 - min_p1
        
        if span > max_span1:
            max_span1 = span
            margin = min(span * margin_ratio, max_margin_px)
            safe_min_p1 = min_p1 + margin
            safe_max_p1 = max_p1 - margin
            
            idx_min = np.argmin(np.abs(p1_vals - safe_min_p1))
            idx_max = np.argmin(np.abs(p1_vals - safe_max_p1))

            best_endpoints1 = (actual_pts[idx_min], actual_pts[idx_max])

    bins1 = np.round(proj1).astype(int)
    unique_bins1 = np.unique(bins1)
    max_span2 = -1
    best_endpoints2 = None
    
    for b in unique_bins1:
        mask_b = (bins1 == b)
        p2_vals = proj2[mask_b]
        actual_pts = pts[mask_b]
        
        if len(p2_vals) == 0:
            continue
            
        min_p2, max_p2 = np.min(p2_vals), np.max(p2_vals)
        span = max_p2 - min_p2
        
        if span > max_span2:
            max_span2 = span
            margin = min(span * margin_ratio, max_margin_px)
            safe_min_p2 = min_p2 + margin
            safe_max_p2 = max_p2 - margin
            
            idx_min = np.argmin(np.abs(p2_vals - safe_min_p2))
            idx_max = np.argmin(np.abs(p2_vals - safe_max_p2))
            
            best_endpoints2 = (actual_pts[idx_min], actual_pts[idx_max])

    if best_endpoints1 is None or best_endpoints2 is None:
        return None

    dy1 = abs(best_endpoints1[0][1] - best_endpoints1[1][1])
    dy2 = abs(best_endpoints2[0][1] - best_endpoints2[1][1])

    if dy1 > dy2:
        hgt_pts = best_endpoints1
        len_pts = best_endpoints2
    else:
        hgt_pts = best_endpoints2
        len_pts = best_endpoints1

    top_pt = hgt_pts[0] if hgt_pts[0][1] < hgt_pts[1][1] else hgt_pts[1]
    bottom_pt = hgt_pts[1] if hgt_pts[0][1] < hgt_pts[1][1] else hgt_pts[0]

    left_pt = len_pts[0] if len_pts[0][0] < len_pts[1][0] else len_pts[1]
    right_pt = len_pts[1] if len_pts[0][0] < len_pts[1][0] else len_pts[0]

    return {
        "top": [int(top_pt[0]), int(top_pt[1])],
        "bottom": [int(bottom_pt[0]), int(bottom_pt[1])],
        "left": [int(left_pt[0]), int(left_pt[1])],
        "right": [int(right_pt[0]), int(right_pt[1])]
    }



def load_model(checkpoint_path):
    global model, processor, device
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading SAM3 on {device} from {checkpoint_path}")

        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        model = build_sam3_image_model(load_from_HF=False, checkpoint_path=checkpoint_path)
        model.to(device)
        model.eval()
        processor = Sam3Processor(model, confidence_threshold=0.0)
        
        logger.info("SAM3 Processor loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {traceback.format_exc()}")
        return False

@app.route('/infer', methods=['POST'])
def infer():
    global processor, device
    if processor is None: return jsonify({"error": "Processor not loaded"}), 500
    
    try:
        data = request.get_json()
        if 'image' not in data: return jsonify({"error": "Missing image"}), 400
            
        text_prompts = data.get('text_prompts', [])
        boundingboxes = data.get('boundingboxes', [])
        if not text_prompts and not boundingboxes: return jsonify({"error": "Provide text_prompts or boundingboxes"}), 400

        if isinstance(text_prompts, str): text_prompts = [text_prompts]
        if boundingboxes and isinstance(boundingboxes[0], (int, float)): boundingboxes = [boundingboxes]
            
        return_mask = data.get('return_mask', False)
        
        try:
            image_bytes = base64.b64decode(data['image'])
            image_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
        except Exception as e:
            return jsonify({"error": f"Image decoding failed: {e}"}), 400
            
        img_w, img_h = image_pil.size
        logger.info(f"Processing. Text: {text_prompts}, boundingboxes: {boundingboxes}")

        points_result, masks_result = {}, {}

        dtype = torch.bfloat16 if (device=='cuda' and torch.cuda.is_bf16_supported()) else torch.float32
        with torch.inference_mode(), torch.autocast(device, dtype=dtype):
            
            inference_state = processor.set_image(image_pil)
            
            for text in text_prompts:
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(state=inference_state, prompt=text)
                
                masks = inference_state.get("masks")
                scores = inference_state.get("scores")
                
                if masks is not None and len(scores) > 0:
                    best_idx = torch.argmax(scores)
                    best_mask_tensor = masks[best_idx]
                    
                    mask_np = best_mask_tensor.cpu().numpy().astype(np.uint8)
                    if len(mask_np.shape) == 3: mask_np = mask_np[0]
                    mask_np = np.where(mask_np > 0, 1, 0).astype(np.uint8)
                    
                    center = get_pole_of_inaccessibility_fast(mask_np)
                    extreme_points = get_oriented_extreme_points(mask_np)
                    
                    point_data = {}
                    if center:
                        point_data["center"] = list(center)
                    if extreme_points:
                        point_data.update(extreme_points)
                        
                    if point_data:
                        points_result[text] = point_data

                    if return_mask:
                        _, buffer = cv2.imencode('.png', (mask_np * 255).astype(np.uint8))
                        masks_result[text] = base64.b64encode(buffer.tobytes()).decode('utf-8')

            for bbox in boundingboxes:
                processor.reset_all_prompts(inference_state)
                
                x_min, y_min, x_max, y_max = bbox
                w = x_max - x_min
                h = y_max - y_min
                box_xywh = [x_min, y_min, w, h]
                
                box_input_cxcywh = box_xywh_to_cxcywh(torch.tensor([box_xywh]).view(-1, 4))
                norm_box_cxcywh = normalize_bbox(box_input_cxcywh, img_w, img_h).flatten().tolist()
                
                inference_state = processor.add_geometric_prompt(
                    state=inference_state, box=norm_box_cxcywh, label=True
                )
                
                masks = inference_state.get("masks")
                scores = inference_state.get("scores")
                identifier = str(bbox)
                
                if masks is not None and len(scores) > 0:
                    best_idx = torch.argmax(scores)
                    best_mask_tensor = masks[best_idx]
                    
                    mask_np = best_mask_tensor.cpu().numpy().astype(np.uint8)
                    if len(mask_np.shape) == 3: mask_np = mask_np[0]
                    mask_np = np.where(mask_np > 0, 1, 0).astype(np.uint8)
                    
                    center = get_pole_of_inaccessibility_fast(mask_np)
                    extreme_points = get_oriented_extreme_points(mask_np)
                    
                    point_data = {}
                    if center:
                        point_data["center"] = list(center)
                    if extreme_points:
                        point_data.update(extreme_points)
                        
                    if point_data:
                        points_result[identifier] = point_data
                    
                    if return_mask:
                        _, buffer = cv2.imencode('.png', (mask_np * 255).astype(np.uint8))
                        masks_result[identifier] = base64.b64encode(buffer.tobytes()).decode('utf-8')

        response_data = {"success": True, "shape": [img_h, img_w], "points": points_result}
        
        if return_mask: 
            response_data["masks"] = masks_result
        
        return Response(json.dumps(response_data, sort_keys=False), mimetype='application/json')
        
    except Exception as e:
        logger.error(f"Inference failed: {traceback.format_exc()}")
        return jsonify({"error": f"Inference crashed: {str(e)}"}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM3 Target Localization Server')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='')
    parser.add_argument('--port', type=int, default=20026)
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Model path not found: {args.checkpoint_path}")
        exit(1)
        
    if not load_model(checkpoint_path=args.checkpoint_path):
        logger.error("Failed to load model.")
        exit(1)
        
    app.run(host='0.0.0.0', port=args.port, debug=False)
    