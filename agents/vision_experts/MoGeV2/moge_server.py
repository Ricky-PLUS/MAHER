import base64
import cv2
import logging
import numpy as np
import torch
import os
import argparse
import json
import open3d as o3d
from typing import Dict, Any, Optional, Tuple, List
from flask import Flask, request, jsonify, Response
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moge.v2 import MoGeModel 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None
device = None


def load_model(checkpoint_path):
    global model, device
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"device：{device}...")
        logger.info(f"model path：{checkpoint_path}")
        
        try:
            model = MoGeModel.from_pretrained(checkpoint_path).to(device)
        except Exception as e:
            logger.warning(f"try trust_remote_code=True. Error: {e}")
            model = MoGeModel.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)
            
        model.eval()
        
        try:
            logger.info("Test model...")
            dummy_input = torch.randn(3, 224, 224).to(device)  
            with torch.no_grad():
                _ = model.infer(dummy_input)
            logger.info("Complete!")
        except Exception as e:
            logger.error(f"Fail: {e}")
            return False
            
        logger.info("MoGe-v2 complete loading!")
        return True
    except Exception as e:
        logger.error(f"Error loading:{e}")
        return False

@app.route('/infer', methods=['POST'])
def infer():
    global model, device
    
    if model is None:
        return jsonify({"error": "no model"}), 500
        
    try:
        data = request.get_json()
        
        if 'image' not in data or 'point1' not in data:
            return jsonify({"error": "data lack"}), 400
            
        point1 = data['point1']
        point2 = data.get('point2') 
        
        try:
            image_bytes = base64.b64decode(data['image'])
            image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return jsonify({"error": f"image: {e}"}), 400
            
        h, w = image_rgb.shape[:2]
        
        input_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        
        logger.info(f"image shape: {w}x{h}, mode: {'two points' if point2 else 'one point'}")
        with torch.no_grad():
            output = model.infer(input_tensor)
            
        depth_map = output['depth'].cpu().numpy()
        points_map = output['points'].cpu().numpy()
        
        if depth_map.ndim == 3: 
            depth_map = depth_map.squeeze(0) 
            
        if points_map.shape[0] == 3: 
            points_map = points_map.transpose(1, 2, 0)
            
        response_data = {
            "success": True,
            "shape": [h, w]
        }

        x1 = max(0, min(int(point1[0]), w - 1))
        y1 = max(0, min(int(point1[1]), h - 1))  
        if point2 is None:
            depth_val = float(depth_map[y1, x1])
            response_data["metric_depth"] = round(depth_val, 4)
            logger.info(f"point 1 [{x1},{y1}] depth: {depth_val:.4f}m")
        else:
            x2 = max(0, min(int(point2[0]), w - 1))
            y2 = max(0, min(int(point2[1]), h - 1))
            
            depth_val1 = float(depth_map[y1, x1])
            depth_val2 = float(depth_map[y2, x2])
            
            response_data["metric_depth1"] = round(depth_val1, 4)
            response_data["metric_depth2"] = round(depth_val2, 4)
            
            p1_3d = points_map[y1, x1]
            p2_3d = points_map[y2, x2]
            
            dist = np.linalg.norm(p1_3d - p2_3d)
            response_data["metric_distance"] = round(float(dist), 4)
            
            logger.info(f"point 1 depth: {depth_val1:.4f}m, point 2 depth: {depth_val2:.4f}m, distance: {dist:.4f}m")
            
        json_str = json.dumps(response_data, sort_keys=False)
        return Response(json_str, mimetype='application/json')

    except Exception as e:
        logger.error(f"Error request: {traceback.format_exc()}")
        return jsonify({"error": f"request {str(e)}"}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoGe-v2 Metric Depth Server')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='./model.pt',
                        help='Path to MoGe-v2 model checkpoint directory or .pt file')
    parser.add_argument('--port', type=int, default=20021,
                        help='Port to run the server on (default: 20021)')
    
    args = parser.parse_args()
    
    logger.info("MoGe-v2 server...")
    
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Error path: {args.checkpoint_path}")
        exit(1)
        
    if not load_model(checkpoint_path=args.checkpoint_path):
        logger.error("Error: load model")
        exit(1)
        
    app.run(host='0.0.0.0', port=args.port, debug=False)