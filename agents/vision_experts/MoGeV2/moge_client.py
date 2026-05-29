import base64
import requests
import cv2
import os
import time
import threading
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

class MoGeClient:
    def __init__(
        self,
        server_url: str = "None",
        cache_dir: Optional[str] = None,
        prefer_cache: bool = True,
        strict_cache: bool = False,
    ):
        if server_url != "None":
            self.server_urls = self._parse_server_urls(server_url)
        else:
            self.server_urls = []
        self._rr_lock = threading.Lock()
        self._rr_index = 0
        env_cache_dir = os.getenv("SPAGENT_MOGE_CACHE_DIR", ".rgpt")
        self.cache_dir = Path(cache_dir or env_cache_dir).expanduser().resolve() if (cache_dir or env_cache_dir) else None
        self.prefer_cache = prefer_cache
        self.strict_cache = strict_cache

    def _parse_server_urls(self, server_url: str):
        candidates = [u.strip().rstrip('/') for u in str(server_url).split(',') if u.strip()]
        if not candidates:
            candidates = ["http://127.0.0.1:20021"]
        return candidates

    def _next_server_url(self) -> str:
        if len(self.server_urls) == 1:
            return self.server_urls[0]
        with self._rr_lock:
            url = self.server_urls[self._rr_index % len(self.server_urls)]
            self._rr_index += 1
        return url

    def _cache_path_for_image(self, image_path: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        image_abs = str(Path(image_path).expanduser().resolve())
        image_hash = hashlib.sha1(image_abs.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{image_hash}.npz"

    def _infer_from_cache(self, image_path: str, point1, point2=None) -> Optional[Dict[str, Any]]:
        cache_path = self._cache_path_for_image(image_path)
        if cache_path is None or not cache_path.exists():
            return None
            
        try:
            with np.load(cache_path, mmap_mode='r') as payload:
                depth_map = payload["depth"]
                
                h = int(payload["height"])
                w = int(payload["width"])
                x1 = max(0, min(int(point1[0]), w - 1))
                y1 = max(0, min(int(point1[1]), h - 1))
                
                response: Dict[str, Any] = {
                    "success": True,
                    "shape": [h, w],
                    "from_cache": True,
                }

                if point2 is None:
                    response["metric_depth"] = round(float(depth_map[y1, x1]), 4)
                    return response

                points_map = payload["points"]
                
                x2 = max(0, min(int(point2[0]), w - 1))
                y2 = max(0, min(int(point2[1]), h - 1))

                depth_val1 = float(depth_map[y1, x1])
                depth_val2 = float(depth_map[y2, x2])
                p1_3d = points_map[y1, x1]
                p2_3d = points_map[y2, x2]
                

                if np.all(np.isfinite(p1_3d)) and np.all(np.isfinite(p2_3d)):
                    dist = np.linalg.norm(p1_3d - p2_3d)
                    dist_result = round(float(dist), 4)
                else:
                    dist_result = 1000

                depth1_result = round(depth_val1, 4) if np.isfinite(depth_val1) else None
                depth2_result = round(depth_val2, 4) if np.isfinite(depth_val2) else None

                response["metric_depth1"] = depth1_result
                response["metric_depth2"] = depth2_result
                response["metric_distance"] = dist_result
                
                return response
                
        except Exception as e:
            print(f"Error reading cache {cache_path}: {e}")
            return None
    
    def infer(self, image_path, point1, point2=None):

        try:
            if not os.path.exists(image_path):
                return None

            if self.prefer_cache:
                cached_result = self._infer_from_cache(image_path, point1, point2)
                if cached_result is not None:
                    return cached_result
                if self.strict_cache:
                    return {
                        "success": False,
                        "error": f"Cache miss for image: {image_path}",
                    }

            if not self.server_urls:
                return None

            image = cv2.imread(image_path)
            if image is None:
                return None
            
            _, buffer = cv2.imencode('.jpg', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            data = {
                'image': image_b64,
                'point1': point1
            }
            if point2 is not None:
                data['point2'] = point2
            
            last_exception = None
            for _ in range(len(self.server_urls)):
                target_url = self._next_server_url()
                try:
                    response = requests.post(
                        f'{target_url}/infer',
                        json=data,
                        headers={'Content-Type': 'application/json'},
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    if result.get('success'):
                        if 'vis_path' in result:
                            del result['vis_path']
                        return result
                except Exception as e:
                    last_exception = e
                    continue
            if last_exception is not None:
                return None
            
            return None
                
        except Exception:
            return None