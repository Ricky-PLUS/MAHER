import base64
import requests
import cv2
import os
import threading

class WildDet3DClient:
    def __init__(self, server_url):
        """
        Initialize the WildDet3D client.
        
        Args:
            server_url: The address of the server, e.g., 'http://127.0.0.1:20027'
        """
        self.server_urls = self._parse_server_urls(server_url)
        self._rr_lock = threading.Lock()
        self._rr_index = 0

    def _parse_server_urls(self, server_url):
        candidates = [u.strip().rstrip('/') for u in str(server_url).split(',') if u.strip()]
        if not candidates:
            candidates = ["http://127.0.0.1:20027"]
        return candidates

    def _next_server_url(self):
        if len(self.server_urls) == 1:
            return self.server_urls[0]
        with self._rr_lock:
            url = self.server_urls[self._rr_index % len(self.server_urls)]
            self._rr_index += 1
        return url
    
    def infer(self, image_path, boundingbox, return_vis=False):
        try:
            if not os.path.exists(image_path):
                print(f"[Error] File not found: {image_path}")
                return None

            image = cv2.imread(image_path)
            if image is None:
                print(f"[Error] Cannot read image: {image_path}")
                return None
            
            _, buffer = cv2.imencode('.jpg', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            if not boundingbox:
                print("[Error] You must provide a boundingbox.")
                return None

            if isinstance(boundingbox[0], list):
                if len(boundingbox) > 1:
                    print(f"[Warning] Multiple boxes detected ({len(boundingbox)}). This model only supports ONE. Using the first box only.")
                boundingbox = boundingbox[0]

            if len(boundingbox) != 4 or not isinstance(boundingbox[0], (int, float)):
                print("[Error] Invalid format. Provide a single box like: [x_min, y_min, x_max, y_max]")
                return None

            data = {
                'image': image_b64,
                'return_vis': return_vis,
                'boundingbox': boundingbox
            }

            last_exception = None
            for _ in range(len(self.server_urls)):
                target_url = self._next_server_url()
                try:
                    response = requests.post(
                        f'{target_url}/infer',
                        json=data,
                        headers={'Content-Type': 'application/json'},
                        timeout=120
                    )
                    response.raise_for_status()
                    result = response.json()
                    if result.get('success'):
                        return result
                    print(f"[Server Error] {result.get('error')}")
                except Exception as e:
                    last_exception = e
                    continue
            
            if last_exception is not None:
                print(f"[Request Exception] {last_exception}")
            return None
                
        except Exception as e:
            print(f"[Request Exception] {e}")
            return None

if __name__ == "__main__":
    client = WildDet3DClient("http://127.0.0.1:20027")
    image_file = ""

    test_box = [495, 215, 885, 506]
    
    result = client.infer(image_file, boundingbox=test_box, return_vis=True)
    
    if result:
        print("Success! Extracted 3D Boxes:", result.get("boxes3d"))
        
        vis_b64 = result.get("vis_image")
        if vis_b64:
            import numpy as np
            img_data = base64.b64decode(vis_b64)
            vis_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite("test_client_output.png", vis_img)
            print("Visualization saved to test_client_output.png")