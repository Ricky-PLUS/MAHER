import base64
import requests
import cv2
import os
import time
import json
import threading
import numpy as np

class Sam3Client:
    def __init__(self, server_url):
        """
        Initialize the SAM3 client.
        
        Args:
            server_url: The address of the server, e.g., 'http://127.0.0.1:20023'
        """
        self.server_urls = self._parse_server_urls(server_url)
        self._rr_lock = threading.Lock()
        self._rr_index = 0

    def _parse_server_urls(self, server_url):
        candidates = [u.strip().rstrip('/') for u in str(server_url).split(',') if u.strip()]
        if not candidates:
            candidates = ["http://127.0.0.1:20023"]
        return candidates

    def _next_server_url(self):
        if len(self.server_urls) == 1:
            return self.server_urls[0]
        with self._rr_lock:
            url = self.server_urls[self._rr_index % len(self.server_urls)]
            self._rr_index += 1
        return url
    
    def infer(self, image_path, text_prompts=None, boundingboxes=None, return_mask=False):
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

            data = {
                'image': image_b64,
                'return_mask': return_mask
            }

            if text_prompts is not None:
                if isinstance(text_prompts, str):
                    text_prompts = [text_prompts]
                data['text_prompts'] = text_prompts

            if boundingboxes is not None:
                if len(boundingboxes) == 4 and isinstance(boundingboxes[0], (int, float)):
                    boundingboxes = [boundingboxes]
                data['boundingboxes'] = boundingboxes

            if 'text_prompts' not in data and 'boundingboxes' not in data:
                print("[Error] You must provide either text_prompts or boundingboxes.")
                return None

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