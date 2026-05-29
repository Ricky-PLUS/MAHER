import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Generator
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from moge.v2 import MoGeModel

NUM_WORKERS = 5
TORCH_START_METHOD = "spawn"
BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images_from_json(json_path: str, image_root: str = None, 
                          skip_exist_check: bool = False) -> Generator[Path, None, None]:
    root = Path(image_root).expanduser().resolve() if image_root else None
    jp = Path(json_path).expanduser().resolve()
    if not jp.exists():
        raise FileNotFoundError(f"JSON file not found: {jp}")

    try:
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    except json.JSONDecodeError:
        data = []
        with jp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    for item_idx, obj in enumerate(data, 1):
        rel_path = obj.get("file_path")
        
        if not rel_path:
            continue
            
        p = Path(rel_path)
        if root is not None and not p.is_absolute():
            p = root / p
        p = p.expanduser().resolve()
        
        if not skip_exist_check:
            if not (p.exists() and p.is_file()):
                continue
        yield p
        
        if item_idx % 1000 == 0:
            logger.debug(f"Parsed {item_idx} items from json...")


def cache_key(image_path: Path) -> str:
    return hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()


def save_cache(cache_dir: Path, image_path: Path, depth: np.ndarray, points: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / f"{cache_key(image_path)}.npz"
    h, w = depth.shape
    np.savez(
        out_file,
        source_path=str(image_path),
        height=np.int32(h),
        width=np.int32(w),
        depth=depth.astype(np.float16),
        points=points.astype(np.float16),
    )
    return out_file

def load_model(checkpoint_path: str, device: str):
    dev = torch.device(device)
    sub_logger = logging.getLogger(f"{__name__}.worker-{os.getpid()}")
    sub_logger.info("Loading MoGe model from %s on %s", checkpoint_path, dev)
    
    load_start = time.time()
    try:
        model = MoGeModel.from_pretrained(checkpoint_path).to(dev)
    except Exception:
        model = MoGeModel.from_pretrained(checkpoint_path, trust_remote_code=True).to(dev)
    load_time = time.time() - load_start
    sub_logger.info(f"Model loaded in {load_time:.2f}s")
    
    model.eval()
    return model, dev


def infer_single(model, device, image_path: Path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        output = model.infer(input_tensor)
    depth = output["depth"].detach().cpu().numpy()
    points = output["points"].detach().cpu().numpy()
    if depth.ndim == 3:
        depth = depth.squeeze(0)
    if points.shape[0] == 3:
        points = points.transpose(1, 2, 0)
    return depth, points

def _worker_init(checkpoint_path: str, device: str, cache_dir: Path, overwrite: bool):
    global _worker_model, _worker_device, _worker_cache_dir, _worker_overwrite, _worker_pid
    
    _worker_pid = os.getpid()
    sub_logger = logging.getLogger(f"{__name__}.worker-{_worker_pid}")

    actual_device = device
    if device.startswith("cuda") and ":" not in device and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = hash(os.getpid()) % num_gpus
        actual_device = f"cuda:{gpu_id}"
        sub_logger.info(f"Auto-assigned to GPU {gpu_id} (from {device})")
    
    _worker_model, _worker_device = load_model(checkpoint_path, actual_device)
    _worker_cache_dir = cache_dir
    _worker_overwrite = overwrite
    
    if actual_device.startswith("cuda"):
        gpu_idx = int(actual_device.split(":")[-1]) if ":" in actual_device else 0
        torch.cuda.set_per_process_memory_fraction(0.95, device=gpu_idx)
        torch.cuda.empty_cache()
        sub_logger.info(f"VRAM limit set to 95% on {actual_device}")


def _process_single_image(image_path_str: str) -> Dict[str, Any]:
    image_path = Path(image_path_str)
    key = cache_key(image_path)
    out_file = _worker_cache_dir / f"{key}.npz"
    
    if out_file.exists() and not _worker_overwrite:
        return {"image_path": str(image_path), "cache_path": str(out_file), "status": "skipped"}
    
    if not image_path.exists():
        return {"image_path": str(image_path), "status": "failed", "error": "File not found"}
    
    try:
        depth, points = infer_single(_worker_model, _worker_device, image_path)
        saved_path = save_cache(_worker_cache_dir, image_path, depth, points)
        return {"image_path": str(image_path), "cache_path": str(saved_path), "status": "ok"}
    except Exception as e:
        return {"image_path": str(image_path), "status": "failed", "error": f"[pid:{_worker_pid}] {str(e)}"}


def main():
    mp.set_start_method(TORCH_START_METHOD, force=True)
    
    parser = argparse.ArgumentParser(description="Precompute MoGe depth/points cache for training images.")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--skip_exist_check", default=True,
                       help="Skip pre-checking file existence (faster startup)")
    parser.add_argument("--stream_mode", default=True,
                       help="Use streaming processing with imap_unordered")

    parser.add_argument("--checkpoint_path", type=str,default="", help="Path to MoGe-v2 checkpoint.")

    parser.add_argument(
        "--json",
        type=str,
        default="",
        help="Path to the JSON file containing training data.",
    )
    parser.add_argument("--cache_dir", type=str, default="", help="Output cache directory.")

    args = parser.parse_args()

    if not args.json:
        raise ValueError("The --json parameter must be provided.")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Stage 1: Parsing json (skip_exist_check={args.skip_exist_check})...")
    parse_start = time.time()
    
    seen = set()
    unique_images: List[str] = []

    for img_path in iter_images_from_json(args.json, args.image_root, args.skip_exist_check):
        path_str = str(img_path)
        if path_str not in seen:
            seen.add(path_str)
            unique_images.append(path_str)
    
    parse_time = time.time() - parse_start
    logger.info(f"Parsed {len(unique_images)} unique images in {parse_time:.2f}s")
    
    if not unique_images:
        raise ValueError("No valid images found for precompute.")

    manifest = {"cache_dir": str(cache_dir), "items": []}
    stats = {"ok": 0, "skipped": 0, "failed": 0}
    
    if args.device.startswith("cuda") and args.num_workers > 1:
        logger.info(f"Stage 2: Starting {args.num_workers} parallel workers on {args.device}...")
        if torch.cuda.is_available():
            logger.info(f"Detected {torch.cuda.device_count()} GPU(s)")
        
        pool_start = time.time()
        
        with mp.Pool(
            processes=args.num_workers,
            initializer=_worker_init,
            initargs=(args.checkpoint_path, args.device, cache_dir, args.overwrite)
        ) as pool:
            pool_init_time = time.time() - pool_start
            logger.info(f"Process pool created in {pool_init_time:.2f}s")
            
            if args.stream_mode:
                logger.info(f"Stage 3: Streaming inference with imap_unordered...")
                
                results_iter = pool.imap_unordered(
                    _process_single_image, 
                    unique_images,
                    chunksize=max(1, BATCH_SIZE // args.num_workers)
                )
                
                for idx, res in enumerate(results_iter, start=1):
                    manifest["items"].append(res)
                    status = res["status"]
                    if status in stats:
                        stats[status] += 1
                    
                    if idx % 50 == 0 or idx == len(unique_images):
                        elapsed = time.time() - pool_start
                        rate = idx / elapsed if elapsed > 0 else 0
                        eta = (len(unique_images) - idx) / rate if rate > 0 and idx < len(unique_images) else 0
                        logger.info(
                            f"Progress: {idx}/{len(unique_images)} | "
                            f"ok={stats['ok']}, skip={stats['skipped']}, fail={stats['failed']} | "
                            f"{rate:.1f} img/s | ETA: {eta:.0f}s"
                        )
            else:
                logger.info(f"Stage 3: Batch inference (legacy mode)...")
                batch_size = 50
                for i in range(0, len(unique_images), batch_size):
                    batch = unique_images[i:i+batch_size]
                    results = pool.map(_process_single_image, batch)
                    for res in results:
                        manifest["items"].append(res)
                        if res["status"] in stats:
                            stats[res["status"]] += 1
                    processed = i + len(batch)
                    if processed % 50 == 0 or processed == len(unique_images):
                        logger.info("Progress: %s/%s (ok=%s, skipped=%s, failed=%s)", 
                                  processed, len(unique_images), stats["ok"], stats["skipped"], stats["failed"])
    else:
        logger.info("Running in single-process mode")
        model, device = load_model(args.checkpoint_path, args.device)
        global _worker_model, _worker_device, _worker_cache_dir, _worker_overwrite, _worker_pid
        _worker_model, _worker_device = model, device
        _worker_cache_dir, _worker_overwrite = cache_dir, args.overwrite
        _worker_pid = os.getpid()
        
        for idx, image_path_str in enumerate(unique_images, start=1):
            res = _process_single_image(image_path_str)
            manifest["items"].append(res)
            if res["status"] in stats:
                stats[res["status"]] += 1
            if idx % 50 == 0:
                logger.info("Progress: %s/%s (ok=%s, skipped=%s, failed=%s)", 
                          idx, len(unique_images), stats["ok"], stats["skipped"], stats["failed"])

    manifest_path = cache_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_time = time.time() - parse_start
    logger.info(f"\n🎉 Precompute finished!")
    logger.info(f"Total: {len(unique_images)} images | ok={stats['ok']}, skipped={stats['skipped']}, failed={stats['failed']}")
    logger.info(f"Total time: {total_time:.1f}s | Avg: {total_time/len(unique_images)*1000:.1f}ms/image")
    logger.info(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()