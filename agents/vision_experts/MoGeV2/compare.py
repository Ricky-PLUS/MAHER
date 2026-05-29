import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from moge_client import MoGeClient

def load_json_data(json_path: Path) -> List[Dict[str, Any]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            return data
    except json.JSONDecodeError:
        data = []
        with json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

def try_resolve_image_path(image_ref: str, candidate_roots: List[Optional[Path]]) -> Optional[Path]:
    p = Path(image_ref)
    if p.is_absolute() and p.exists():
        return p.resolve()
    for root in candidate_roots:
        if root is None:
            continue
        candidate = (root / p).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None

def get_image_refs(obj: Dict[str, Any]) -> List[str]:
    file_path = obj.get("file_path")
    if file_path and isinstance(file_path, str):
        return [file_path]
    return []
    
def round4_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return round(float(v), 4)
    except Exception:
        return None

def is_within_range(online_val: float, cached_val: float) -> bool:
    if online_val == 0.0:
        return cached_val == 0.0
    lower_bound = min(0.9 * online_val, 1.1 * online_val)
    upper_bound = max(0.9 * online_val, 1.1 * online_val)
    return lower_bound <= cached_val <= upper_bound

def main():
    parser = argparse.ArgumentParser(description="Compare MoGe online /infer vs cache read for ALL items.")
    parser.add_argument("--json", type=str, default="", help="Training json to sample from.")
    parser.add_argument("--image_root", type=str, default="", help="Root to resolve relative image paths.")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:20021", help="MoGe server URL.")
    parser.add_argument("--cache_dir", type=str, default="", help="MoGe cache dir (defaults to env SPAGENT_MOGE_CACHE_DIR).")

    parser.add_argument("--output_json", type=str, default="comparison_results3.json", help="Path to save the summary and detailed comparison results.")
    args = parser.parse_args()
    
    json_path = Path(args.json).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
        
    cache_dir = args.cache_dir or os.getenv("SPAGENT_MOGE_CACHE_DIR")
    if not cache_dir:
        raise ValueError("cache_dir is required (set --cache_dir or env SPAGENT_MOGE_CACHE_DIR).")
        
    cache_dir_path = Path(cache_dir).expanduser().resolve()
    json_dir = json_path.parent
    cwd = Path.cwd().resolve()
    
    candidate_roots: List[Optional[Path]] = [
        Path(args.image_root).expanduser().resolve() if args.image_root else None,
        json_dir,
        cwd,
        Path(os.getenv("WORKSPACE", "")).expanduser().resolve(),
    ]

    data_list = load_json_data(json_path)
    if not data_list:
        raise ValueError("No data found in json.")
        
    client_online = MoGeClient(server_url=args.server_url, cache_dir=str(cache_dir_path), prefer_cache=False)
    client_cache = MoGeClient(
        server_url=args.server_url,
        cache_dir=str(cache_dir_path),
        prefer_cache=True,
        strict_cache=True,
    )
    
    ok_depth = 0
    fail_depth = 0
    ok_dist = 0
    fail_dist = 0
    
    total_time_online = 0.0
    total_time_cache = 0.0

    fixed_depth_point = [50.0, 50.0]
    fixed_dist_point1 = [20.0, 100.0]
    fixed_dist_point2 = [100.0, 20.0]

    detailed_results = []

    print(f"Starting comparison for ALL {len(data_list)} items in the json file...")

    for idx, obj in enumerate(data_list, start=1):
        if idx % 100 == 0:
            print(f"Progress: {idx} / {len(data_list)} processed...")

        image_refs = get_image_refs(obj)
        
        record = {
            "index": idx,
            "image_refs": image_refs,
            "image_path": None,
            "depth_status": None,
            "dist_status": None
        }

        if not image_refs:
            record["status"] = "SKIP_NO_IMAGES"
            detailed_results.append(record)
            continue
            
        image_path = None
        for ref in image_refs:
            image_path = try_resolve_image_path(ref, candidate_roots)
            if image_path is not None:
                break
                
        if image_path is None:
            record["status"] = "FAIL_RESOLVE_PATH"
            detailed_results.append(record)
            continue

        record["image_path"] = str(image_path)
        record["status"] = "PROCESSED"

        t0 = time.time()
        online_depth = client_online.infer(str(image_path), point1=fixed_depth_point, point2=None)
        total_time_online += time.time() - t0

        t0 = time.time()
        cached_depth = client_cache.infer(str(image_path), point1=fixed_depth_point, point2=None)
        total_time_cache += time.time() - t0

        online_depth_val = round4_or_none(online_depth.get("metric_depth") if online_depth else None)
        cached_depth_val = round4_or_none(cached_depth.get("metric_depth") if cached_depth else None)
        
        record["depth_details"] = {
            "online": online_depth_val,
            "cached": cached_depth_val
        }

        if online_depth_val is None or cached_depth_val is None:
            fail_depth += 1
            record["depth_status"] = "FAIL_NONE"
        else:
            if is_within_range(online_depth_val, cached_depth_val):
                ok_depth += 1
                record["depth_status"] = "OK"
            else:
                fail_depth += 1
                record["depth_status"] = "MISMATCH"

        t0 = time.time()
        online_dist = client_online.infer(str(image_path), point1=fixed_dist_point1, point2=fixed_dist_point2)
        total_time_online += time.time() - t0

        t0 = time.time()
        cached_dist = client_cache.infer(str(image_path), point1=fixed_dist_point1, point2=fixed_dist_point2)
        total_time_cache += time.time() - t0

        online_dist_val = round4_or_none(online_dist.get("metric_distance") if online_dist else None)
        cached_dist_val = round4_or_none(cached_dist.get("metric_distance") if cached_dist else None)

        record["dist_details"] = {
            "online": online_dist_val,
            "cached": cached_dist_val
        }

        if online_dist_val is None or cached_dist_val is None:
            fail_dist += 1
            record["dist_status"] = "FAIL_NONE"
        else:
            if is_within_range(online_dist_val, cached_dist_val):
                ok_dist += 1
                record["dist_status"] = "OK"
            else:
                fail_dist += 1
                record["dist_status"] = "MISMATCH"

        detailed_results.append(record)

    speedup = total_time_online / total_time_cache if total_time_cache > 0 else 0.0

    final_report = {
        "summary": {
            "total_items_processed": len(data_list),
            "depth_validation": {
                "ok": ok_depth,
                "fail": fail_depth
            },
            "distance_validation": {
                "ok": ok_dist,
                "fail": fail_dist
            }
        },
        "performance": {
            "total_time_online_sec": round(total_time_online, 4),
            "total_time_cache_sec": round(total_time_cache, 4),
            "speedup_multiplier": round(speedup, 2)
        },
        "detailed_results": detailed_results
    }

    out_json_path = Path(args.output_json).resolve()
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n==== Execution Completed ====")
    print(f"Results successfully saved to: {out_json_path}")
    print(f"Total Processed: {len(data_list)}")
    print(f"Depth OK: {ok_depth} | FAIL: {fail_depth}")
    print(f"Dist  OK: {ok_dist} | FAIL: {fail_dist}")
    print(f"online {total_time_online}")
    print(f"cache {total_time_cache}")
    print(f"Speedup: {speedup:.2f}x faster using cache.")


if __name__ == "__main__":
    main()