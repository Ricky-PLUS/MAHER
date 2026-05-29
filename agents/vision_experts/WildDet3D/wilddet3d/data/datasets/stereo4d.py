"""Stereo4D 3D dataset (real stereo depth)."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import CommonKeys as K

from .coco3d import COCO3DDataset

# Stereo4D v3 depth directory (meters, 512x512 .npy files)
_STEREO4D_DEPTH_DIR = (
    "/weka/oe-training-default/weikaih/3d_boundingbox_detection"
    "/video_data/stereo4d_test/stereo4d_dataset_v3/depth"
)


def load_stereo4d_class_map(
    annotation_path: str,
) -> dict[str, int]:
    """Load class map from Stereo4D annotation file.

    Returns a mapping from category name to category ID.
    """
    cache_path = annotation_path.replace(".json", "_class_map.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    with open(annotation_path) as f:
        data = json.load(f)
    class_map = {cat["name"]: cat["id"] for cat in data["categories"]}
    with open(cache_path, "w") as f:
        json.dump(class_map, f)
    return class_map


class Stereo4D3DDataset(COCO3DDataset):
    """Stereo4D 3D dataset with real stereo depth.

    Images from Stereo4D test set with human-reviewed 3D bounding
    boxes. Depth maps are real stereo depth (meters, 512x512).

    Key differences from InTheWild3DDataset:
      - Depth is real stereo depth (meters), not estimated depth (mm).
      - All images are 512x512.
      - No confidence masking needed (stereo depth is high quality).
    """

    def __init__(
        self,
        class_map: dict[str, int],
        max_depth: float = 100.0,
        per_image_categories: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class.

        Args:
            class_map: Mapping from category name to category ID.
            max_depth: Maximum depth in meters (clip beyond this).
            per_image_categories: If True, boxes2d_names only contains
                the GT categories present in each image.
        """
        self.per_image_categories = per_image_categories

        super().__init__(
            class_map=class_map,
            det_map=class_map,
            max_depth=max_depth,
            **kwargs,
        )

    def __getitem__(self, idx: int):
        """Get single sample, optionally with per-image category filtering."""
        data_dict = super().__getitem__(idx)
        if self.per_image_categories:
            class_ids_in_img = data_dict[K.boxes2d_classes]
            if len(class_ids_in_img) > 0:
                unique_global_ids = sorted(set(class_ids_in_img.tolist()))
                data_dict[K.boxes2d_names] = [
                    self.categories[gid] for gid in unique_global_ids
                ]
            else:
                data_dict[K.boxes2d_names] = []
        return data_dict

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """Return path to the .npy stereo depth file for this image.

        Derives the depth filename from the image file_path.
        """
        file_path = img.get("file_path", "")
        if not file_path:
            return None
        stem = os.path.splitext(os.path.basename(file_path))[0]
        depth_path = os.path.join(_STEREO4D_DEPTH_DIR, f"{stem}.npy")
        return depth_path if os.path.exists(depth_path) else None

    def get_depth_map(self, sample: DictStrAny) -> np.ndarray:
        """Load stereo depth .npy (meters, 512x512).

        No mm-to-meters conversion needed (already in meters).
        Resize to original resolution if needed.
        """
        depth = np.load(sample["depth_filename"])  # (H, W) float32, meters

        orig_h = sample["img"]["height"]
        orig_w = sample["img"]["width"]

        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(
                depth,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Clip to max_depth
        depth[depth > self.max_depth] = 0.0

        return depth.astype(np.float32)
