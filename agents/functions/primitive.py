import sys
import os
from pathlib import Path
from PIL import Image

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

PRIMITIVES_REGISTRY = {}

def register_primitive(func):
    PRIMITIVES_REGISTRY[func.__name__] = func
    return func

_wilddet3d_client_instance = None

def _get_wilddet3d_client():
    global _wilddet3d_client_instance
    if _wilddet3d_client_instance is None:
        from vision_experts.WildDet3D.wilddet3d_client import WildDet3DClient
        server_url = os.getenv("SPAGENT_WILDDET3D_SERVER_URL", "http://127.0.0.1:20027")
        _wilddet3d_client_instance = WildDet3DClient(server_url)
    return _wilddet3d_client_instance

def get_3d_bounding_box(image_path: str, boundingbox: list):
    """Returns the 3D bounding box parameters in camera coordinate space for an object
    specified by a 2D bounding box in pixel space. Uses the WildDet3D model internally.
    This is the foundational primitive — all higher-level spatial functions must be built on top of it.

    Args:
        image_path (str): Absolute path to the image file on disk.
        boundingbox (list): A 2D bounding box [xmin, ymin, xmax, ymax] in pixel coordinates.

    Returns:
        list: A nested list containing the 3D bounding box predictions
        [[cx, cy, cz, dx, dy, dz, qw, qx, qy, qz]], where each item is a continuous
        numerical float value defined as follows:
        - cx (index 0): Center X-coordinate in camera space (horizontal offset, left is negative, in meters).
        - cy (index 1): Center Y-coordinate in camera space (vertical offset, in meters).
        - cz (index 2): Center Z-coordinate in camera space (depth/distance directly in front of the camera, in meters).
        - dx (index 3): Spatial dimension 'Length' of the 3D bounding box (in meters).
        - dy (index 4): Spatial dimension 'Width' of the 3D bounding box (in meters).
        - dz (index 5): Spatial dimension 'Height' of the 3D bounding box (in meters).
        - qw (index 6): Scalar (real) component of the orientation unit quaternion.
        - qx (index 7): X-axis imaginary vector component of the orientation unit quaternion.
        - qy (index 8): Y-axis imaginary vector component of the orientation unit quaternion.
        - qz (index 9): Z-axis imaginary vector component of the orientation unit quaternion.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    client = _get_wilddet3d_client()
    result = client.infer(image_path, boundingbox=boundingbox, return_vis=False)

    if result is None:
        raise RuntimeError("WildDet3D server returned None.")
    
    boxes3d = result.get("boxes3d")
    if boxes3d is None:
        raise RuntimeError("Backend succeeded but 'boxes3d' key is missing.")
    return boxes3d


@register_primitive
def get_3d_position(image_path: str, boundingbox: list):
    """Returns the 3D center coordinates of an object specified by a 2D bounding box.

    Args:
        image_path (str): Absolute path to the image file on disk.
        boundingbox (list): A 2D bounding box [xmin, ymin, xmax, ymax] in pixel coordinates.

    Returns:
        list: [[cx, cy, cz]] where:
        - cx (index 0): Center X-coordinate in camera space (meters).
        - cy (index 1): Center Y-coordinate in camera space (meters).
        - cz (index 2): Center Z-coordinate / depth (meters).
    """
    boxes = get_3d_bounding_box(image_path, boundingbox)
    if not boxes or len(boxes[0]) < 3:
        return [[0.0, 0.0, 0.0]]
    return [[boxes[0][0], boxes[0][1], boxes[0][2]]]


@register_primitive
def get_3d_dimensions(image_path: str, boundingbox: list):
    """Returns the 3D dimensions (length, width, height) of an object specified by a 2D bounding box.

    Args:
        image_path (str): Absolute path to the image file on disk.
        boundingbox (list): A 2D bounding box [xmin, ymin, xmax, ymax] in pixel coordinates.

    Returns:
        list: [[dx, dy, dz]] where:
        - dx (index 0): Length (meters).
        - dy (index 1): Width (meters).
        - dz (index 2): Height (meters).
    """
    boxes = get_3d_bounding_box(image_path, boundingbox)
    if not boxes or len(boxes[0]) < 6:
        return [[0.0, 0.0, 0.0]]
    return [[boxes[0][3], boxes[0][4], boxes[0][5]]]


@register_primitive
def get_3d_orientation(image_path: str, boundingbox: list):
    """Returns the orientation quaternion of an object specified by a 2D bounding box.

    Args:
        image_path (str): Absolute path to the image file on disk.
        boundingbox (list): A 2D bounding box [xmin, ymin, xmax, ymax] in pixel coordinates.

    Returns:
        list: [[qw, qx, qy, qz]] where:
        - qw (index 0): Scalar (real) component of the orientation unit quaternion.
        - qx (index 1): X-axis imaginary component.
        - qy (index 2): Y-axis imaginary component.
        - qz (index 3): Z-axis imaginary component.
    """
    boxes = get_3d_bounding_box(image_path, boundingbox)
    if not boxes or len(boxes[0]) < 10:
        return [[0.0, 0.0, 0.0, 0.0]]
    return [[boxes[0][6], boxes[0][7], boxes[0][8], boxes[0][9]]]


_moge_client_instance = None

def _get_moge_client():
    global _moge_client_instance
    if _moge_client_instance is None:
        from vision_experts.MoGeV2.moge_client import MoGeClient
        server_url = os.getenv("SPAGENT_MOGE_SERVER_URL", "http://127.0.0.1:20021")
        _moge_client_instance = MoGeClient(
            server_url=server_url,
            prefer_cache=False,
            strict_cache=False,
        )
    return _moge_client_instance


@register_primitive
def get_point_depth(image_path: str, point: list):
    """Returns the metric depth (distance from camera in meters) of a specific point in the image.

    Args:
        image_path (str): Absolute path to the image file on disk.
        point (list): The [x, y] coordinates of the target point in absolute pixel coordinates.

    Returns:
        float: The metric depth of the point in meters from the camera.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not (isinstance(point, list) and len(point) == 2 and all(isinstance(i, (int, float)) for i in point)):
        raise ValueError("point must be a valid list of two numbers [x, y]")

    client = _get_moge_client()
    result = client.infer(image_path, [int(point[0]), int(point[1])], None)

    if result is None or not result.get('success'):
        raise RuntimeError(f"MoGe depth estimation failed: {result}")

    depth = result.get('metric_depth', result.get('metric_depth1'))
    if depth is None:
        raise RuntimeError("MoGe returned success but no depth value.")

    return depth


if __name__ == "__main__":

    image_file = ""

    # test_box = [495, 215, 885, 506]

    # result = get_3d_bounding_box(image_file, boundingbox=test_box)

    # if result:
    #     print("Success! Extracted 3D Boxes:", result)

    # Test get_point_depth
    test_point = [640, 360]
    depth = get_point_depth(image_file, point=test_point)
    print(f"Depth at point {test_point}: {depth} meters")
