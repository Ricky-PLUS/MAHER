"""Dynamically generated VADAR additional functions."""

import math
from agents.functions.primitive import *

def _compare_depth_to_camera(image_path: str, bbox1: list, bbox2: list) -> int:

    pos1 = get_3d_position(image_path, bbox1)
    pos2 = get_3d_position(image_path, bbox2)
    cz1 = pos1[0][2]
    cz2 = pos2[0][2]
    if cz1 < cz2:
        return 0
    elif cz2 < cz1:
        return 1
    else:
        return -1

def _compare_point_depth(image_path: str, point_a: list, point_b: list) -> int:

    depth_a = get_point_depth(image_path, point_a)
    depth_b = get_point_depth(image_path, point_b)
    if depth_a < depth_b:
        return 0
    elif depth_b < depth_a:
        return 1
    else:
        return -1

def _get_3d_distance(image_path: str, bbox1: list, bbox2: list) -> float:

    import math

    pos1 = get_3d_position(image_path, bbox1)
    pos2 = get_3d_position(image_path, bbox2)

    if pos1 is None or pos2 is None:
        return 0.0

    cx1, cy1, cz1 = pos1[0]
    cx2, cy2, cz2 = pos2[0]

    dx = cx2 - cx1
    dy = cy2 - cy1
    dz = cz2 - cz1

    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    return distance

def _get_horizontal_distance(image_path: str, bbox1: list, bbox2: list) -> float:

    pos1 = get_3d_position(image_path, bbox1)
    pos2 = get_3d_position(image_path, bbox2)
    if pos1 is None or pos2 is None:
        return 0.0
    cx1 = pos1[0][0]
    cx2 = pos2[0][0]
    return abs(cx1 - cx2)

def _get_shortest_distance_between_objects(image_path: str, boundingbox1: list, boundingbox2: list) -> float:

    import math

    pos1_res = get_3d_position(image_path, boundingbox1)
    dim1_res = get_3d_dimensions(image_path, boundingbox1)

    pos2_res = get_3d_position(image_path, boundingbox2)
    dim2_res = get_3d_dimensions(image_path, boundingbox2)

    cx1, cy1, cz1 = pos1_res[0]
    dx1, dy1, dz1 = dim1_res[0]
    
    cx2, cy2, cz2 = pos2_res[0]
    dx2, dy2, dz2 = dim2_res[0]
    
    dist_x = max(0.0, abs(cx1 - cx2) - (dx1 + dx2) / 2.0)
    dist_y = max(0.0, abs(cy1 - cy2) - (dy1 + dy2) / 2.0)
    dist_z = max(0.0, abs(cz1 - cz2) - (dz1 + dz2) / 2.0)

    shortest_distance = math.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
    
    return shortest_distance

def _get_object_height(image_path: str, boundingbox: list) -> float:

    result = get_3d_dimensions(image_path, boundingbox)
    if result is None or len(result) == 0:
        return 0.0
    return result[0][2]

def _get_object_width(image_path: str, boundingbox: list) -> float:


    dims = get_3d_dimensions(image_path, boundingbox)
    if dims is None or len(dims) == 0:
        return 0.0
    return dims[0][1]

def _get_point_depth_meters(image_path: str, point: list) -> float:

    if point is None or len(point) != 2:
        return 0.0
    x, y = point
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return 0.0
    return get_point_depth(image_path, [float(x), float(y)])

def _get_vertical_distance(image_path: str, bbox1: list, bbox2: list) -> float:


    pos1 = get_3d_position(image_path, bbox1)
    pos2 = get_3d_position(image_path, bbox2)
    if not pos1 or not pos2 or len(pos1[0]) < 2 or len(pos2[0]) < 2:
        return 0.0
    cy1 = pos1[0][1]
    cy2 = pos2[0][1]
    return abs(cy1 - cy2)
