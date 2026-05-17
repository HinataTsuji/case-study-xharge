"""
Geometry utility functions for polygon math, bounding boxes, and inset operations.
"""
import math
from typing import List, Tuple, Dict


def point_in_polygon(px: float, py: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm to test if point (px, py) is inside a polygon."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def rect_inside_polygon(
    rx: float, ry: float, rw: float, rh: float,
    poly: List[Tuple[float, float]]
) -> bool:
    """Check if a rectangle (including centre) is fully inside a polygon."""
    corners = [
        (rx, ry),
        (rx + rw, ry),
        (rx + rw, ry + rh),
        (rx, ry + rh),
        (rx + rw / 2, ry + rh / 2),
    ]
    return all(point_in_polygon(cx, cy, poly) for cx, cy in corners)


def rect_overlaps_rect(
    ax: float, ay: float, aw: float, ah: float,
    bx: float, by: float, bw: float, bh: float,
) -> bool:
    """AABB overlap test."""
    return not (ax + aw <= bx or ax >= bx + bw or ay + ah <= by or ay >= by + bh)


def polygon_area(pts: List[Tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    area = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2


def bounding_box(pts: List[Tuple[float, float]]) -> Dict[str, float]:
    """Return bounding box of polygon points."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def inset_polygon(pts: List[Tuple[float, float]], d: float) -> List[Tuple[float, float]]:
    """
    Inset (shrink) a polygon by distance d using averaged edge normals.
    Simplified Minkowski offset.
    """
    if len(pts) < 3 or d <= 0:
        return pts

    n = len(pts)

    # Determine winding direction
    winding_sum = 0.0
    for i in range(n):
        j = (i + 1) % n
        winding_sum += (pts[j][0] - pts[i][0]) * (pts[j][1] + pts[i][1])
    clockwise = winding_sum > 0

    result = []
    for i in range(n):
        prev = pts[(i - 1 + n) % n]
        curr = pts[i]
        nxt = pts[(i + 1) % n]

        e1x = curr[0] - prev[0]
        e1y = curr[1] - prev[1]
        e2x = nxt[0] - curr[0]
        e2y = nxt[1] - curr[1]

        len1 = math.sqrt(e1x * e1x + e1y * e1y) or 1
        len2 = math.sqrt(e2x * e2x + e2y * e2y) or 1

        if clockwise:
            n1x, n1y = e1y / len1, -e1x / len1
            n2x, n2y = e2y / len2, -e2x / len2
        else:
            n1x, n1y = -e1y / len1, e1x / len1
            n2x, n2y = -e2y / len2, e2x / len2

        nx = n1x + n2x
        ny = n1y + n2y
        nl = math.sqrt(nx * nx + ny * ny) or 1
        nx /= nl
        ny /= nl

        dot = nx * n1x + ny * n1y
        scale = d / dot if dot > 0.1 else d

        result.append((curr[0] + nx * scale, curr[1] + ny * scale))

    return result
