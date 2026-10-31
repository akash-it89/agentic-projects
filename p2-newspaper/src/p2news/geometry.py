"""Box helpers. A box is (x0, y0, x1, y1) in page pixels, origin top-left."""

from collections.abc import Iterable

Box = tuple[float, float, float, float]


def union(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def width(b: Box) -> float:
    return max(0.0, b[2] - b[0])


def height(b: Box) -> float:
    return max(0.0, b[3] - b[1])


def area(b: Box) -> float:
    return width(b) * height(b)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    total = area(a) + area(b) - inter
    return inter / total if total else 0.0


def x_overlap_ratio(a: Box, b: Box) -> float:
    """Horizontal overlap as a share of the narrower box: 1.0 means one sits fully above the other."""
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / max(1.0, min(width(a), width(b)))


def vertical_gap(upper: Box, lower: Box) -> float:
    """Distance from the bottom of `upper` to the top of `lower` (negative if they overlap)."""
    return lower[1] - upper[3]


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def contains_point(b: Box, p: tuple[float, float]) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def polygon_to_box(points: Iterable[Iterable[float]]) -> Box:
    pts = [tuple(p) for p in points]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
