"""Draw each article's box in its own colour, so a person can check a page in seconds."""

import colorsys
from pathlib import Path

import cv2
import numpy as np

from .schema import PageResult


def _colour(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 0.9)  # golden-ratio hues stay distinct
    return int(b * 255), int(g * 255), int(r * 255)


def draw_overlay(gray: np.ndarray, result: PageResult, out_path: Path, show_regions: bool = True) -> Path:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    thickness = max(2, img.shape[1] // 600)
    for i, a in enumerate(result.articles):
        c = _colour(i)
        if show_regions and len(a.regions) > 1:
            for r in a.regions:
                x0, y0, x1, y1 = (int(v) for v in r)
                cv2.rectangle(img, (x0, y0), (x1, y1), c, 1)
        x0, y0, x1, y1 = (int(v) for v in a.bbox)
        cv2.rectangle(img, (x0, y0), (x1, y1), c, thickness)
        cv2.putText(img, a.article_id.rsplit("-", 1)[-1], (x0 + 4, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, c, 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("could not encode overlay")
    out_path.write_bytes(buf.tobytes())  # write via bytes: cv2.imwrite fails on non-ASCII Windows paths
    return out_path
