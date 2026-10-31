"""Printed rules between articles and columns: the strongest "different article" signal on old pages."""

import cv2
import numpy as np

from .geometry import Box, x_overlap_ratio


def detect_separators(gray: np.ndarray, min_len_ratio: float = 1 / 30) -> tuple[list[Box], list[Box]]:
    """Returns (horizontal_rules, vertical_rules) as boxes. A rule is a long, thin run of dark pixels."""
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    h, w = gray.shape

    def rules(kernel: tuple[int, int], horizontal: bool) -> list[Box]:
        mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        for x, y, bw_, bh, _area in stats[1:n]:
            long_side, short_side = (bw_, bh) if horizontal else (bh, bw_)
            if short_side <= 12 and long_side >= 8 * short_side:
                out.append((float(x), float(y), float(x + bw_), float(y + bh)))
        return out

    return (rules((max(10, int(w * min_len_ratio)), 1), True),
            rules((1, max(10, int(h * min_len_ratio))), False))


def rule_between_vertically(upper: Box, lower: Box, horizontal_rules: list[Box], tol: float = 4.0) -> bool:
    """True if a horizontal rule lies in the gap between two stacked boxes and spans their width."""
    top, bottom = upper[3] - tol, lower[1] + tol
    for r in horizontal_rules:
        mid = (r[1] + r[3]) / 2
        if top <= mid <= bottom and (x_overlap_ratio(r, upper) > 0.5 or x_overlap_ratio(r, lower) > 0.5):
            return True
    return False
