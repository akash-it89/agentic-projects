"""Scan clean-up. Old newsprint is grey, uneven and slightly rotated; each step here is cheap."""

from pathlib import Path

import cv2
import numpy as np


def load_gray(path: Path) -> np.ndarray:
    # cv2.imread can't open non-ASCII paths on Windows; decoding from bytes works everywhere.
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """Angle in degrees from the dominant direction of text pixels. Small angles only; scans are rarely worse."""
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0)).astype(np.float32)
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect's angle convention differs between OpenCV versions; fold it into (-45, 45].
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(angle) if abs(angle) <= max_angle else 0.0


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enhance(gray: np.ndarray) -> np.ndarray:
    """Even out faded and blotchy areas so faint type is still detected."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def preprocess(path: Path, fix_skew: bool = True) -> np.ndarray:
    gray = load_gray(path)
    if fix_skew:
        gray = deskew(gray, estimate_skew(gray))
    return enhance(gray)
