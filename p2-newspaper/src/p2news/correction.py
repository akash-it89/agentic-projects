"""Fix text only where OCR is unsure. High-confidence blocks keep PaddleOCR's text (free); low-confidence
blocks are re-read by a local vision model. Count how many get escalated: that share drives cost."""

import logging

import cv2
import numpy as np

from .grouping import dehyphenate
from .schema import Block

log = logging.getLogger(__name__)

READ_PROMPT = ("This image is a block of text cut from an old newspaper page. Transcribe exactly the text you "
               "see, line by line, without correcting spelling or adding anything. OCR guess for reference:\n{guess}")


class VLMReader:
    def __init__(self, host: str, model: str):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model
        self.available = True

    def read(self, crop: np.ndarray, guess: str) -> str | None:
        if not self.available:
            return None
        ok, png = cv2.imencode(".png", crop)
        if not ok:
            return None
        try:
            r = self.client.chat(model=self.model, options={"temperature": 0}, messages=[
                {"role": "user", "content": READ_PROMPT.format(guess=guess[:1500]), "images": [png.tobytes()]}])
            return r["message"]["content"].strip() or None
        except Exception as e:  # noqa: BLE001  any failure (server down, model not pulled) → keep OCR text
            log.warning("Vision model unavailable (%s); keeping OCR text for the rest of the run", e)
            self.available = False
            return None


def crop(gray: np.ndarray, box, pad: int = 6) -> np.ndarray:
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    return gray[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]


def correct_blocks(blocks: list[Block], gray: np.ndarray, keep_score: float,
                   reader: VLMReader | None) -> int:
    """Sets block.text for every block; returns how many were sent to the vision model."""
    escalated = 0
    for b in blocks:
        b.text = dehyphenate([ln.text for ln in b.lines])
        if reader is None or b.mean_score >= keep_score:
            continue
        better = reader.read(crop(gray, b.box), b.text)
        if better:
            b.text = dehyphenate(better.splitlines())
            escalated += 1
    return escalated
