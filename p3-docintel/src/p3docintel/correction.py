"""Optional: re-read low-confidence OCR lines with a local vision model (useful for Tamil and Arabic, where
PaddleOCR's boxes are good but its text is weaker). The box stays the same; only the text changes."""

import io
import logging
from pathlib import Path

from PIL import Image

from .langdetect import NAMES
from .schema import Line

log = logging.getLogger(__name__)

PROMPT = ("This image is one line cut from a {language} document. An OCR engine read it as: {guess!r}. "
          "Return exactly the text in the image, nothing else.")


class VLMReader:
    def __init__(self, host: str, model: str):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model
        self.available = True

    def read(self, crop_png: bytes, guess: str, lang: str) -> str | None:
        if not self.available:
            return None
        try:
            r = self.client.chat(model=self.model, options={"temperature": 0}, messages=[
                {"role": "user", "content": PROMPT.format(language=NAMES.get(lang, lang), guess=guess),
                 "images": [crop_png]}])
            return r["message"]["content"].strip() or None
        except Exception as e:  # noqa: BLE001  server down, model not pulled, ... → keep OCR text
            log.warning("Vision model unavailable (%s); keeping OCR text", e)
            self.available = False
            return None


def crop_line(png: Path, bbox, pad: float = 0.004) -> bytes:
    with Image.open(png) as img:
        w, h = img.size
        x0, y0, x1, y1 = bbox
        box = (max(0, int((x0 - pad) * w)), max(0, int((y0 - pad) * h)),
               min(w, int((x1 + pad) * w)), min(h, int((y1 + pad) * h)))
        buf = io.BytesIO()
        img.crop(box).save(buf, format="PNG")
        return buf.getvalue()


def correct_lines(lines: list[Line], png: Path, lang: str, reader: VLMReader, keep_score: float) -> int:
    """Re-reads lines below keep_score in place. Returns how many were changed."""
    changed = 0
    for ln in lines:
        if ln.confidence >= keep_score or ln.engine == "pdf_text":
            continue
        text = reader.read(crop_line(png, ln.bbox), ln.text, lang)
        if text and text != ln.text:
            ln.text, ln.engine = text, f"{ln.engine}+vlm"
            changed += 1
    return changed
