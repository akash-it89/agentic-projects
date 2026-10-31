"""OCR engines for scans and photos. Each returns Lines with boxes normalized to the page image."""

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from PIL import Image

from .config import Settings, get_settings
from .schema import Line

log = logging.getLogger(__name__)

# PaddleOCR recognition model per language, as PaddleOCR 3.4 itself maps them. They must be named
# explicitly: once a detection model name is passed, PaddleOCR ignores `lang` and falls back to a
# generic recognizer, which would silently read Tamil and Arabic with the wrong model.
PADDLE_REC_MODEL = {"en": "en_PP-OCRv5_mobile_rec", "ta": "ta_PP-OCRv5_mobile_rec",
                    "ar": "arabic_PP-OCRv5_mobile_rec"}
TEXTRACT_LANGS = {"en"}  # Textract doesn't read Tamil or Arabic


class OCREngine(Protocol):
    name: str

    def read(self, png: Path, page_no: int) -> list[Line]: ...


def _norm_box(x0, y0, x1, y1, w, h):
    return tuple(round(min(1.0, max(0.0, v)), 5) for v in (x0 / w, y0 / h, x1 / w, y1 / h))


class PaddleEngine:
    name = "paddle"

    def __init__(self, s: Settings, lang: str):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise RuntimeError("Scans and photos need OCR, which isn't installed. "
                               "Run `uv sync --group ocr`, then restart the app.") from None

        # PaddleOCR 3.x argument names (tested with 3.4). Check its docs if a later release renames them.
        self.ocr = PaddleOCR(device=s.ocr_device,
                             text_detection_model_name=s.det_model,
                             text_recognition_model_name=PADDLE_REC_MODEL.get(lang, PADDLE_REC_MODEL["en"]),
                             use_doc_orientation_classify=True,  # phone photos are often rotated
                             use_doc_unwarping=False, use_textline_orientation=False,
                             text_det_limit_type="max", text_det_limit_side_len=4000)

    def read(self, png: Path, page_no: int) -> list[Line]:
        with Image.open(png) as img:
            w, h = img.size
        res = self.ocr.predict(str(png))[0]
        out = []
        for box, text, score in zip(res["rec_boxes"], res["rec_texts"], res["rec_scores"], strict=True):
            if text.strip():
                x0, y0, x1, y1 = (float(v) for v in box)
                out.append(Line(id=f"p{page_no}-L{len(out) + 1}", page=page_no, text=text.strip(),
                                bbox=_norm_box(x0, y0, x1, y1, w, h), confidence=round(float(score), 4),
                                engine="paddle"))
        return out


class TextractEngine:
    """AWS Textract DetectDocumentText (the cheapest Textract call; lines only). Responses are cached on disk
    by file hash, so re-processing the same page never costs twice."""

    name = "textract"

    def __init__(self, s: Settings):
        import boto3

        self.client = boto3.client("textract", region_name=s.aws_region)
        self.cache = s.cache_dir / "textract"
        self.cache.mkdir(parents=True, exist_ok=True)

    def read(self, png: Path, page_no: int) -> list[Line]:
        data = png.read_bytes()
        cached = self.cache / f"{hashlib.sha256(data).hexdigest()}.json"
        if cached.exists():
            resp = json.loads(cached.read_text(encoding="utf-8"))
        else:
            resp = self.client.detect_document_text(Document={"Bytes": data})
            cached.write_text(json.dumps(resp), encoding="utf-8")
        out = []
        for b in resp["Blocks"]:
            if b["BlockType"] == "LINE":
                g = b["Geometry"]["BoundingBox"]  # already normalized: Left, Top, Width, Height
                out.append(Line(id=f"p{page_no}-L{len(out) + 1}", page=page_no, text=b["Text"],
                                bbox=_norm_box(g["Left"], g["Top"], g["Left"] + g["Width"], g["Top"] + g["Height"],
                                               1, 1),
                                confidence=round(b["Confidence"] / 100, 4), engine="textract"))
        return out


@lru_cache
def get_engine(engine: str, lang: str) -> OCREngine:
    """Models load once per (engine, language) and stay in memory for later uploads."""
    s = get_settings()
    if engine == "textract" and lang in TEXTRACT_LANGS:
        return TextractEngine(s)
    if engine == "textract":
        log.info("Textract doesn't read %s; using PaddleOCR for this document", lang)
    return PaddleEngine(s, lang)
