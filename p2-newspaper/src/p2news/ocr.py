"""PaddleOCR adapter: image -> list[Line]. Everything else in the pipeline only sees Line objects,
so you can swap in Tesseract or Surya by writing another class with the same `read` method."""

from typing import Protocol

import cv2
import numpy as np

from .config import Settings, get_settings
from .geometry import polygon_to_box
from .schema import Line


class OCREngine(Protocol):
    name: str

    def read(self, gray: np.ndarray) -> list[Line]: ...


# Recognition model per language, as PaddleOCR 3.4 itself maps them.
PADDLE_REC_MODEL = {"en": "en_PP-OCRv5_mobile_rec", "ta": "ta_PP-OCRv5_mobile_rec",
                    "ar": "arabic_PP-OCRv5_mobile_rec"}


class PaddleEngine:
    name = "paddleocr"

    def __init__(self, s: Settings | None = None):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise RuntimeError("PaddleOCR isn't installed. Run `uv sync --group ocr`, or use --gold-lines "
                               "to test the layout logic without OCR.") from None

        s = s or get_settings()
        # PaddleOCR 3.x argument names (tested with 3.4). Once a detection model name is passed, PaddleOCR
        # ignores `lang`, so the recognition model per language must be named explicitly too.
        self.ocr = PaddleOCR(
            device=s.ocr_device,
            text_detection_model_name=s.det_model,
            text_recognition_model_name=PADDLE_REC_MODEL[s.ocr_lang],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_type="max",
            text_det_limit_side_len=s.det_limit_side_len,
            text_det_unclip_ratio=s.det_unclip_ratio,
            text_det_box_thresh=s.det_box_thresh,
        )

    def read(self, gray: np.ndarray) -> list[Line]:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
        res = self.ocr.predict(bgr)[0]
        texts, scores = res["rec_texts"], res["rec_scores"]
        if "rec_boxes" in res and len(res["rec_boxes"]) == len(texts):
            boxes = [tuple(float(v) for v in b) for b in res["rec_boxes"]]
        else:
            boxes = [polygon_to_box(p) for p in res["rec_polys"]]
        return [Line(box=b, text=t, score=float(sc)) for b, t, sc in zip(boxes, texts, scores, strict=True)
                if t.strip()]


class GoldEngine:
    """Returns the lines stored in a gold file. Lets you test layout + grouping without any OCR installed,
    and measures the layout logic on its own (perfect OCR, so every error is a grouping error)."""

    name = "gold-lines"

    def __init__(self, lines: list[Line]):
        self.lines = lines

    def read(self, gray: np.ndarray) -> list[Line]:
        return list(self.lines)
