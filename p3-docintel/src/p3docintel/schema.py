"""The common schema. Every OCR engine writes Lines; the UI and every later stage read only these models.

Boxes are (x0, y0, x1, y1) normalized to 0..1 of the page, so they work at any display size.
"""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]


def union(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / total if total > 0 else 0.0


class Line(BaseModel):
    id: str  # "p1-L12": page 1, line 12. The LLM cites these.
    page: int
    text: str
    bbox: BBox
    confidence: float = 1.0
    engine: Literal["pdf_text", "paddle", "textract", "paddle+vlm", "textract+vlm"]


class ExtractedField(BaseModel):
    name: str  # snake_case, e.g. "licence_number"
    value: str
    type: str = "text"  # date | id | name | address | amount | phone | email | text
    line_ids: list[str]
    confidence: float = 1.0
    attributes: dict = Field(default_factory=dict)
    page: int | None = None
    bbox: BBox | None = None  # computed by code from line_ids, never written by the LLM


class Page(BaseModel):
    number: int
    width: int  # rendered PNG size in pixels
    height: int
    source: Literal["pdf_text", "ocr"]


class DocResult(BaseModel):
    doc_id: str
    filename: str
    status: Literal["processing", "done", "error"] = "processing"
    stage: str = "queued"
    error: str | None = None
    language: str = ""
    pages: list[Page] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)
    fields: list[ExtractedField] = Field(default_factory=list)
    doc_type: str = ""
    doc_type_confidence: float = 0.0
    summary: str = ""
    stats: dict = Field(default_factory=dict)


class ChatAnswer(BaseModel):
    answer: str
    found: bool
    line_ids: list[str] = Field(default_factory=list)
    boxes: list[dict] = Field(default_factory=list)  # [{"page": 1, "bbox": [...]}] for highlighting
