from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .geometry import Box


@dataclass
class Line:
    """One OCR text line."""

    box: Box
    text: str
    score: float


@dataclass
class Block:
    """Consecutive lines merged into a paragraph-like region."""

    id: int
    lines: list[Line]
    box: Box
    line_height: float
    is_headline: bool = False
    column: int = 0
    text: str = ""

    @property
    def mean_score(self) -> float:
        return sum(ln.score for ln in self.lines) / len(self.lines)


class Article(BaseModel):
    article_id: str
    bbox: Box
    text: str
    confidence: float
    block_ids: list[int]
    regions: list[Box] = Field(description="the block boxes: use these if one rectangle swallows other content")
    source_engine: str = "paddleocr"


class PageResult(BaseModel):
    page_id: str
    width: int
    height: int
    language: str
    articles: list[Article]
    stats: dict = Field(default_factory=dict)


@dataclass
class PairDecision:
    a: int
    b: int
    same_article: bool
    confidence: float
    source: str  # "rule" | "llm" | "fallback"
    features: dict = field(default_factory=dict)
