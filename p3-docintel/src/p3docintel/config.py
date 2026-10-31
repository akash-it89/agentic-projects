from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings come from environment variables or .env (names are case-insensitive)."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    runs_dir: Path = PROJECT_ROOT / "runs"

    # OCR
    ocr_engine: Literal["paddle", "textract"] = "paddle"
    ocr_device: Literal["cpu", "gpu"] = "cpu"
    det_model: str = "PP-OCRv5_mobile_det"
    render_dpi: int = 200
    min_text_layer_words: int = 5  # fewer words than this on a PDF page = treat it as a scan

    # Correction
    correction: Literal["none", "vlm"] = "none"
    vlm_model: str = "qwen2.5vl:7b"
    keep_score: float = 0.85

    # LLM
    llm_provider: Literal["ollama", "bedrock"] = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    classify_model: str = "qwen2.5:3b"
    bedrock_model_id: str = "us.amazon.nova-lite-v1:0"
    aws_region: str = "us-east-1"
    max_lines_per_call: int = 120  # long pages are sent to the LLM in slices this size

    # Chat
    retriever: Literal["bm25", "ollama"] = "bm25"
    embed_model: str = "bge-m3"
    chunk_lines: int = 5
    top_chunks: int = 6

    @property
    def docs_dir(self) -> Path:
        return self.runs_dir / "docs"

    @property
    def cache_dir(self) -> Path:
        return self.runs_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
