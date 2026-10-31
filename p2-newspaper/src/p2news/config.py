import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings come from environment variables or .env (names are case-insensitive)."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    runs_dir: Path = PROJECT_ROOT / "runs"
    # Not called LANG: that name is already a system variable on macOS/Linux (e.g. en_US.UTF-8).
    ocr_lang: Literal["en", "ta", "ar"] = "en"
    ocr_device: Literal["cpu", "gpu"] = "cpu"

    # PaddleOCR detection. The mobile model is several times faster on CPU; compare accuracy with
    # PP-OCRv5_server_det on your own pages before choosing.
    det_model: str = "PP-OCRv5_mobile_det"
    # PaddlePaddle's oneDNN CPU acceleration. Unset = off on Windows (a known PaddlePaddle 3.x bug raises
    # NotImplementedError there), on elsewhere. Set OCR_MKLDNN=true/false to force it.
    ocr_mkldnn: bool | None = None
    det_limit_side_len: int = 4000
    det_unclip_ratio: float = 1.5
    det_box_thresh: float = 0.6

    # Lines -> blocks -> columns
    merge_gap_k: float = 0.8
    merge_overlap_min: float = 0.5
    size_tolerance: float = 0.3
    headline_ratio: float = 1.3
    column_gap_ratio: float = 0.05

    # Article decider
    decider: Literal["rules", "cascade"] = "cascade"
    ollama_host: str = "http://localhost:11434"
    decider_model: str = "qwen2.5:3b"

    # Text correction
    correction: Literal["none", "vlm"] = "none"
    vlm_model: str = "qwen2.5vl:7b"
    keep_score: float = 0.9

    def resolved_ocr_mkldnn(self) -> bool:
        return self.ocr_mkldnn if self.ocr_mkldnn is not None else sys.platform != "win32"

    @property
    def rtl(self) -> bool:
        return self.ocr_lang == "ar"


@lru_cache
def get_settings() -> Settings:
    return Settings()
