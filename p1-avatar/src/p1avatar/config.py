from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All settings come from environment variables or .env (names are case-insensitive)."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data" / "airport"
    runs_dir: Path = PROJECT_ROOT / "runs"
    device: Literal["auto", "cuda", "cpu"] = "auto"

    # Retrieval
    qdrant_url: str = "http://localhost:6333"
    collection: str = "airport"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    retrieve_k: int = 20
    top_passages: int = 4
    min_score: float = 0.35

    # Answer LLM
    llm_provider: Literal["ollama", "bedrock"] = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    bedrock_model_id: str = "us.amazon.nova-lite-v1:0"
    aws_region: str = "us-east-1"

    # Voice
    asr_backend: Literal["local", "speaches"] = "local"
    whisper_model: str = "large-v3"
    speaches_base_url: str = "http://localhost:8000/v1"
    speaches_model: str = "Systran/faster-whisper-large-v3"
    kokoro_base_url: str = "http://localhost:8880/v1"

    # Avatar
    avatar_enabled: bool = True
    ditto_url: str = "http://localhost:8090"

    # Real-time agent
    agent_mode: Literal["cascaded", "nova"] = "cascaded"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def media_dir(self) -> Path:
        return self.runs_dir / "media"


@lru_cache
def get_settings() -> Settings:
    return Settings()
