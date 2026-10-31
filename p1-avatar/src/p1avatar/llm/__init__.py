from functools import lru_cache

from ..config import get_settings
from .base import LLM


@lru_cache
def get_llm() -> LLM:
    s = get_settings()
    if s.llm_provider == "bedrock":
        from .bedrock import BedrockLLM
        return BedrockLLM(model_id=s.bedrock_model_id, region=s.aws_region)
    from .ollama import OllamaLLM
    return OllamaLLM(base_url=s.ollama_base_url, model=s.ollama_model)


__all__ = ["LLM", "get_llm"]
