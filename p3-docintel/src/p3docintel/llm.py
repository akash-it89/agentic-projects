"""LLM clients. Ollama (local, default) or Bedrock. Every caller asks for JSON and gets a dict back."""

import json
import re
from functools import lru_cache
from typing import Protocol

from .config import get_settings


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 1500) -> str: ...


class OllamaLLM:
    def __init__(self, host: str, model: str):
        from openai import OpenAI

        # Ollama's OpenAI-compatible endpoint; it ignores the key, but the SDK requires one.
        self.client = OpenAI(base_url=f"{host.rstrip('/')}/v1", api_key="ollama")
        self.model = model
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 1500) -> str:
        out = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if json_mode else None,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return out.choices[0].message.content or ""


class BedrockLLM:
    def __init__(self, model_id: str, region: str):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.name = f"bedrock:{model_id}"

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 1500) -> str:
        if json_mode:
            system += "\nRespond with a single JSON object and nothing else."
        r = self.client.converse(modelId=self.model_id, system=[{"text": system}],
                                 messages=[{"role": "user", "content": [{"text": user}]}],
                                 inferenceConfig={"temperature": 0, "maxTokens": max_tokens})
        return "".join(p.get("text", "") for p in r["output"]["message"]["content"])


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in ``` fences or add a sentence around it. Take the first object."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


@lru_cache
def get_llm(model: str | None = None) -> LLM:
    s = get_settings()
    if s.llm_provider == "bedrock":
        return BedrockLLM(s.bedrock_model_id, s.aws_region)
    return OllamaLLM(s.ollama_host, model or s.ollama_model)
