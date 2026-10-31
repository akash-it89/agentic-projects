from typing import Protocol


class LLM(Protocol):
    """Anything with this method can answer questions: Ollama, Bedrock, or a fake in tests."""

    name: str

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str: ...
