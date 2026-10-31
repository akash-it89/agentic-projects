from openai import OpenAI


class OllamaLLM:
    """Ollama through its OpenAI-compatible endpoint. Works with any OpenAI-compatible server (vLLM etc.)."""

    def __init__(self, base_url: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key="ollama")  # Ollama ignores the key, the SDK needs one
        self.model = model
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        out = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (out.choices[0].message.content or "").strip()
