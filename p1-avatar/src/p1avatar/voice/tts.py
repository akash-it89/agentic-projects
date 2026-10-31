from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from ..config import get_settings
from ..lang import KOKORO_LANG_CODE, KOKORO_VOICES, default_voice, normalize_for_tts, safe_lang


class KokoroTTS:
    """Kokoro-82M served by Kokoro-FastAPI (OpenAI-compatible /v1/audio/speech)."""

    def __init__(self, base_url: str):
        self.client = OpenAI(base_url=base_url, api_key="local")

    def synthesize(self, text: str, lang: str, out_path: Path, voice: str | None = None) -> Path:
        lang = safe_lang(lang)
        if voice not in KOKORO_VOICES[lang]:
            voice = default_voice(lang)  # never let an English voice read Hindi text
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice=voice,
            input=normalize_for_tts(text),
            response_format="wav",
            extra_body={"lang_code": KOKORO_LANG_CODE[lang]},
        ) as resp:
            resp.stream_to_file(out_path)
        return out_path


@lru_cache
def get_tts() -> KokoroTTS:
    return KokoroTTS(get_settings().kokoro_base_url)
