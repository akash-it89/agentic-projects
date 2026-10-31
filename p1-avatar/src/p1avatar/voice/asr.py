"""Speech to text, limited to the kiosk's languages.

Whisper often misdetects the language of short clips (a 2-second "where is gate C22?" can come back
as Welsh). We pick the most likely *supported* language, and re-transcribe if Whisper picked another one.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import Settings, get_settings
from ..lang import SUPPORTED

# OpenAI-style verbose_json returns language names, not codes.
_NAME_TO_CODE = {"english": "en", "hindi": "hi", "spanish": "es"}


@dataclass
class Transcript:
    text: str
    lang: str
    lang_prob: float


def pick_language(detected: str, probs: list[tuple[str, float]] | None) -> tuple[str, float]:
    detected = _NAME_TO_CODE.get(detected, detected)
    probs = probs or []
    if detected in SUPPORTED:
        return detected, next((p for code, p in probs if code == detected), 1.0)
    allowed = [(c, p) for c, p in probs if c in SUPPORTED]
    return max(allowed, key=lambda x: x[1]) if allowed else ("en", 0.0)


class LocalASR:
    """faster-whisper in this process. Reads webm/ogg/wav directly (it decodes with PyAV)."""

    def __init__(self, s: Settings):
        from faster_whisper import WhisperModel

        device = s.resolved_device()
        self.model = WhisperModel(s.whisper_model, device=device,
                                  compute_type="float16" if device == "cuda" else "int8")

    def transcribe(self, path: Path) -> Transcript:
        segments, info = self.model.transcribe(str(path), beam_size=5, vad_filter=True)
        text = " ".join(seg.text for seg in segments).strip()
        lang, prob = pick_language(info.language, info.all_language_probs)
        if lang != info.language:
            segments, _ = self.model.transcribe(str(path), beam_size=5, vad_filter=True, language=lang)
            text = " ".join(seg.text for seg in segments).strip()
        return Transcript(text, lang, prob)


class SpeachesASR:
    """faster-whisper running in the Speaches container (OpenAI-compatible /v1/audio/transcriptions)."""

    def __init__(self, s: Settings):
        from openai import OpenAI

        self.client = OpenAI(base_url=s.speaches_base_url, api_key="local")
        self.model = s.speaches_model

    def transcribe(self, path: Path) -> Transcript:
        with open(path, "rb") as f:
            r = self.client.audio.transcriptions.create(model=self.model, file=f, response_format="verbose_json")
        lang, prob = pick_language(getattr(r, "language", "en") or "en", None)
        return Transcript(r.text.strip(), lang, prob)


@lru_cache
def get_asr():
    s = get_settings()
    return SpeachesASR(s) if s.asr_backend == "speaches" else LocalASR(s)
