"""Everything language-specific lives here, so adding a language means editing one file."""

import re

SUPPORTED = ("en", "hi", "es")

NAMES = {"en": "English", "hi": "Hindi", "es": "Spanish"}

# How the LLM is told which language to answer in. Hindi needs the script spelled out,
# otherwise small models often answer in romanized Hindi, which TTS mispronounces.
LLM_LANGUAGE = {"en": "English", "hi": "Hindi, written in Devanagari script", "es": "Spanish"}

REFUSAL = {
    "en": "Sorry, I don't have that information. Please ask at the airport information desk.",
    "hi": "माफ़ कीजिए, मेरे पास यह जानकारी नहीं है। कृपया एयरपोर्ट सूचना डेस्क से संपर्क करें।",
    "es": "Lo siento, no tengo esa información. Por favor, consulte en el mostrador de información del aeropuerto.",
}

# Kokoro v1.0 voice IDs; the first one is the default. Check the Kokoro VOICES list if one is missing.
KOKORO_VOICES = {
    "en": ["af_heart", "am_michael", "bf_emma"],
    "hi": ["hf_alpha", "hf_beta", "hm_omega"],
    "es": ["ef_dora", "em_alex"],
}

# Kokoro-FastAPI language codes (a = American English, h = Hindi, e = Spanish).
KOKORO_LANG_CODE = {"en": "a", "hi": "h", "es": "e"}


def default_voice(lang: str) -> str:
    return KOKORO_VOICES.get(lang, KOKORO_VOICES["en"])[0]


def safe_lang(lang: str | None) -> str:
    return lang if lang in SUPPORTED else "en"


_GATE = re.compile(r"\b([A-Z])(\d{1,3})\b")
_MARKDOWN = re.compile(r"[*_`#>]+")


def normalize_for_tts(text: str) -> str:
    """Make text easier to speak: 'C22' -> 'C 22', and strip Markdown symbols the LLM may emit."""
    text = _MARKDOWN.sub("", text)
    text = _GATE.sub(r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
_SPANISH_MARKS = re.compile(r"[ñáéíóúü¿¡]", re.IGNORECASE)
_SPANISH_WORDS = {"el", "la", "los", "las", "de", "del", "que", "en", "es", "está", "por", "para", "una", "dónde", "donde", "cómo"}
_ENGLISH_WORDS = {"the", "is", "are", "of", "to", "and", "in", "where", "what", "how", "for", "a", "an"}


def detect_lang(text: str) -> str | None:
    """Cheap language guess for en/hi/es: Devanagari means Hindi; otherwise compare common words.

    Good enough to check "did we answer in the right language" in evals and to pick a TTS voice.
    Not a general-purpose language detector.
    """
    if not text.strip():
        return None
    devanagari = len(_DEVANAGARI.findall(text))
    latin = len(_LATIN.findall(text))
    if devanagari > latin:
        return "hi"
    if latin == 0:
        return None
    words = re.findall(r"[a-záéíóúüñ]+", text.lower())
    es = sum(w in _SPANISH_WORDS for w in words) + 2 * len(_SPANISH_MARKS.findall(text))
    en = sum(w in _ENGLISH_WORDS for w in words)
    return "es" if es > en else "en"
