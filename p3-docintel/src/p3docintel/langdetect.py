"""Which script is this text in? Counting characters per Unicode block is enough for en / ta / ar."""

import re

SUPPORTED = ("en", "ta", "ar")
NAMES = {"en": "English", "ta": "Tamil", "ar": "Arabic"}

_SCRIPTS = {
    "ta": re.compile(r"[஀-௿]"),
    "ar": re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"),
    "en": re.compile(r"[A-Za-z]"),
}


def detect_script(text: str) -> str | None:
    counts = {lang: len(rx.findall(text)) for lang, rx in _SCRIPTS.items()}
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang if n else None


def is_rtl(lang: str) -> bool:
    return lang == "ar"
