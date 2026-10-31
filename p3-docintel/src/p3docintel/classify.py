"""Document type (a small, cheap model is enough: the trainer's advice) and a short summary."""

import logging

from .langdetect import NAMES
from .llm import LLM, parse_json
from .schema import Line

log = logging.getLogger(__name__)

DOC_TYPES = ("driving_licence", "national_id", "passport", "utility_bill", "housing_document", "lease_agreement",
             "invoice", "receipt", "bank_statement", "payslip", "pamphlet", "letter", "form", "certificate", "other")

CLASSIFY_SYSTEM = f"""Classify the document into exactly one of these types: {", ".join(DOC_TYPES)}.
Return JSON: {{"doc_type": "<one of the types>", "confidence": <0 to 1>, "reason": "<a few words>"}}"""

SUMMARY_SYSTEM = """Summarise the document for someone who hasn't read it: what it is, who it concerns, and the key \
facts (dates, amounts, obligations). Use only information in the text.
Return JSON: {"summary": "<3 to 5 sentences in English>"%s}"""


def document_text(lines: list[Line], max_lines: int) -> str:
    return "\n".join(ln.text for ln in lines[:max_lines])


def classify(lines: list[Line], llm: LLM, max_lines: int = 80) -> tuple[str, float]:
    try:
        data = parse_json(llm.complete(CLASSIFY_SYSTEM, document_text(lines, max_lines), max_tokens=200))
    except ValueError as e:
        log.warning("Classification returned unusable JSON: %s", e)
        return "other", 0.0
    doc_type = str(data.get("doc_type", "")).strip().lower()
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    if doc_type not in DOC_TYPES:  # the model invented a type: don't pass that on
        return "other", 0.0
    return doc_type, round(max(0.0, min(1.0, confidence)), 3)


def summarise(lines: list[Line], llm: LLM, language: str, max_lines: int = 200) -> str:
    extra = ""
    if language != "en":
        extra = f', "summary_original": "<the same summary in {NAMES.get(language, language)}>"'
    try:
        data = parse_json(llm.complete(SUMMARY_SYSTEM % extra, document_text(lines, max_lines), max_tokens=600))
    except ValueError as e:
        log.warning("Summary returned unusable JSON: %s", e)
        return ""
    summary = str(data.get("summary", "")).strip()
    if data.get("summary_original"):
        summary += f"\n\n{str(data['summary_original']).strip()}"
    return summary
