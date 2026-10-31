"""Field extraction. The core idea of the project:

    The LLM never produces coordinates. It cites line IDs; code turns the cited lines into a box.

So every box points at real text on the page, and a field that cites a line that doesn't exist (a
hallucination) is dropped. We also check that the value actually appears in the lines it cites.
"""

import logging
import re

from .langdetect import NAMES
from .llm import LLM, parse_json
from .schema import ExtractedField, Line, union

log = logging.getLogger(__name__)

FIELD_TYPES = ("date", "id", "name", "address", "amount", "phone", "email", "text")

SYSTEM = f"""You extract information from documents. You get the document's text lines, each with an ID \
like [p1-L3]. List every piece of information in the document as fields.

Return JSON: {{"fields": [{{"name": "snake_case_name", "value": "exact text", "type": one of {list(FIELD_TYPES)}, \
"line_ids": ["p1-L3"], "attributes": {{}}}}]}}

Rules:
- Copy "value" exactly as written in the document. Don't reformat dates or numbers.
- "line_ids" must be the IDs of the lines the value comes from. Use only IDs shown below.
- Use "attributes" for useful facts about the value, e.g. {{"currency": "USD"}} or {{"format": "DD/MM/YYYY"}}.
- Skip headings and boilerplate that carry no information."""


def number_lines(lines: list[Line]) -> str:
    return "\n".join(f"[{ln.id}] {ln.text}" for ln in lines)


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def build_fields(raw_fields: list[dict], lines_by_id: dict[str, Line]) -> list[ExtractedField]:
    out: list[ExtractedField] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_fields:
        if not isinstance(raw, dict) or not str(raw.get("value", "")).strip():
            continue
        ids = [i for i in raw.get("line_ids") or [] if i in lines_by_id]
        if not ids:
            log.debug("Dropping field %r: cites no known line", raw.get("name"))
            continue
        name = re.sub(r"\W+", "_", str(raw.get("name") or "field").strip().lower()).strip("_") or "field"
        value = str(raw["value"]).strip()
        if (name, value) in seen:
            continue
        seen.add((name, value))

        page = lines_by_id[ids[0]].page
        on_page = [i for i in ids if lines_by_id[i].page == page]  # one box per field: its first page
        cited_text = " ".join(lines_by_id[i].text for i in ids)
        found = _norm(value) in _norm(cited_text)
        attributes = dict(raw.get("attributes") or {})
        if not found:
            attributes["value_found_in_lines"] = False
        ocr_conf = min(lines_by_id[i].confidence for i in ids)
        ftype = raw.get("type") if raw.get("type") in FIELD_TYPES else "text"
        out.append(ExtractedField(name=name, value=value, type=ftype, line_ids=ids, attributes=attributes,
                                  page=page, bbox=union(lines_by_id[i].bbox for i in on_page),
                                  confidence=round(ocr_conf if found else min(ocr_conf, 0.5), 4)))
    return out


def extract_fields(lines: list[Line], llm: LLM, language: str, max_lines: int = 120) -> list[ExtractedField]:
    """Sends the document in slices of max_lines so long documents fit the model's context."""
    by_id = {ln.id: ln for ln in lines}
    raw: list[dict] = []
    for start in range(0, len(lines), max_lines):
        chunk = lines[start:start + max_lines]
        user = f"Document language: {NAMES.get(language, language)}\n\nLINES:\n{number_lines(chunk)}"
        try:
            raw += parse_json(llm.complete(SYSTEM, user)).get("fields", [])
        except ValueError as e:
            log.warning("Extraction call returned unusable JSON (%s); skipping lines %d-%d",
                        e, start + 1, start + len(chunk))
    return build_fields(raw, by_id)
