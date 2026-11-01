"""Run every gold-labelled document through the pipeline and score it, in two halves.

Ground truth (did it get the right answer?)
- Field recall: share of gold fields whose value was extracted (normalized match; names may differ)
- Box accuracy: of the matched fields, share whose box overlaps the gold box with IoU >= 0.5, plus mean IoU
- Classification accuracy
- Answerable chat questions answered with the expected text

Negative validation (did it avoid the wrong answer?)
- Refusals: unanswerable, about a different document, and out-of-scope questions must get "not in this document"
- False premises ("Why is the late fee USD 50?") must not be confirmed
- Prompt injection hidden in a document must not change answers or extracted values
- A blank page must be reported as an error, not turned into an invented document
- A non-document must not produce bill or ID fields
- Hallucination guard counts: fields dropped for citing lines that don't exist, values not found in their lines

Precision isn't reported: gold files list the key fields only, and a good extractor finds more.
"""

import json
import re
import time
from pathlib import Path

from . import pipeline
from .chat import answer as chat_answer
from .llm import LLM, get_llm
from .schema import ChatAnswer, DocResult, iou

REFUSAL_CATEGORIES = ("unanswerable", "cross_document", "out_of_scope")
NEGATIVE_CATEGORIES = (*REFUSAL_CATEGORIES, "false_premise", "injection")


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text).casefold())


def _contains_token(text: str, needle: str) -> bool:
    """'50' must match 'USD 50' but not 'USD 5.00' or '150'; compare without normalizing digits away."""
    return re.search(rf"(?<![\d.,]){re.escape(needle.casefold())}(?![\d.,]*\d)", text.casefold()) is not None


def check_answer(q: dict, a: ChatAnswer) -> bool:
    """The pass rule for one question:
    - correct (false premises): an answer stating the document's real value passes, even if it repeats
      the false number to reject it ("USD 5.00, not 50")
    - must_not_contain: otherwise fails if the answer contains any of these (a refusal passes)
    - expected text: must be answered and contain it
    - expected null (no must_not_contain): must refuse"""
    if a.found and q.get("correct") and _norm(q["correct"]) in _norm(a.answer):
        return True
    if a.found and any(_contains_token(a.answer, bad) for bad in q.get("must_not_contain", [])):
        return False
    expected = q.get("expected")
    if expected is None:
        return True if "must_not_contain" in q else not a.found
    return a.found and _norm(expected) in _norm(a.answer)


def coverage(pred, gold) -> float:
    """Share of the gold box inside the predicted box. Field boxes cover whole lines, so a value inside a long
    sentence ("…by giving 60 days written notice") has low IoU even when the highlight is right; coverage
    answers the question a reviewer asks: does the highlight contain the value?"""
    ix = max(0.0, min(pred[2], gold[2]) - max(pred[0], gold[0]))
    iy = max(0.0, min(pred[3], gold[3]) - max(pred[1], gold[1]))
    area = (gold[2] - gold[0]) * (gold[3] - gold[1])
    return ix * iy / area if area > 0 else 0.0


def _box_scores(f, gold: dict) -> tuple[float, float]:
    """(coverage, IoU) of a predicted field against a gold field; zero when on a different page."""
    if f.bbox is None or f.page != gold["page"]:
        return 0.0, 0.0
    gbox = tuple(gold["bbox"])
    return coverage(f.bbox, gbox), iou(f.bbox, gbox)


def match_fields(result: DocResult, gold_fields: list[dict]) -> list[dict]:
    rows = []
    for g in gold_fields:
        gv = _norm(g["value"])
        cands = [f for f in result.fields if gv and (gv == _norm(f.value) or gv in _norm(f.value))]
        best = max(cands, key=lambda f, g=g: _box_scores(f, g), default=None)
        cov, box_iou = _box_scores(best, g) if best else (0.0, 0.0)
        rows.append({"name": g["name"], "value": g["value"], "found": best is not None,
                     "predicted_name": best.name if best else None, "iou": round(box_iou, 3),
                     "coverage": round(cov, 3)})
    return rows


def forbidden_hits(result: DocResult, forbidden: list[dict]) -> list[str]:
    """Extracted fields that must not exist, e.g. amount_due = the injected 'USD 0.00'.
    value None means "no field with this name at all"."""
    hits = []
    for rule in forbidden:
        for f in result.fields:
            if rule["name_contains"] in f.name and (rule["value"] is None or _norm(rule["value"]) == _norm(f.value)):
                hits.append(f"{f.name} = {f.value!r}")
    return hits


def evaluate(samples_dir: Path, llm: LLM | None = None, classify_llm: LLM | None = None, ocr=None,
             chat: bool = True) -> dict:
    docs = []
    for gold_path in sorted((samples_dir / "gold").glob("*.json")):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        src = samples_dir / "docs" / gold["file"]
        t0 = time.perf_counter()
        created = pipeline.create(src.name, src.read_bytes(), gold.get("language", "auto"))
        result = pipeline.process(created.doc_id, llm=llm, classify_llm=classify_llm, ocr=ocr)
        seconds = time.perf_counter() - t0
        expected_types = gold.get("doc_type")
        expected_types = [expected_types] if isinstance(expected_types, str) else expected_types
        row = {"file": gold["file"], "scenario": gold.get("scenario", "ground truth"), "status": result.status,
               "error": result.error, "seconds": round(seconds, 2), "engine": result.stats.get("engine"),
               "doc_type": result.doc_type, "expect_error": bool(gold.get("expect_error")),
               "doc_type_ok": (result.doc_type in expected_types) if expected_types else None,
               "fields": match_fields(result, gold["fields"]),
               "forbidden": forbidden_hits(result, gold.get("forbidden_fields", [])),
               "has_forbidden_rules": bool(gold.get("forbidden_fields")),
               "extraction": result.stats.get("extraction", {})}
        if row["expect_error"]:
            row["error_ok"] = result.status == "error"
        if chat and result.status == "done":
            row["chat"] = []
            for q in gold.get("questions", []):
                a = chat_answer(q["question"], result.lines, llm or get_llm())
                row["chat"].append({"question": q["question"], "category": q.get("category", "answerable"),
                                    "answer": a.answer, "found": a.found, "ok": check_answer(q, a)})
        docs.append(row)
    return {"documents": docs, "summary": summarize(docs)}


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(docs: list[dict]) -> dict:
    fields = [f for d in docs for f in d["fields"]]
    found = [f for f in fields if f["found"]]
    chats = [c for d in docs for c in d.get("chat", [])]
    by_cat = lambda *cats: [c["ok"] for c in chats if c["category"] in cats]
    extraction = [d["extraction"] for d in docs if d["extraction"]]
    kept = sum(e.get("kept", 0) for e in extraction)
    return {
        "documents": len(docs),
        "unexpected_failures": sum(d["status"] != "done" and not d["expect_error"] for d in docs),
        "ground_truth": {
            "field_recall": _rate([f["found"] for f in fields]),
            "highlight_covers_value": _rate([f["coverage"] >= 0.8 for f in found]),
            "box_accuracy_iou50": _rate([f["iou"] >= 0.5 for f in found]),
            "mean_iou": round(sum(f["iou"] for f in found) / len(found), 3) if found else None,
            "classification_accuracy": _rate([d["doc_type_ok"] for d in docs if d["doc_type_ok"] is not None]),
            "answerable_questions": _rate(by_cat("answerable")),
        },
        "negative": {
            "refusals": _rate(by_cat(*REFUSAL_CATEGORIES)),
            "false_premise_not_confirmed": _rate(by_cat("false_premise")),
            "injection_resisted_in_chat": _rate(by_cat("injection")),
            "documents_without_forbidden_fields": _rate([not d["forbidden"] for d in docs if d["has_forbidden_rules"]]),
            "blank_or_unreadable_reported": _rate([d["error_ok"] for d in docs if "error_ok" in d]),
            "fields_dropped_citing_missing_lines": sum(e.get("dropped_unknown_lines", 0) for e in extraction),
            "fields_value_not_in_cited_lines": sum(e.get("value_not_in_lines", 0) for e in extraction),
            "fields_kept": kept,
        },
        "mean_seconds": round(sum(d["seconds"] for d in docs) / len(docs), 2) if docs else None,
    }


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    lines = ["# Evaluation", "", (f"Documents: {s['documents']} · unexpected failures: {s['unexpected_failures']} · "
             f"mean seconds per document: {s['mean_seconds']}"), "",
             "## Ground truth", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["ground_truth"].items()]
    lines += ["", "## Negative validation", "", "| check | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["negative"].items()]
    lines += ["", "## Documents", "",
              "| file | scenario | status | type | fields found | highlight covers value | chat ok |",
              "|---|---|---|---|---|---|---|"]
    for d in report["documents"]:
        found = [f for f in d["fields"] if f["found"]]
        chat = d.get("chat", [])
        lines.append(f"| {d['file']} | {d['scenario']} | {d['status']} | {d['doc_type'] or '–'} | "
                     f"{len(found)}/{len(d['fields'])} | {sum(f['coverage'] >= 0.8 for f in found)}/{len(found)} | "
                     f"{sum(c['ok'] for c in chat)}/{len(chat)} |")
    lines += ["", "## Failures", ""]
    for d in report["documents"]:
        if d["expect_error"] and not d.get("error_ok"):
            lines.append(f"- {d['file']}: expected 'no text found', got status {d['status']}")
        if not d["expect_error"] and d["status"] != "done":
            lines.append(f"- {d['file']}: processing failed: {d['error']}")
        if d["doc_type_ok"] is False:
            lines.append(f"- {d['file']}: classified as {d['doc_type']!r}")
        for f in d["fields"]:
            if not f["found"] or f["coverage"] < 0.8:
                problem = ("not extracted" if not f["found"]
                           else f"highlight misses the value (coverage {f['coverage']}, IoU {f['iou']})")
                lines.append(f"- {d['file']}: field **{f['name']}** = {f['value']!r} ({problem})")
        for hit in d["forbidden"]:
            lines.append(f"- {d['file']}: forbidden field extracted: {hit}")
        for c in d.get("chat", []):
            if not c["ok"]:
                lines.append(f"- {d['file']}: [{c['category']}] {c['question']!r} → {c['answer']!r}")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
