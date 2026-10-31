"""Run every gold-labelled document through the pipeline and score it.

- Field recall: share of gold fields whose value was extracted (normalized match; names may differ)
- Box accuracy: of the matched fields, share whose box overlaps the gold box with IoU >= 0.5, plus mean IoU
- Classification accuracy
- Chat: answer contains the expected text, or correctly says "not in this document" when expected is null

Precision isn't reported: gold files list the key fields only, and a good extractor finds more.
"""

import json
import re
import time
from pathlib import Path

from . import pipeline
from .chat import answer as chat_answer
from .llm import LLM, get_llm
from .schema import DocResult, iou


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text).casefold())


def match_fields(result: DocResult, gold_fields: list[dict]) -> list[dict]:
    rows = []
    for g in gold_fields:
        gv = _norm(g["value"])
        cands = [f for f in result.fields if gv and (gv == _norm(f.value) or gv in _norm(f.value))]
        best = max(cands, key=lambda f: iou(f.bbox, tuple(g["bbox"])) if f.bbox and f.page == g["page"] else 0,
                   default=None)
        box_iou = iou(best.bbox, tuple(g["bbox"])) if best and best.bbox and best.page == g["page"] else 0.0
        rows.append({"name": g["name"], "value": g["value"], "found": best is not None,
                     "predicted_name": best.name if best else None, "iou": round(box_iou, 3)})
    return rows


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
        row = {"file": gold["file"], "status": result.status, "error": result.error, "seconds": round(seconds, 2),
               "engine": result.stats.get("engine"), "doc_type": result.doc_type,
               "doc_type_ok": result.doc_type == gold["doc_type"], "fields": match_fields(result, gold["fields"])}
        if chat and result.status == "done":
            row["chat"] = []
            for q in gold.get("questions", []):
                a = chat_answer(q["question"], result.lines, llm or get_llm())
                ok = (not a.found) if q["expected"] is None else (a.found and _norm(q["expected"]) in _norm(a.answer))
                row["chat"].append({"question": q["question"], "answer": a.answer, "ok": ok})
        docs.append(row)
    return {"documents": docs, "summary": summarize(docs)}


def summarize(docs: list[dict]) -> dict:
    fields = [f for d in docs for f in d["fields"]]
    found = [f for f in fields if f["found"]]
    chats = [c for d in docs for c in d.get("chat", [])]
    return {
        "documents": len(docs),
        "failed": sum(d["status"] != "done" for d in docs),
        "field_recall": round(len(found) / len(fields), 3) if fields else None,
        "box_accuracy_iou50": round(sum(f["iou"] >= 0.5 for f in found) / len(found), 3) if found else None,
        "mean_iou": round(sum(f["iou"] for f in found) / len(found), 3) if found else None,
        "classification_accuracy": round(sum(d["doc_type_ok"] for d in docs) / len(docs), 3) if docs else None,
        "chat_accuracy": round(sum(c["ok"] for c in chats) / len(chats), 3) if chats else None,
        "mean_seconds": round(sum(d["seconds"] for d in docs) / len(docs), 2) if docs else None,
    }


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    lines = ["# Evaluation", "", "| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s.items()]
    lines += ["", "## Documents", "", "| file | engine | type ok | fields found | boxes IoU≥0.5 | seconds |",
              "|---|---|---|---|---|---|"]
    for d in report["documents"]:
        found = [f for f in d["fields"] if f["found"]]
        lines.append(f"| {d['file']} | {d['engine']} | {d['doc_type_ok']} | {len(found)}/{len(d['fields'])} | "
                     f"{sum(f['iou'] >= 0.5 for f in found)}/{len(found)} | {d['seconds']} |")
    lines += ["", "## Missed or misplaced fields", ""]
    for d in report["documents"]:
        for f in d["fields"]:
            if not f["found"] or f["iou"] < 0.5:
                problem = "not extracted" if not f["found"] else f"box IoU {f['iou']}"
                lines.append(f"- {d['file']}: **{f['name']}** = {f['value']!r} ({problem})")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
