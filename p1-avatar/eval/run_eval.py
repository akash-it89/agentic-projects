"""Evaluate the text pipeline (retrieval + answer) against a question set.

    uv run python eval/run_eval.py                                  # eval/questions.sample.jsonl
    uv run python eval/run_eval.py --questions eval/my_questions.jsonl

Each line: {"id", "lang", "question", "answerable": bool, "expected_keywords": [..] (optional)}

Writes runs/eval-<timestamp>/results.jsonl and report.md. results.jsonl has question / contexts /
answer fields, so it can be loaded into RAGAS or DeepEval for faithfulness scoring.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1avatar.answer import answer
from p1avatar.config import get_settings
from p1avatar.lang import SUPPORTED, detect_lang


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(rows: list[dict], answer_fn=answer) -> tuple[list[dict], dict]:
    results = []
    for q in rows:
        t0 = time.perf_counter()
        a = answer_fn(q["question"], q["lang"])
        ms = (time.perf_counter() - t0) * 1000
        keywords = q.get("expected_keywords") or []
        results.append({
            "id": q["id"], "lang": q["lang"], "question": q["question"], "answerable": q["answerable"],
            "answer": a.text, "refused": a.refused, "reason": a.reason,
            "contexts": [p["text"] for p in a.passages], "top_score": a.passages[0]["score"] if a.passages else None,
            "correct_refusal": (not q["answerable"]) and a.refused,
            "over_refusal": q["answerable"] and a.refused,
            "keyword_hit": all(k.lower() in a.text.lower() for k in keywords) if keywords and not a.refused else None,
            "lang_match": detect_lang(a.text) == q["lang"],
            "ms": round(ms),
        })
    return results, summarize(results)


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[dict]) -> dict:
    out = {}
    for lang in ("all", *SUPPORTED):
        rs = [r for r in results if lang == "all" or r["lang"] == lang]
        if not rs:
            continue
        unans = [r for r in rs if not r["answerable"]]
        ans = [r for r in rs if r["answerable"]]
        out[lang] = {
            "n": len(rs),
            "refusal_accuracy": _rate([r["correct_refusal"] for r in unans]),
            "over_refusal": _rate([r["over_refusal"] for r in ans]),
            "keyword_hit": _rate([r["keyword_hit"] for r in ans if r["keyword_hit"] is not None]),
            "lang_match": _rate([r["lang_match"] for r in rs]),
            "median_ms": round(statistics.median(r["ms"] for r in rs)),
        }
    return out


def write_report(results: list[dict], summary: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)

    s = get_settings()
    lines = [f"# Eval report {out_dir.name}", "",
             f"LLM: `{s.llm_provider}` · MIN_SCORE: `{s.min_score}` · embed: `{s.embed_model}`", "",
             "| lang | n | refusal accuracy | over-refusal | keyword hit | language match | median ms |",
             "|---|---|---|---|---|---|---|"]
    for lang, m in summary.items():
        lines.append(f"| {lang} | {m['n']} | {m['refusal_accuracy']} | {m['over_refusal']} | {m['keyword_hit']} "
                     f"| {m['lang_match']} | {m['median_ms']} |")
    lines += ["", "## Failures", ""]
    for r in results:
        if r["over_refusal"] or (not r["answerable"] and not r["refused"]) or r["keyword_hit"] is False \
                or not r["lang_match"]:
            lines.append(f"- **{r['id']}** {r['question']} → {r['answer']} "
                         f"(reason={r['reason']}, top_score={r['top_score']})")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.sample.jsonl")
    args = parser.parse_args()
    results, summary = evaluate(load(args.questions))
    report = write_report(results, summary, get_settings().runs_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}")
    print(json.dumps(summary, indent=2))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
