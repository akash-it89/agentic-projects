"""Score predicted articles against gold labels.

- Box quality: one-to-one matching by IoU (Hungarian), precision / recall / F1 at IoU 0.5 and 0.75
- Grouping quality: pairwise F1 over gold lines ("are these two lines in the same article?").
  Needs gold lines; pages labelled with article boxes only get box metrics.
- Text quality: character error rate (CER) on matched articles
"""

import json
import re
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import center, contains_point, iou

THRESHOLDS = (0.5, 0.75)


def match(pred: list, gold: list) -> list[tuple[int, int, float]]:
    if not pred or not gold:
        return []
    m = np.array([[iou(p, g) for g in gold] for p in pred])
    rows, cols = linear_sum_assignment(-m)
    return [(int(r), int(c), float(m[r, c])) for r, c in zip(rows, cols, strict=True)]


def box_counts(pred: list, gold: list, thr: float) -> tuple[int, int, int]:
    """(true positives, number predicted, number gold)."""
    tp = sum(1 for _, _, v in match(pred, gold) if v >= thr)
    return tp, len(pred), len(gold)


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def _pairs(sizes) -> int:
    return sum(n * (n - 1) // 2 for n in sizes)


def grouping_counts(pred_articles: list[dict], gold_lines: list[dict]) -> tuple[int, int, int]:
    """(same-article line pairs both agree on, pairs predicted together, pairs gold together)."""
    cells: Counter = Counter()
    pred_sizes: Counter = Counter()
    gold_sizes: Counter = Counter()
    for i, ln in enumerate(gold_lines):
        c = center(ln["box"])
        pred_id = next((a["article_id"] for a in pred_articles
                        if any(contains_point(r, c) for r in (a.get("regions") or [a["bbox"]]))), f"none-{i}")
        cells[(ln["article_id"], pred_id)] += 1
        pred_sizes[pred_id] += 1
        gold_sizes[ln["article_id"]] += 1
    return _pairs(cells.values()), _pairs(pred_sizes.values()), _pairs(gold_sizes.values())


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evaluate_page(pred: dict, gold: dict) -> dict:
    pb, gb = [a["bbox"] for a in pred["articles"]], [a["bbox"] for a in gold["articles"]]
    out = {"page_id": gold["page_id"], "n_pred": len(pb), "n_gold": len(gb)}
    for thr in THRESHOLDS:
        out[f"box@{thr}"] = box_counts(pb, gb, thr)
    if gold.get("lines"):
        out["grouping"] = grouping_counts(pred["articles"], gold["lines"])
    errors = chars = 0
    for pi, gi, v in match(pb, gb):
        g_text = _norm(gold["articles"][gi].get("text", ""))
        if v >= 0.5 and g_text:
            errors += jiwer.cer(g_text, _norm(pred["articles"][pi]["text"]) or " ") * len(g_text)
            chars += len(g_text)
    out["cer"] = (errors, chars)
    return out


def summarize(pages: list[dict]) -> dict:
    total = {}
    for thr in THRESHOLDS:
        key = f"box@{thr}"
        total[key] = prf(*(sum(p[key][i] for p in pages) for i in range(3)))
    grouped = [p["grouping"] for p in pages if "grouping" in p]
    if grouped:
        tp, n_pred, n_gold = (sum(g[i] for g in grouped) for i in range(3))
        total["grouping_pairwise"] = prf(tp, n_pred, n_gold)
    errors, chars = (sum(p["cer"][i] for p in pages) for i in range(2))
    total["cer"] = round(errors / chars, 4) if chars else None
    total["pages"] = len(pages)
    return total


def evaluate_dirs(pred_dir: Path, gold_dir: Path) -> tuple[dict, list[dict]]:
    pages = []
    for gold_path in sorted(Path(gold_dir).glob("*.json")):
        pred_path = Path(pred_dir) / gold_path.name
        if not pred_path.exists():
            continue
        pages.append(evaluate_page(json.loads(pred_path.read_text(encoding="utf-8")),
                                   json.loads(gold_path.read_text(encoding="utf-8"))))
    if not pages:
        raise SystemExit(f"No page in {pred_dir} has a matching gold file in {gold_dir}")
    return summarize(pages), pages


def write_report(summary: dict, pages: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "pages": pages}, indent=2), encoding="utf-8")
    lines = ["# Evaluation", "", f"Pages: {summary['pages']}", "", "| metric | precision | recall | F1 |", "|---|---|---|---|"]
    for key in [k for k in summary if isinstance(summary[k], dict)]:
        m = summary[key]
        lines.append(f"| {key} | {m['precision']} | {m['recall']} | {m['f1']} |")
    lines += ["", f"Character error rate on matched articles: {summary['cer']}", "",
              "## Per page (box F1 at IoU 0.5)", "", "| page | predicted | gold | F1 |", "|---|---|---|---|"]
    for p in pages:
        lines.append(f"| {p['page_id']} | {p['n_pred']} | {p['n_gold']} | {prf(*p['box@0.5'])['f1']} |")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
