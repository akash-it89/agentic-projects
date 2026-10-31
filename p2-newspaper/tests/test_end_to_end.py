"""Synthetic page -> pipeline (perfect OCR lines, rules-only decider) -> evaluation.

This checks the layout and grouping logic, not OCR. Thresholds come from observed runs (box F1 about 0.95
on 8 pages) with some margin; a drop below them means a layout change broke something.
"""

import json

from p2news.config import Settings
from p2news.costmodel import estimate
from p2news.evaluate import evaluate_page, summarize
from p2news.pipeline import Pipeline, run_batch
from p2news.synth import generate


def test_synthetic_pages_end_to_end(tmp_path):
    generate(tmp_path / "synth", n=4, seed=7)
    s = Settings(decider="rules", runs_dir=tmp_path / "runs", _env_file=None)
    summary = run_batch(tmp_path / "synth" / "pages", tmp_path / "out", pipeline=Pipeline(s),
                        gold_lines_dir=tmp_path / "synth" / "gold", overlay=True)
    assert summary == {"processed": 4, "skipped": 0, "failed": 0, "out_dir": str(tmp_path / "out")}

    pages = []
    for gold_path in sorted((tmp_path / "synth" / "gold").glob("*.json")):
        pred = json.loads((tmp_path / "out" / "pages" / gold_path.name).read_text(encoding="utf-8"))
        pages.append(evaluate_page(pred, json.loads(gold_path.read_text(encoding="utf-8"))))
    m = summarize(pages)
    assert m["box@0.5"]["f1"] >= 0.85
    assert m["grouping_pairwise"]["f1"] >= 0.95
    assert m["cer"] is not None and m["cer"] < 0.02

    assert len(list((tmp_path / "out" / "overlays").glob("*.png"))) == 4

    # Re-running resumes: nothing left to do.
    again = run_batch(tmp_path / "synth" / "pages", tmp_path / "out", pipeline=Pipeline(s),
                      gold_lines_dir=tmp_path / "synth" / "gold")
    assert again["processed"] == 0 and again["skipped"] == 4

    rows = [json.loads(x) for x in (tmp_path / "out" / "timings.jsonl").read_text(encoding="utf-8").splitlines()]
    e = estimate(rows, pages=1_000_000, machine_price_per_hour=1.0, workers=10)
    assert e["pilot_pages"] == 4 and e["total_cost"] > 0 and e["llm_share_of_pairs"] == 0
