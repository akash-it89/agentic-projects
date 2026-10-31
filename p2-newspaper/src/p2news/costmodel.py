"""Extrapolate a pilot run to the full archive. This table is what the client cares about.

    cost per page = machine price per hour x hours per page
                  + LLM decider calls per page x cost per call
                  + vision re-reads per page x cost per call
"""

import json
import statistics
from pathlib import Path


def load_timings(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "timings.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No timings in {path}. Run the pipeline first.")
    return rows


def estimate(rows: list[dict], pages: int, machine_price_per_hour: float, workers: int = 1,
             llm_cost_per_call: float = 0.0, vlm_cost_per_call: float = 0.0) -> dict:
    sec_per_page = statistics.mean(r["total"] for r in rows)
    llm_calls = statistics.mean(r["by_source"].get("llm", 0) for r in rows)
    pairs = statistics.mean(r["pairs"] for r in rows) or 1
    vlm_calls = statistics.mean(r["vlm_blocks"] for r in rows)
    stages = {k: round(statistics.mean(r["seconds"].get(k, 0) for r in rows), 3) for k in rows[0]["seconds"]}

    machine_hours = pages * sec_per_page / 3600
    cost_per_page = (machine_price_per_hour * sec_per_page / 3600
                     + llm_calls * llm_cost_per_call + vlm_calls * vlm_cost_per_call)
    return {
        "pilot_pages": len(rows),
        "seconds_per_page": round(sec_per_page, 3),
        "seconds_per_stage": stages,
        "pages_per_hour_per_worker": round(3600 / sec_per_page) if sec_per_page else None,
        "llm_share_of_pairs": round(llm_calls / pairs, 4),
        "vlm_blocks_per_page": round(vlm_calls, 3),
        "archive_pages": pages,
        "machine_hours": round(machine_hours, 1),
        "wall_clock_hours": round(machine_hours / max(1, workers), 1),
        "cost_per_page": round(cost_per_page, 6),
        "total_cost": round(cost_per_page * pages, 2),
    }


def format_report(e: dict) -> str:
    rows = [("Pilot pages measured", e["pilot_pages"]), ("Seconds per page", e["seconds_per_page"]),
            ("Pages per hour per worker", e["pages_per_hour_per_worker"]),
            ("Share of pairs sent to the LLM", e["llm_share_of_pairs"]),
            ("Vision re-reads per page", e["vlm_blocks_per_page"]),
            ("Archive size (pages)", f"{e['archive_pages']:,}"), ("Machine hours", f"{e['machine_hours']:,}"),
            ("Wall-clock hours", f"{e['wall_clock_hours']:,}"), ("Cost per page", e["cost_per_page"]),
            ("Total cost", f"{e['total_cost']:,}")]
    out = ["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    out += ["", "Seconds per stage: " + ", ".join(f"{k} {v}" for k, v in e["seconds_per_stage"].items())]
    return "\n".join(out)
