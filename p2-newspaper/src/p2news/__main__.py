"""Command line:

    uv run python -m p2news synth   --out data/synthetic --pages 5
    uv run python -m p2news run     --input data/synthetic/pages --out runs/synth-gold --gold-lines data/synthetic/gold
    uv run python -m p2news run     --input data/real/pages --out runs/real-v1
    uv run python -m p2news eval    --pred runs/real-v1/pages --gold data/real/gold
    uv run python -m p2news cost    --run runs/real-v1 --pages 2000000 --price 0.50 --workers 8
    uv run python -m p2news labelstudio export.json data/real/gold
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from .config import get_settings


def main() -> None:
    p = argparse.ArgumentParser(prog="p2news", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("synth", help="generate synthetic pages with gold labels")
    sp.add_argument("--out", type=Path, default=Path("data/synthetic"))
    sp.add_argument("--pages", type=int, default=5)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--noisy", action="store_true", help="add blur and grain like an old scan")

    rp = sub.add_parser("run", help="process a folder of page images")
    rp.add_argument("--input", type=Path, required=True)
    rp.add_argument("--out", type=Path, default=None, help="default: runs/<timestamp>")
    rp.add_argument("--gold-lines", type=Path, default=None,
                    help="use the lines from these gold files instead of OCR (tests layout logic alone)")
    rp.add_argument("--limit", type=int, default=None)
    rp.add_argument("--no-overlay", action="store_true")
    rp.add_argument("--force", action="store_true", help="re-process pages that already have output")

    ep = sub.add_parser("eval", help="score predictions against gold files")
    ep.add_argument("--pred", type=Path, required=True, help="the pages/ folder of a run")
    ep.add_argument("--gold", type=Path, required=True)
    ep.add_argument("--out", type=Path, default=None, help="default: next to --pred")

    cp = sub.add_parser("cost", help="extrapolate a pilot run to the full archive")
    cp.add_argument("--run", type=Path, required=True)
    cp.add_argument("--pages", type=int, required=True, help="total pages in the archive")
    cp.add_argument("--price", type=float, required=True, help="machine price per hour (your currency)")
    cp.add_argument("--workers", type=int, default=1)
    cp.add_argument("--llm-cost", type=float, default=0.0, help="cost per decider LLM call (0 if local)")
    cp.add_argument("--vlm-cost", type=float, default=0.0, help="cost per vision re-read (0 if local)")

    lp = sub.add_parser("labelstudio", help="convert a Label Studio JSON export to gold files")
    lp.add_argument("export", type=Path)
    lp.add_argument("out", type=Path)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "synth":
        from .synth import generate
        paths = generate(args.out, args.pages, args.seed, args.noisy)
        print(f"Wrote {len(paths)} pages to {args.out / 'pages'} and gold labels to {args.out / 'gold'}")

    elif args.cmd == "run":
        from .pipeline import run_batch
        out = args.out or get_settings().runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        print(json.dumps(run_batch(args.input, out, gold_lines_dir=args.gold_lines, overlay=not args.no_overlay,
                                   force=args.force, limit=args.limit), indent=2))

    elif args.cmd == "eval":
        from .evaluate import evaluate_dirs, write_report
        summary, pages = evaluate_dirs(args.pred, args.gold)
        report = write_report(summary, pages, args.out or args.pred.parent / "eval")
        print(json.dumps(summary, indent=2))
        print(f"Report: {report}")

    elif args.cmd == "cost":
        from .costmodel import estimate, format_report, load_timings
        print(format_report(estimate(load_timings(args.run), args.pages, args.price, args.workers,
                                     args.llm_cost, args.vlm_cost)))

    elif args.cmd == "labelstudio":
        from .labelstudio import convert
        print(f"Wrote {len(convert(args.export, args.out))} gold files to {args.out}")


if __name__ == "__main__":
    main()
