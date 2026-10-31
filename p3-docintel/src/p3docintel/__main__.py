"""Command line:

    uv run python -m p3docintel serve                       # web app on http://localhost:8000
    uv run python -m p3docintel samples                     # write sample documents + gold labels to samples/
    uv run python -m p3docintel process samples/docs/utility_bill.pdf
    uv run python -m p3docintel eval                        # score every sample (needs Ollama or Bedrock)
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT, get_settings


def main() -> None:
    p = argparse.ArgumentParser(prog="p3docintel", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="run the web app")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)

    gp = sub.add_parser("samples", help="generate sample documents with gold labels")
    gp.add_argument("--out", type=Path, default=PROJECT_ROOT / "samples")

    pp = sub.add_parser("process", help="process one file and print the result")
    pp.add_argument("file", type=Path)
    pp.add_argument("--lang", default="auto", choices=["auto", "en", "ta", "ar"])

    ep = sub.add_parser("eval", help="score the pipeline on gold-labelled documents")
    ep.add_argument("--samples", type=Path, default=PROJECT_ROOT / "samples")
    ep.add_argument("--no-chat", action="store_true")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("p3docintel.api:app", host=args.host, port=args.port)

    elif args.cmd == "samples":
        from .samples import generate
        for path in generate(args.out):
            print(path)

    elif args.cmd == "process":
        from . import pipeline
        created = pipeline.create(args.file.name, args.file.read_bytes(), args.lang)
        r = pipeline.process(created.doc_id)
        print(json.dumps({"doc_id": r.doc_id, "status": r.status, "error": r.error, "doc_type": r.doc_type,
                          "language": r.language, "summary": r.summary,
                          "fields": [{"name": f.name, "value": f.value, "page": f.page, "confidence": f.confidence}
                                     for f in r.fields], "stats": r.stats}, indent=2, ensure_ascii=False))

    elif args.cmd == "eval":
        from .evaluate import evaluate, write_report
        report = evaluate(args.samples, chat=not args.no_chat)
        path = write_report(report, get_settings().runs_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}")
        print(json.dumps(report["summary"], indent=2))
        print(f"Report: {path}")


if __name__ == "__main__":
    main()
