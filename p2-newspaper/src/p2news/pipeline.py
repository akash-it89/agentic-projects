"""Page image -> articles. One page per call; run_batch adds resume, per-page error isolation and timings."""

import json
import logging
import time
import traceback
from pathlib import Path

from .blocks import mark_headlines, merge_lines, page_line_height
from .config import Settings, get_settings
from .correction import VLMReader, correct_blocks
from .decider import CascadeDecider, LLMDecider
from .grouping import build_articles
from .layout import assign_columns, candidate_pairs, features, reading_order
from .ocr import GoldEngine, OCREngine, PaddleEngine
from .overlay import draw_overlay
from .preprocess import preprocess
from .schema import Line, PageResult, PairDecision
from .separators import detect_separators

log = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".jp2"}


class Pipeline:
    def __init__(self, s: Settings | None = None, ocr: OCREngine | None = None,
                 decider: CascadeDecider | None = None, reader: VLMReader | None = None):
        self.s = s or get_settings()
        self._ocr = ocr
        llm = LLMDecider(self.s.ollama_host, self.s.decider_model) if self.s.decider == "cascade" else None
        self.decider = decider or CascadeDecider(llm)
        if reader is None and self.s.correction == "vlm":
            reader = VLMReader(self.s.ollama_host, self.s.vlm_model)
        self.reader = reader

    @property
    def ocr(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = PaddleEngine(self.s)  # loaded on first page so `--help` and tests stay fast
        return self._ocr

    def process(self, image_path: Path, page_id: str | None = None, ocr: OCREngine | None = None):
        """Returns (PageResult, preprocessed grayscale image, pair decisions)."""
        s, page_id = self.s, page_id or Path(image_path).stem
        # Get the engine before starting the clock: loading OCR models on the first page is a one-off
        # cost and would otherwise inflate the per-page numbers the cost model extrapolates from.
        engine = ocr or self.ocr
        timings: dict[str, float] = {}
        t = time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal t
            now = time.perf_counter()
            timings[stage] = round(now - t, 3)
            t = now

        # Gold lines are already in original-image coordinates, so don't deskew under them.
        gray = preprocess(image_path, fix_skew=not isinstance(engine, GoldEngine))
        lap("preprocess")

        lines: list[Line] = engine.read(gray)
        lap("ocr")

        h_rules, v_rules = detect_separators(gray)
        blocks = merge_lines(lines, s.merge_gap_k, s.merge_overlap_min, s.size_tolerance, h_rules, v_rules,
                             s.headline_ratio)
        line_h = page_line_height(lines)
        mark_headlines(blocks, line_h, s.headline_ratio)
        n_columns = assign_columns(blocks, gray.shape[1], s.column_gap_ratio)
        ordered = reading_order(blocks, rtl=s.rtl)
        lap("layout")

        escalated = correct_blocks(ordered, gray, s.keep_score, self.reader)
        lap("correction")

        before = dict(self.decider.counts)
        decisions: list[PairDecision] = [self.decider.decide(a, b, features(a, b, line_h, h_rules))
                                         for a, b in candidate_pairs(ordered)]
        lap("decider")

        articles = build_articles(ordered, decisions, page_id, engine.name)
        lap("grouping")

        by_source = {k: v - before.get(k, 0) for k, v in self.decider.counts.items()}
        result = PageResult(
            page_id=page_id, width=gray.shape[1], height=gray.shape[0], language=s.ocr_lang, articles=articles,
            stats={"lines": len(lines), "blocks": len(blocks), "columns": n_columns,
                   "separators": {"horizontal": len(h_rules), "vertical": len(v_rules)},
                   "pairs": len(decisions), "decisions_by_source": by_source,
                   "vlm_blocks": escalated, "seconds": timings, "total_seconds": round(sum(timings.values()), 3)},
        )
        return result, gray, decisions


def list_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_gold_lines(gold_dir: Path, page_id: str) -> list[Line]:
    data = json.loads((gold_dir / f"{page_id}.json").read_text(encoding="utf-8"))
    return [Line(box=tuple(ln["box"]), text=ln["text"], score=ln.get("score", 1.0)) for ln in data["lines"]]


def run_batch(input_dir: Path, out_dir: Path, *, pipeline: Pipeline | None = None, gold_lines_dir: Path | None = None,
              overlay: bool = True, force: bool = False, limit: int | None = None) -> dict:
    """Process every image in input_dir. Re-running skips finished pages, so a crash at page 90,000
    costs you one page, not the whole run."""
    pipeline = pipeline or Pipeline()
    pages_dir, overlay_dir = out_dir / "pages", out_dir / "overlays"
    pages_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    mode = "w" if force else "a"  # resuming appends; --force redoes every page, so start the logs fresh
    with (open(out_dir / "timings.jsonl", mode, encoding="utf-8") as timings_f,
          open(out_dir / "pairs.jsonl", mode, encoding="utf-8") as pairs_f,
          open(out_dir / "errors.jsonl", mode, encoding="utf-8") as errors_f):
        for path in list_images(input_dir)[:limit]:
            page_id = path.stem
            out_json = pages_dir / f"{page_id}.json"
            if out_json.exists() and not force:
                skipped += 1
                continue
            try:
                ocr = GoldEngine(load_gold_lines(gold_lines_dir, page_id)) if gold_lines_dir else None
                result, gray, decisions = pipeline.process(path, page_id, ocr=ocr)
            except Exception as e:  # noqa: BLE001  one bad scan must not stop a million-page run
                failed += 1
                errors_f.write(json.dumps({"page_id": page_id, "error": repr(e),
                                           "trace": traceback.format_exc(limit=3)}) + "\n")
                log.error("%s failed: %s", page_id, e)
                continue
            out_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            if overlay:
                draw_overlay(gray, result, overlay_dir / f"{page_id}.png")
            st = result.stats
            timings_f.write(json.dumps({"page_id": page_id, "seconds": st["seconds"], "total": st["total_seconds"],
                                        "pairs": st["pairs"], "by_source": st["decisions_by_source"],
                                        "vlm_blocks": st["vlm_blocks"], "blocks": st["blocks"]}) + "\n")
            pairs_f.writelines(json.dumps({"page_id": page_id, "a": d.a, "b": d.b, "same_article": d.same_article,
                                          "confidence": d.confidence, "source": d.source,
                                          "features": d.features}) + "\n" for d in decisions)
            done += 1
            log.info("%s: %d articles in %.1fs", page_id, len(result.articles), st["total_seconds"])
    return {"processed": done, "skipped": skipped, "failed": failed, "out_dir": str(out_dir)}
