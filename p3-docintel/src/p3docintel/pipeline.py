"""Upload -> pages -> lines (text layer or OCR) -> [correction] -> fields -> type -> summary.

Each stage updates result.stage and saves, so the UI can show progress while it polls.
"""

import logging
import time
from pathlib import Path

from PIL import Image

from . import store
from .classify import classify, summarise
from .config import Settings, get_settings
from .correction import VLMReader, correct_lines
from .extract import extract_fields
from .ingest import render
from .langdetect import SUPPORTED, detect_script
from .llm import LLM, get_llm
from .ocr import OCREngine, get_engine
from .schema import DocResult, Line

log = logging.getLogger(__name__)


def create(filename: str, data: bytes, language: str = "auto") -> DocResult:
    doc_id = store.new_doc_id()
    store.save_upload(doc_id, filename, data)
    result = DocResult(doc_id=doc_id, filename=Path(filename).name,
                       language=language if language in SUPPORTED else "auto")
    store.save(result)
    return result


def process(doc_id: str, *, llm: LLM | None = None, classify_llm: LLM | None = None,
            ocr: OCREngine | None = None, s: Settings | None = None) -> DocResult:
    s = s or get_settings()
    result = store.load(doc_id)
    timings: dict[str, float] = {}
    t = time.perf_counter()

    def stage(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        if result.stage != "queued":
            timings[result.stage] = round(now - t, 3)
        t = now
        result.stage = name
        result.stats["seconds"] = timings
        store.save(result)

    try:
        stage("rendering pages")
        rendered = render(store.source_path(doc_id), store.pages_dir(doc_id), s.render_dpi, s.min_text_layer_words)
        result.pages = [r.page for r in rendered]

        # Language: the user's choice, else detected from the text layer, else English for OCR.
        lang = result.language
        if lang == "auto":
            lang = detect_script(" ".join(ln.text for r in rendered for ln in r.text_lines)) or "en"
        result.language = lang

        stage("reading text")
        lines: list[Line] = []
        engine = None
        reader = VLMReader(s.ollama_host, s.vlm_model) if s.correction == "vlm" else None
        for r in rendered:
            if r.text_lines:
                page_lines = r.text_lines
            else:
                engine = engine or ocr or get_engine(s.ocr_engine, lang)
                page_lines = engine.read(r.png, r.page.number)
                with Image.open(r.png) as im:  # OCR may have saved the page upright, swapping width and height
                    r.page.width, r.page.height = im.size
                if reader:
                    result.stats["vlm_lines"] = result.stats.get("vlm_lines", 0) + correct_lines(
                        page_lines, r.png, lang, reader, s.keep_score)
            lines += page_lines
        result.lines = lines
        result.stats["engine"] = engine.name if engine else "pdf_text"
        if not lines:
            raise ValueError("No text found. If this is a photo, check it isn't blank or upside down.")

        # A caller-supplied LLM is used for every step; otherwise classification gets the small, cheap model.
        if classify_llm is None:
            classify_llm = llm or (get_llm(s.classify_model) if s.classify_model else None)
        llm = llm or get_llm()
        classify_llm = classify_llm or llm
        stage("extracting fields")
        result.fields = extract_fields(lines, llm, lang, s.max_lines_per_call)

        stage("classifying")
        result.doc_type, result.doc_type_confidence = classify(lines, classify_llm)

        stage("summarising")
        result.summary = summarise(lines, llm, lang)

        stage("done")
        result.status = "done"
        result.stats["llm"] = llm.name
        result.stats["total_seconds"] = round(sum(timings.values()), 3)
    except Exception as e:  # show the error in the UI instead of leaving the document stuck on "processing"
        log.exception("Processing %s failed", doc_id)
        result.status, result.error = "error", f"{type(e).__name__}: {e}"
    store.save(result)
    return result
