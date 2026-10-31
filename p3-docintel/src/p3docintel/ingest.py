"""Uploads -> page PNGs (what the viewer shows and OCR reads) + the PDF text layer when there is one."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps

from .schema import Line, Page
from .text_layer import text_layer_lines

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class RenderedPage:
    page: Page
    png: Path
    text_lines: list[Line]  # empty for scans and photos: those need OCR


def supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in PDF_SUFFIXES | IMAGE_SUFFIXES


def render(source: Path, out_dir: Path, dpi: int = 200, min_words: int = 5) -> list[RenderedPage]:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _render_pdf(source, out_dir, dpi, min_words)
    if suffix in IMAGE_SUFFIXES:
        return _render_image(source, out_dir)
    raise ValueError(f"Unsupported file type {suffix!r}. Upload a PDF or an image.")


def _render_pdf(source: Path, out_dir: Path, dpi: int, min_words: int) -> list[RenderedPage]:
    pages = []
    with pymupdf.open(source) as doc:
        for i, pg in enumerate(doc, start=1):
            png = out_dir / f"page-{i}.png"
            pix = pg.get_pixmap(dpi=dpi)
            pix.save(png)
            lines = text_layer_lines(pg, i)
            has_text = sum(len(ln.text.split()) for ln in lines) >= min_words
            pages.append(RenderedPage(Page(number=i, width=pix.width, height=pix.height,
                                           source="pdf_text" if has_text else "ocr"),
                                      png, lines if has_text else []))
    return pages


def _render_image(source: Path, out_dir: Path) -> list[RenderedPage]:
    pages = []
    with Image.open(source) as img:
        frames = getattr(img, "n_frames", 1)  # multi-page TIFF
        for i in range(frames):
            img.seek(i)
            # Phone photos carry their rotation in EXIF; apply it so boxes match what the viewer shows.
            frame = ImageOps.exif_transpose(img.copy()).convert("RGB")
            png = out_dir / f"page-{i + 1}.png"
            frame.save(png)
            pages.append(RenderedPage(Page(number=i + 1, width=frame.width, height=frame.height, source="ocr"),
                                      png, []))
    return pages
