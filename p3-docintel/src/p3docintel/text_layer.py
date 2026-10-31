"""Born-digital PDFs already contain every word with its exact position. Use that before paying for OCR."""

import pymupdf

from .schema import Line


def text_layer_lines(page: pymupdf.Page, page_no: int) -> list[Line]:
    w, h = page.rect.width, page.rect.height
    grouped: dict[tuple[int, int], list] = {}
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words", sort=True):
        grouped.setdefault((block, line), []).append((x0, y0, x1, y1, word))
    lines = []
    for words in grouped.values():
        text = " ".join(wd[4] for wd in words).strip()
        if not text:
            continue
        box = (min(wd[0] for wd in words) / w, min(wd[1] for wd in words) / h,
               max(wd[2] for wd in words) / w, max(wd[3] for wd in words) / h)
        lines.append(Line(id=f"p{page_no}-L{len(lines) + 1}", page=page_no, text=text,
                          bbox=tuple(round(min(1.0, max(0.0, v)), 5) for v in box), engine="pdf_text"))
    return lines
