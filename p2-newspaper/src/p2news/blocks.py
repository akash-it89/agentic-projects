"""Lines -> blocks. PaddleOCR returns one box per line (often several per line); a block is a run of lines
with the same type size and small gaps, i.e. a paragraph or a headline. This is the "gap between lines"
tuning the trainer described: MERGE_GAP_K and MERGE_OVERLAP_MIN are the knobs."""

import statistics

from .geometry import Box, height, union, vertical_gap, x_overlap_ratio
from .schema import Block, Line
from .separators import rule_between_vertically


def page_line_height(lines: list[Line]) -> float:
    return statistics.median(height(ln.box) for ln in lines) if lines else 1.0


def _same_row(a: Box, b: Box) -> bool:
    """Two boxes on the same text line (OCR sometimes splits one line into pieces)."""
    overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap > 0.6 * min(height(a), height(b))


def _rule_between_horizontally(left: Box, right: Box, vertical_rules: list[Box]) -> bool:
    return any(left[2] - 4 <= (r[0] + r[2]) / 2 <= right[0] + 4 and r[1] <= left[3] and r[3] >= left[1]
               for r in vertical_rules)


def join_row_fragments(lines: list[Line], max_gap_k: float = 0.6, vertical_rules: list[Box] | None = None) -> list[Line]:
    """Merge pieces of the same printed line that sit next to each other within one column.

    The gap limit is a fraction of the line height: word spaces are smaller than that, column gutters
    are usually larger. A vertical rule between two pieces always keeps them apart.
    """
    if not lines:
        return []
    vertical_rules = vertical_rules or []
    h = page_line_height(lines)
    lines = sorted(lines, key=lambda ln: (round(ln.box[1] / max(1.0, h / 2)), ln.box[0]))
    out: list[Line] = []
    for ln in lines:
        prev = out[-1] if out else None
        if (prev and _same_row(prev.box, ln.box) and 0 <= ln.box[0] - prev.box[2] <= max_gap_k * h
                and not _rule_between_horizontally(prev.box, ln.box, vertical_rules)):
            out[-1] = Line(union([prev.box, ln.box]), f"{prev.text} {ln.text}", min(prev.score, ln.score))
        else:
            out.append(ln)
    return out


def merge_lines(lines: list[Line], gap_k: float = 0.8, overlap_min: float = 0.5, size_tolerance: float = 0.3,
                horizontal_rules: list[Box] | None = None, vertical_rules: list[Box] | None = None,
                headline_ratio: float = 1.3, lookback: int = 40) -> list[Block]:
    horizontal_rules = horizontal_rules or []
    lines = sorted(join_row_fragments(lines, vertical_rules=vertical_rules), key=lambda ln: (ln.box[1], ln.box[0]))
    h_med = page_line_height(lines)

    def headline_sized(h: float) -> bool:
        return h >= headline_ratio * h_med

    blocks: list[Block] = []
    for ln in lines:
        target = None
        for b in reversed(blocks[-lookback:]):
            gap = vertical_gap(b.box, ln.box)
            # Box height depends on the letters ("coal." is shorter than "Bridge"), so a smaller line is
            # normal inside body text. What must not mix is headline-sized and body-sized type.
            same_size = (headline_sized(height(ln.box)) == headline_sized(b.line_height)
                         and height(ln.box) / max(1.0, b.line_height) <= 1 + size_tolerance)
            # Gaps scale with type size: a two-line headline has bigger spacing than body text.
            line_h = max(h_med, b.line_height)
            if (x_overlap_ratio(b.box, ln.box) >= overlap_min
                    and -0.3 * line_h <= gap <= gap_k * line_h
                    and same_size
                    and not rule_between_vertically(b.box, ln.box, horizontal_rules)):
                target = b
                break
        if target:
            target.lines.append(ln)
            target.box = union([target.box, ln.box])
        else:
            blocks.append(Block(id=len(blocks), lines=[ln], box=ln.box, line_height=height(ln.box)))
    for b in blocks:
        b.line_height = statistics.median(height(ln.box) for ln in b.lines)
    return blocks


def mark_headlines(blocks: list[Block], page_line_h: float, ratio: float = 1.3) -> None:
    for b in blocks:
        b.is_headline = b.line_height >= ratio * page_line_h
