"""Columns, reading order and the candidate pairs the decider has to judge."""

import itertools
import re

from .geometry import vertical_gap
from .schema import Block
from .separators import rule_between_vertically

_SENTENCE_END = re.compile(r"[.!?\"'”’)]\s*$")


def assign_columns(blocks: list[Block], page_width: float, gap_ratio: float = 0.05) -> int:
    """Cluster block left edges: a new column starts where the left edges jump by more than gap_ratio x width.
    Returns the number of columns. Headlines spanning several columns get the column of their left edge."""
    if not blocks:
        return 0
    body = [b for b in blocks if not b.is_headline] or blocks
    starts: list[float] = []
    for x in sorted(b.box[0] for b in body):
        if not starts or x - starts[-1] > gap_ratio * page_width:
            starts.append(x)
    tol = gap_ratio * page_width
    for b in blocks:
        b.column = max([i for i, s in enumerate(starts) if b.box[0] >= s - tol] or [0])
    return len(starts)


def reading_order(blocks: list[Block], rtl: bool = False) -> list[Block]:
    """Top to bottom within a column; columns left to right (right to left for Arabic)."""
    return sorted(blocks, key=lambda b: (-b.column if rtl else b.column, b.box[1]))


def block_text(b: Block) -> str:
    return " ".join(ln.text for ln in b.lines)


def features(a: Block, b: Block, page_line_h: float, horizontal_rules: list) -> dict:
    same_col = a.column == b.column
    gap = vertical_gap(a.box, b.box) if same_col else None
    a_text, b_text = (a.text or block_text(a)).strip(), (b.text or block_text(b)).strip()
    return {
        "same_column": same_col,
        "column_jump": b.column == a.column + 1 or b.column == a.column - 1,
        "gap_lines": round(gap / page_line_h, 2) if gap is not None else None,
        "size_ratio": round(b.line_height / max(1.0, a.line_height), 2),
        "rule_between": bool(same_col and rule_between_vertically(a.box, b.box, horizontal_rules)),
        "a_is_headline": a.is_headline,
        "b_is_headline": b.is_headline,
        "a_ends_sentence": bool(_SENTENCE_END.search(a_text)),
        "a_ends_hyphen": a_text.endswith("-"),
        "b_starts_lower": bool(b_text[:1].islower()),
    }


def candidate_pairs(ordered: list[Block]) -> list[tuple[Block, Block]]:
    """Neighbours in reading order: the next block down a column, or the top of the next column."""
    return list(itertools.pairwise(ordered))
