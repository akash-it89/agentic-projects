from conftest import line

from p2news.blocks import join_row_fragments, mark_headlines, merge_lines
from p2news.layout import assign_columns, reading_order


def column(x0, y_start, n, step=30, h=20, width=400):
    return [line(x0, y_start + i * step, x0 + width, y_start + i * step + h) for i in range(n)]


def test_stacked_lines_merge_into_one_block():
    blocks = merge_lines(column(0, 0, 5))
    assert len(blocks) == 1 and len(blocks[0].lines) == 5


def test_big_gap_starts_a_new_block():
    blocks = merge_lines(column(0, 0, 3) + column(0, 300, 3))
    assert len(blocks) == 2


def test_columns_stay_separate():
    blocks = merge_lines(column(0, 0, 4) + column(500, 0, 4))
    assert len(blocks) == 2


def test_rule_between_lines_blocks_the_merge():
    rule = (0, 44, 400, 47)  # printed rule in the gap between the second line (ends y=42) and the third
    lines = [line(0, 0, 400, 20), line(0, 30, 400, 42), line(0, 50, 400, 70)]
    assert len(merge_lines(lines, horizontal_rules=[rule])) == 2


def test_headline_line_does_not_join_body_and_is_marked():
    lines = [line(0, 0, 400, 40, "Big Headline")] + column(0, 52, 4)
    blocks = merge_lines(lines)
    assert len(blocks) == 2
    mark_headlines(blocks, page_line_h=20)
    assert blocks[0].is_headline and not blocks[1].is_headline


def test_two_line_headline_stays_together():
    lines = [line(0, 0, 400, 40, "Council Approves"), line(0, 56, 300, 96, "Bridge Plan")]
    assert len(merge_lines(lines + column(0, 110, 3))) == 2


def test_row_fragments_join_but_not_across_gutter_or_rule():
    pieces = [line(0, 0, 100, 20, "the council"), line(108, 0, 200, 20, "met today")]
    assert [ln.text for ln in join_row_fragments(pieces)] == ["the council met today"]
    across_gutter = [line(0, 0, 100, 20), line(160, 0, 260, 20)]
    assert len(join_row_fragments(across_gutter)) == 2
    with_rule = [line(0, 0, 100, 20), line(108, 0, 200, 20)]
    assert len(join_row_fragments(with_rule, vertical_rules=[(103, -5, 105, 30)])) == 2


def test_columns_and_reading_order():
    blocks = merge_lines(column(500, 0, 3) + column(0, 0, 3) + column(0, 300, 2))
    n = assign_columns(blocks, page_width=1000)
    assert n == 2
    order = reading_order(blocks)
    assert [b.column for b in order] == [0, 0, 1]
    assert order[0].box[1] < order[1].box[1]
    assert [b.column for b in reading_order(blocks, rtl=True)] == [1, 0, 0]
