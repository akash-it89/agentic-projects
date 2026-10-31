from conftest import block

from p2news.grouping import build_articles, dehyphenate, group
from p2news.schema import PairDecision


def test_union_find():
    groups = sorted(sorted(g) for g in group(5, [(0, 1), (1, 2), (3, 4)]))
    assert groups == [[0, 1, 2], [3, 4]]


def test_dehyphenate():
    assert dehyphenate(["the infor-", "mation desk", "  ", "New-", "York"]) == "the information desk New- York"


def test_articles_numbered_in_reading_order_with_union_box():
    ordered = [block(0, (0, 0, 100, 50), "Headline"), block(1, (0, 60, 100, 200), "body one"),
               block(2, (0, 300, 100, 400), "Other story"), block(3, (200, 0, 300, 100), "body two", column=1)]
    decisions = [PairDecision(0, 1, True, 0.9, "rule"), PairDecision(1, 2, False, 0.9, "rule"),
                 PairDecision(2, 3, False, 0.6, "fallback")]
    arts = build_articles(ordered, decisions, "p1")
    assert [a.article_id for a in arts] == ["p1-a001", "p1-a002", "p1-a003"]
    assert arts[0].block_ids == [0, 1] and arts[0].bbox == (0, 0, 100, 200)
    assert arts[0].text == "Headline\n\nbody one"
    assert len(arts[0].regions) == 2
