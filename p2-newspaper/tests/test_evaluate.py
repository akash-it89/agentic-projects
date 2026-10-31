from p2news.evaluate import box_counts, evaluate_page, grouping_counts, prf, summarize


def test_perfect_and_shifted_boxes():
    gold = [(0, 0, 100, 100), (200, 0, 300, 100)]
    assert prf(*box_counts(gold, gold, 0.5))["f1"] == 1.0
    shifted = [(60, 0, 160, 100), (200, 0, 300, 100)]       # first box IoU = 40/160 = 0.25
    assert box_counts(shifted, gold, 0.5) == (1, 2, 2)


def test_grouping_counts_split_article():
    gold_lines = [{"box": (0, y, 100, y + 10), "article_id": "A"} for y in (0, 20, 40, 60)]
    one = [{"article_id": "p", "bbox": (0, 0, 100, 70)}]
    assert grouping_counts(one, gold_lines) == (6, 6, 6)
    split = [{"article_id": "p", "bbox": (0, 0, 100, 30)}, {"article_id": "q", "bbox": (0, 35, 100, 70)}]
    assert grouping_counts(split, gold_lines) == (2, 2, 6)   # precision 1.0, recall 1/3


def test_evaluate_page_and_summary_with_cer():
    gold = {"page_id": "p", "articles": [{"article_id": "A", "bbox": (0, 0, 100, 100), "text": "the mayor said"}],
            "lines": []}
    pred = {"articles": [{"article_id": "x", "bbox": (0, 0, 100, 100), "text": "the mayer said"}]}
    s = summarize([evaluate_page(pred, gold)])
    assert s["box@0.5"]["f1"] == 1.0
    assert s["cer"] == round(1 / 14, 4)
    assert "grouping_pairwise" not in s                     # no gold lines on this page
