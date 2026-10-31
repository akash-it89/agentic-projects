from p2news.geometry import iou, polygon_to_box, union, vertical_gap, x_overlap_ratio


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 3) == round(50 / 150, 3)


def test_union_and_gap():
    assert union([(0, 0, 10, 10), (5, 20, 30, 25)]) == (0, 0, 30, 25)
    assert vertical_gap((0, 0, 10, 10), (0, 15, 10, 20)) == 5


def test_x_overlap_uses_narrower_box():
    assert x_overlap_ratio((0, 0, 100, 10), (20, 0, 60, 10)) == 1.0
    assert x_overlap_ratio((0, 0, 100, 10), (200, 0, 300, 10)) == 0.0


def test_polygon_to_box():
    assert polygon_to_box([[1, 2], [9, 2], [9, 8], [1, 8]]) == (1, 2, 9, 8)
