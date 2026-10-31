import json

from p2news.labelstudio import convert


def test_convert_percent_boxes_and_text(tmp_path):
    export = [{
        "data": {"image": "/data/upload/1/3f2a9c1b-page_0042.png"},
        "annotations": [{"result": [
            {"id": "r1", "type": "rectanglelabels", "original_width": 2000, "original_height": 3000,
             "value": {"x": 10, "y": 20, "width": 50, "height": 10, "rectanglelabels": ["Article"]}},
            {"id": "r1", "type": "textarea", "value": {"text": ["Mayor speaks", "on budget"]}},
            {"id": "r2", "type": "rectanglelabels", "original_width": 2000, "original_height": 3000,
             "value": {"x": 0, "y": 0, "width": 5, "height": 5, "rectanglelabels": ["Advert"]}},
        ]}],
    }]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    [out] = convert(path, tmp_path / "gold")
    gold = json.loads(out.read_text(encoding="utf-8"))
    assert gold["page_id"] == "page_0042"
    assert len(gold["articles"]) == 1
    assert gold["articles"][0]["bbox"] == [200.0, 600.0, 1200.0, 900.0]
    assert gold["articles"][0]["text"] == "Mayor speaks on budget"
