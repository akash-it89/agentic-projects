import json
import time

from fakes import FakeLLM
from fastapi.testclient import TestClient

from p3docintel import pipeline
from p3docintel.evaluate import evaluate
from p3docintel.ingest import render
from p3docintel.samples import BILL_FIELDS, LEASE_FIELDS_P1, LEASE_FIELDS_P2
from p3docintel.schema import iou


def bill_llm():
    return FakeLLM(dict(BILL_FIELDS), "utility_bill", {"due": "25/10/2026", "late fee": "USD 5.00"})


def test_text_layer_lines_are_normalized_and_unique(samples, tmp_path):
    pages = render(samples / "docs" / "utility_bill.pdf", tmp_path / "pages")
    assert len(pages) == 1 and pages[0].page.source == "pdf_text" and pages[0].png.exists()
    lines = pages[0].text_lines
    assert len({x.id for x in lines}) == len(lines) > 10
    assert all(0 <= v <= 1 for x in lines for v in x.bbox)


def test_image_upload_needs_ocr(samples, tmp_path):
    [page] = render(samples / "docs" / "licence_card.png", tmp_path / "pages")
    assert page.page.source == "ocr" and page.text_lines == []


def test_bill_pdf_end_to_end_boxes_match_gold(samples):
    gold = json.loads((samples / "gold" / "utility_bill.json").read_text(encoding="utf-8"))
    created = pipeline.create("utility_bill.pdf", (samples / "docs" / "utility_bill.pdf").read_bytes())
    r = pipeline.process(created.doc_id, llm=bill_llm())
    assert r.status == "done", r.error
    assert r.language == "en" and r.doc_type == "utility_bill" and r.summary == "A sample document."
    assert "made_up" not in {f.name for f in r.fields}  # cited a line that doesn't exist
    by_name = {f.name: f for f in r.fields}
    for g in gold["fields"]:
        f = by_name[g["name"]]
        # the field box is the whole line ("Due date: 25/10/2026"); the gold box is just the value
        assert f.page == g["page"]
        gx0, gy0, gx1, _ = g["bbox"]
        assert f.bbox[0] <= gx0 + 0.01 and f.bbox[2] >= gx1 - 0.01 and abs(f.bbox[1] - gy0) < 0.01
    assert set(r.stats["seconds"]) >= {"rendering pages", "extracting fields", "summarising"}


def test_lease_fields_found_on_both_pages(samples):
    values = dict(LEASE_FIELDS_P1 + LEASE_FIELDS_P2)
    created = pipeline.create("lease.pdf", (samples / "docs" / "lease_agreement.pdf").read_bytes())
    r = pipeline.process(created.doc_id, llm=FakeLLM(values, "lease_agreement"))
    assert r.status == "done" and len(r.pages) == 2
    assert {f.name: f.page for f in r.fields}["notice_period"] == 2


class FakeOCR:
    """Stands in for PaddleOCR: returns the gold values as lines at their gold positions."""

    name = "fake-ocr"

    def __init__(self, gold):
        self.gold = gold

    def read(self, png, page_no):
        from p3docintel.schema import Line
        return [Line(id=f"p{page_no}-L{i + 1}", page=page_no, text=g["value"], bbox=tuple(g["bbox"]),
                     confidence=0.9, engine="paddle") for i, g in enumerate(self.gold["fields"])]


def test_scan_goes_through_ocr_and_boxes_follow_ocr(samples):
    gold = json.loads((samples / "gold" / "utility_bill_scan.json").read_text(encoding="utf-8"))
    created = pipeline.create("scan.png", (samples / "docs" / "utility_bill_scan.png").read_bytes(), "en")
    r = pipeline.process(created.doc_id, llm=bill_llm(), ocr=FakeOCR(gold))
    assert r.status == "done" and r.stats["engine"] == "fake-ocr"
    by_name = {f.name: f for f in r.fields}
    assert all(iou(by_name[g["name"]].bbox, tuple(g["bbox"])) > 0.99 for g in gold["fields"])


def test_broken_document_reports_an_error():
    created = pipeline.create("broken.pdf", b"this is not a pdf")
    r = pipeline.process(created.doc_id, llm=FakeLLM())
    assert r.status == "error" and r.error


def test_evaluate_on_samples_with_fakes(samples, tmp_path):
    import shutil
    sub = tmp_path / "s"
    (sub / "gold").mkdir(parents=True)
    shutil.copytree(samples / "docs", sub / "docs")
    shutil.copy(samples / "gold" / "utility_bill.json", sub / "gold")
    report = evaluate(sub, llm=bill_llm())
    s = report["summary"]
    assert s["field_recall"] == 1.0 and s["classification_accuracy"] == 1.0 and s["chat_accuracy"] == 1.0
    assert s["box_accuracy_iou50"] is not None


def test_api_upload_process_and_chat(samples, monkeypatch):
    from p3docintel import api

    llm = bill_llm()
    monkeypatch.setattr(pipeline, "get_llm", lambda model=None: llm)
    monkeypatch.setattr(api, "get_llm", lambda model=None: llm)
    client = TestClient(api.app)
    assert client.get("/").status_code == 200

    with open(samples / "docs" / "utility_bill.pdf", "rb") as f:
        res = client.post("/api/documents", files={"file": ("bill.pdf", f, "application/pdf")},
                          data={"language": "auto"})
    assert res.status_code == 202
    doc_id = res.json()["doc_id"]

    for _ in range(50):  # TestClient runs background tasks before returning, but poll like the UI does
        doc = client.get(f"/api/documents/{doc_id}").json()
        if doc["status"] != "processing":
            break
        time.sleep(0.1)
    assert doc["status"] == "done" and doc["fields"]

    page = client.get(f"/api/documents/{doc_id}/pages/1")
    assert page.status_code == 200 and page.headers["content-type"] == "image/png"

    chat = client.post(f"/api/documents/{doc_id}/chat", json={"question": "When is the payment due?"}).json()
    assert chat["found"] and chat["answer"] == "25/10/2026" and chat["boxes"]

    assert client.post("/api/documents", files={"file": ("x.exe", b"MZ")}).status_code == 415
    assert client.get("/api/documents/../../etc").status_code == 404
    assert client.get("/api/documents/zzzzzzzzzzzz").status_code == 404
