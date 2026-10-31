import pytest
from fakes import FakeLLM

from p3docintel.chat import BM25, NOT_FOUND, answer, make_chunks
from p3docintel.classify import classify
from p3docintel.extract import build_fields
from p3docintel.langdetect import detect_script
from p3docintel.llm import parse_json
from p3docintel.schema import Line, iou, union


def ln(i, text, page=1, y=0.1, conf=1.0):
    return Line(id=f"p{page}-L{i}", page=page, text=text, bbox=(0.1, y, 0.5, y + 0.02), confidence=conf,
                engine="paddle")


def test_parse_json_handles_fences_and_chatter():
    assert parse_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"b": [1, 2]} hope that helps') == {"b": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_box_helpers():
    assert union([(0, 0, 0.2, 0.2), (0.1, 0.5, 0.3, 0.6)]) == (0, 0, 0.3, 0.6)
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0


def test_build_fields_grounding_rules():
    lines = {x.id: x for x in [ln(1, "Account number: ACC-1", y=0.1), ln(2, "Due date: 25/10/2026", y=0.2, conf=0.8),
                               ln(1, "Signed", page=2)]}
    raw = [
        {"name": "Account Number", "value": "ACC-1", "type": "id", "line_ids": ["p1-L1"]},
        {"name": "due_date", "value": "25 October 2026", "type": "date", "line_ids": ["p1-L2"]},  # reformatted
        {"name": "ghost", "value": "x", "line_ids": ["p7-L1"]},                                   # hallucinated ID
        {"name": "account_number", "value": "ACC-1", "line_ids": ["p1-L1"]},                      # duplicate
        {"name": "span", "value": "ACC-1", "type": "weird", "line_ids": ["p1-L1", "p2-L1"]},       # two pages
    ]
    fields = build_fields(raw, lines)
    assert [f.name for f in fields] == ["account_number", "due_date", "span"]
    acc, due, span = fields
    assert acc.bbox == lines["p1-L1"].bbox and acc.confidence == 1.0 and acc.type == "id"
    assert due.confidence == 0.5 and due.attributes["value_found_in_lines"] is False
    assert span.page == 1 and span.bbox == lines["p1-L1"].bbox and span.type == "text"


def test_classify_rejects_invented_types():
    assert classify([ln(1, "Electricity bill")], FakeLLM(doc_type="utility_bill")) == ("utility_bill", 0.9)
    assert classify([ln(1, "x")], FakeLLM(doc_type="spaceship_manual")) == ("other", 0.0)


def test_bm25_ranks_the_matching_chunk_first():
    scores = BM25(["the rent is due monthly", "pets are not allowed", "the deposit is returned"]).scores("pets?")
    assert scores.index(max(scores)) == 1


def test_bm25_works_for_tamil_and_arabic_tokens():
    scores = BM25(["வாடகை மாதம் 1250", "الإيجار الشهري ١٢٥٠"]).scores("الإيجار")
    assert scores[1] > scores[0]


def test_chunks_overlap_and_stay_on_one_page():
    lines = [ln(i, f"line {i}", y=i / 20) for i in range(1, 10)] + [ln(1, "page two", page=2)]
    chunks = make_chunks(lines, size=5)
    assert chunks[0].line_ids[-1] == chunks[1].line_ids[0]  # one line of overlap
    assert all(len({i.split("-")[0] for i in c.line_ids}) == 1 for c in chunks)


def test_chat_cites_only_retrieved_lines_and_refuses_otherwise(isolated_settings):
    lines = [ln(1, "Monthly rent: USD 1,250", y=0.1), ln(2, "No pets are allowed", y=0.3)]
    llm = FakeLLM(answers={"pets": "No pets are allowed"})
    a = answer("Are pets allowed?", lines, llm)
    assert a.found and a.line_ids == ["p1-L2"] and a.boxes == [{"page": 1, "bbox": lines[1].bbox}]
    missing = answer("Who pays for electricity?", lines, llm)
    assert not missing.found and missing.answer == NOT_FOUND


def test_detect_script():
    assert detect_script("Driving licence") == "en"
    assert detect_script("ஓட்டுநர் உரிமம்") == "ta"
    assert detect_script("رخصة القيادة") == "ar"
    assert detect_script("1234 --") is None
