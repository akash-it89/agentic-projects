"""Chat must cope with how small local models really write JSON, without ever showing an unsupported answer.

Before this was handled, 7 of 8 correct-but-imperfect replies were thrown away as "not in this document".
"""

import json

import pytest

from p3docintel.chat import NOT_FOUND, answer
from p3docintel.schema import Line


def line(i, text, y):
    return Line(id=f"p1-L{i}", page=1, text=text, bbox=(0.1, y, 0.6, y + 0.02), engine="pdf_text")


LINES = [line(1, "SAMPLE POWER & WATER CO.", 0.05), line(2, "Account number: ACC-4471-0092", 0.2),
         line(3, "Due date: 25/10/2026", 0.3), line(4, "Amount due: USD 87.40", 0.35),
         line(5, "Please pay by the due date to avoid a late fee of USD 5.00.", 0.5)]


class Scripted:
    name = "scripted"

    def __init__(self, reply):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)

    def complete(self, system, user, json_mode=True, max_tokens=1500):
        return self.reply


@pytest.mark.parametrize("reply, expected_text, expected_id", [
    ({"found": True, "answer": "25/10/2026", "line_ids": ["p1-L3"]}, "25/10/2026", "p1-L3"),
    ({"found": True, "answer": "25/10/2026", "line_ids": ["[p1-L3]"]}, "25/10/2026", "p1-L3"),
    ({"found": True, "answer": "25/10/2026", "line_ids": "p1-L3"}, "25/10/2026", "p1-L3"),
    ({"found": True, "answer": "25/10/2026", "line_ids": ["L3"]}, "25/10/2026", "p1-L3"),
    ({"found": True, "answer": "The payment is due on 25/10/2026."}, "The payment is due on 25/10/2026.", "p1-L3"),
    ({"answer": "25/10/2026", "line_ids": ["p1-L3"]}, "25/10/2026", "p1-L3"),
    ({"found": "true", "answer": "USD 5.00", "line_ids": ["p1-L5"]}, "USD 5.00", "p1-L5"),
    ({"found": True, "answer": "The late fee is USD 5.00 [p1-L5]", "line_ids": []}, "The late fee is USD 5.00", "p1-L5"),
    ({"found": True, "response": "25/10/2026", "line_ids": ["p1-L3"]}, "25/10/2026", "p1-L3"),
    ('Here is the answer:\n```json\n{"found": true, "answer": "USD 87.40", "line_ids": ["p1 - L4"]}\n```',
     "USD 87.40", "p1-L4"),
], ids=["ideal", "brackets", "string", "short-id", "no-ids", "no-found", "found-string", "ids-in-answer",
        "renamed-key", "fenced-and-spaced"])
def test_imperfect_but_correct_replies_are_answered_with_a_highlight(reply, expected_text, expected_id):
    a = answer("When is the payment due?", LINES, Scripted(reply))
    assert a.found and a.answer == expected_text
    assert expected_id in a.line_ids and a.boxes and a.boxes[0]["page"] == 1


@pytest.mark.parametrize("reply, reason", [
    ({"found": False, "answer": "", "line_ids": []}, "not_in_document"),
    ({"found": "false", "answer": "25/10/2026", "line_ids": ["p1-L3"]}, "not_in_document"),
    # made up: the date isn't on any line, and it cites nothing
    ({"found": True, "answer": "The due date is 31/12/2026"}, "no_supporting_line"),
    # cites a line that wasn't retrieved / doesn't exist, and the words aren't in the excerpts
    ({"found": True, "answer": "Your contract renews automatically", "line_ids": ["p7-L99"]}, "no_supporting_line"),
    ("I think it is due soon.", "unreadable_reply"),
], ids=["refusal", "found-false-string", "invented-date", "invented-citation", "not-json"])
def test_unsupported_answers_are_still_refused(reply, reason):
    a = answer("When is the payment due?", LINES, Scripted(reply))
    assert not a.found and a.answer == NOT_FOUND and a.reason == reason


def test_ambiguous_short_id_is_not_guessed():
    two_pages = [*LINES, Line(id="p2-L3", page=2, text="Due date for the next bill: 25/11/2026",
                              bbox=(0.1, 0.3, 0.6, 0.32), engine="pdf_text")]
    a = answer("When is the payment due?", two_pages,
               Scripted({"found": True, "answer": "something unrelated", "line_ids": ["L3"]}))
    assert not a.found  # "L3" exists on both pages, and the answer text matches neither


def test_api_explains_when_the_model_is_unreachable(monkeypatch, samples):
    from fakes import FakeLLM
    from fastapi.testclient import TestClient

    from p3docintel import api, pipeline
    from p3docintel.samples import BILL_FIELDS

    created = pipeline.create("bill.pdf", (samples / "docs" / "utility_bill.pdf").read_bytes())
    pipeline.process(created.doc_id, llm=FakeLLM(dict(BILL_FIELDS)))

    class Down:
        name = "down"

        def complete(self, *a, **k):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(api, "get_llm", lambda model=None: Down())
    res = TestClient(api.app).post(f"/api/documents/{created.doc_id}/chat", json={"question": "Due date?"})
    assert res.status_code == 503 and "ollama pull" in res.json()["detail"]
