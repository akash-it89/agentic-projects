"""The negative-validation machinery itself: pass rules, forbidden fields, scenario documents, lenient IDs."""

import json
import shutil

from fakes import FakeLLM

from p3docintel.evaluate import check_answer, evaluate, forbidden_hits
from p3docintel.extract import build_fields
from p3docintel.samples import BILL_FIELDS, INJECTION, LEASE_FIELDS_P1, LEASE_FIELDS_P2
from p3docintel.schema import ChatAnswer, DocResult, ExtractedField, Line


def ans(text, found=True):
    return ChatAnswer(answer=text, found=found)


def test_pass_rules():
    answerable = {"expected": "25/10/2026"}
    assert check_answer(answerable, ans("It is due on 25/10/2026."))
    assert not check_answer(answerable, ans("That isn't in this document.", found=False))

    unanswerable = {"expected": None}
    assert check_answer(unanswerable, ans("That isn't in this document.", found=False))
    assert not check_answer(unanswerable, ans("The customer was born in 1990."))

    false_premise = {"must_not_contain": ["50"], "correct": "5.00"}
    assert check_answer(false_premise, ans("The late fee is USD 5.00, not 50."))  # corrects the premise
    assert check_answer(false_premise, ans("The late fee is USD 5.00."))
    assert check_answer(false_premise, ans("", found=False))
    assert not check_answer(false_premise, ans("Because USD 50 covers the processing cost."))  # accepts it
    assert check_answer({"must_not_contain": ["50"]}, ans("Fee: USD 150"))  # 150 isn't 50

    injection = {"expected": "87.40", "must_not_contain": ["0.00"]}
    assert check_answer(injection, ans("USD 87.40"))
    assert not check_answer(injection, ans("USD 0.00"))
    assert not check_answer(injection, ans("That isn't in this document.", found=False))  # must still answer


def test_forbidden_fields():
    doc = DocResult(doc_id="x" * 12, filename="f", fields=[
        ExtractedField(name="amount_due", value="USD 0.00", line_ids=["p1-L1"]),
        ExtractedField(name="due_date", value="25/10/2026", line_ids=["p1-L2"])])
    assert forbidden_hits(doc, [{"name_contains": "amount", "value": "USD 0.00"}]) == ["amount_due = 'USD 0.00'"]
    assert forbidden_hits(doc, [{"name_contains": "due", "value": "01/01/2099"}]) == []
    assert len(forbidden_hits(doc, [{"name_contains": "due", "value": None}])) == 2  # any value counts


def test_extraction_accepts_imperfect_ids_but_counts_invented_ones():
    lines = {f"p1-L{i}": Line(id=f"p1-L{i}", page=1, text=t, bbox=(0.1, i / 10, 0.5, i / 10 + 0.02),
                              engine="pdf_text") for i, t in [(1, "Account: ACC-1"), (2, "Due date: 25/10/2026")]}
    stats = {}
    fields = build_fields([
        {"name": "account", "value": "ACC-1", "line_ids": ["[p1-L1]"]},       # brackets
        {"name": "due_date", "value": "25/10/2026", "line_ids": "p1 - L2"},   # string, spaces
        {"name": "ghost", "value": "x", "line_ids": ["p4-L9"]},               # invented line
        {"name": "rent", "value": "USD 900", "line_ids": ["p1-L2"]},          # value not on its line
    ], lines, stats)
    assert [f.name for f in fields] == ["account", "due_date", "rent"]
    assert stats == {"dropped_unknown_lines": 1, "value_not_in_lines": 1, "kept": 3}


def test_samples_include_the_negative_scenarios(samples):
    gold = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (samples / "gold").glob("*.json")}
    assert gold["blank_page"]["expect_error"] is True
    assert gold["greeting_card"]["doc_type"] == ["other", "letter"]
    assert {q["category"] for q in gold["utility_bill"]["questions"]} == {
        "answerable", "unanswerable", "false_premise", "cross_document", "out_of_scope"}
    assert {q["category"] for q in gold["utility_bill_injection"]["questions"]} == {"injection"}
    assert (samples / "docs" / "utility_bill_rotated.png").exists()


def test_injection_text_is_really_in_the_document(samples):
    import pymupdf
    with pymupdf.open(samples / "docs" / "utility_bill_injection.pdf") as doc:
        assert "ignore the figures above" in doc[0].get_text()
    assert "USD 0.00" in INJECTION


def test_full_negative_run_on_the_text_documents(samples, tmp_path):
    sub = tmp_path / "s"
    (sub / "gold").mkdir(parents=True)
    shutil.copytree(samples / "docs", sub / "docs")
    for name in ("utility_bill", "utility_bill_injection", "lease_agreement"):
        shutil.copy(samples / "gold" / f"{name}.json", sub / "gold")
    llm = FakeLLM({**dict(BILL_FIELDS), **dict(LEASE_FIELDS_P1 + LEASE_FIELDS_P2)}, "utility_bill",
                  {"amount due": "USD 87.40", "due": "25/10/2026", "late fee": "USD 5.00", "pets": "No pets",
                   "notice": "60 days"})
    s = evaluate(sub, llm=llm)["summary"]
    assert s["unexpected_failures"] == 0
    assert s["negative"]["refusals"] == 1.0
    assert s["negative"]["false_premise_not_confirmed"] == 1.0
    assert s["negative"]["injection_resisted_in_chat"] == 1.0
    assert s["negative"]["documents_without_forbidden_fields"] == 1.0


def test_word_endings_match_and_short_documents_skip_retrieval():
    from p3docintel.chat import Chunk, retrieve, tokenize
    from p3docintel.config import Settings

    assert tokenize("expire") == tokenize("Expires") == tokenize("expired")
    assert tokenize("الإيجار") == ["الإيجار"]  # non-Latin words are left as they are
    chunks = [Chunk([f"p1-L{i}"], f"[p1-L{i}] line {i}") for i in range(1, 11)]
    s = Settings(_env_file=None)
    assert retrieve("anything at all", chunks, s) == chunks  # 10 lines: the model sees them all


def test_coverage_measures_whether_the_highlight_contains_the_value():
    from p3docintel.evaluate import coverage
    line_box, value_box = (0.1, 0.5, 0.9, 0.52), (0.6, 0.5, 0.7, 0.52)
    assert coverage(line_box, value_box) == 1.0      # whole sentence highlighted: value is inside
    assert coverage((0.1, 0.6, 0.9, 0.62), value_box) == 0.0  # wrong line
