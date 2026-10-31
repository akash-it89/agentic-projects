from conftest import block

from p2news.decider import CascadeDecider, fallback_decision, rule_decision

BASE = {"same_column": True, "column_jump": False, "gap_lines": 0.8, "size_ratio": 1.0, "rule_between": False,
        "a_is_headline": False, "b_is_headline": False, "a_ends_sentence": True, "a_ends_hyphen": False,
        "b_starts_lower": False}


def f(**kw):
    return {**BASE, **kw}


def test_rules():
    assert rule_decision(f(rule_between=True)) == (False, 0.95)
    assert rule_decision(f(b_is_headline=True))[0] is False
    assert rule_decision(f(a_is_headline=True, gap_lines=1.8))[0] is True
    assert rule_decision(f(a_is_headline=True, b_is_headline=True, gap_lines=1.0))[0] is True
    assert rule_decision(f(gap_lines=1.2))[0] is True
    assert rule_decision(f(gap_lines=5.0))[0] is False
    assert rule_decision(f(gap_lines=2.5)) is None                         # unclear: left to the LLM
    assert rule_decision(f(same_column=False, column_jump=True, gap_lines=None)) is None


def test_fallback_for_column_jumps():
    jump = f(same_column=False, column_jump=True, gap_lines=None)
    assert fallback_decision({**jump, "a_ends_sentence": False, "b_starts_lower": True})[0] is True
    assert fallback_decision({**jump, "a_ends_hyphen": True})[0] is True
    assert fallback_decision(jump)[0] is False


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def decide(self, a, b, feats):
        self.calls += 1
        return self.answer


def test_cascade_only_asks_llm_when_rules_are_unsure():
    llm = FakeLLM((True, 0.7))
    d = CascadeDecider(llm)
    a, b = block(0, (0, 0, 10, 10)), block(1, (0, 50, 10, 60))
    assert d.decide(a, b, f(gap_lines=1.0)).source == "rule"
    assert llm.calls == 0
    result = d.decide(a, b, f(gap_lines=2.5))
    assert result.source == "llm" and result.same_article and llm.calls == 1
    assert dict(d.counts) == {"rule": 1, "llm": 1}


def test_cascade_falls_back_when_llm_has_no_answer():
    d = CascadeDecider(FakeLLM(None))
    a, b = block(0, (0, 0, 10, 10)), block(1, (500, 0, 510, 10), column=1)
    assert d.decide(a, b, f(same_column=False, column_jump=True, gap_lines=None)).source == "fallback"
