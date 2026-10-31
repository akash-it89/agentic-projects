from p1avatar.answer import NO_ANSWER, answer
from p1avatar.lang import REFUSAL


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str):
        self.reply, self.calls = reply, []

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        self.calls.append((system, user))
        return self.reply


FOUND = {"found": True, "passages": [{"text": "Gate C22 is in Terminal 2.", "file": "en/a.md", "lang": "en", "score": 0.9}]}
NOT_FOUND = {"found": False, "passages": []}


def test_no_passages_refuses_without_calling_llm():
    llm = FakeLLM("Gate C22 is on the moon.")
    a = answer("Where is the pharmacy?", "es", search=lambda q: NOT_FOUND, llm=llm)
    assert a.refused and a.reason == "no_passages"
    assert a.text == REFUSAL["es"]
    assert llm.calls == []


def test_llm_sentinel_becomes_refusal_in_user_language():
    a = answer("¿Dónde está la farmacia?", "hi", search=lambda q: FOUND, llm=FakeLLM(NO_ANSWER))
    assert a.refused and a.reason == "llm_no_answer"
    assert a.text == REFUSAL["hi"]


def test_answer_uses_passages_and_language():
    llm = FakeLLM("Gate C22 is in Terminal 2.")
    a = answer("Where is gate C22?", "en", search=lambda q: FOUND, llm=llm)
    assert not a.refused and a.text == "Gate C22 is in Terminal 2."
    system, user = llm.calls[0]
    assert "English" in system
    assert "[1] Gate C22 is in Terminal 2." in user


def test_hindi_prompt_asks_for_devanagari():
    llm = FakeLLM("गेट C22 टर्मिनल 2 में है।")
    answer("गेट C22 कहाँ है?", "hi", search=lambda q: FOUND, llm=llm)
    assert "Devanagari" in llm.calls[0][0]


def test_unknown_language_falls_back_to_english_and_empty_question_refuses():
    a = answer("   ", "fr", search=lambda q: FOUND, llm=FakeLLM("x"))
    assert a.refused and a.lang == "en" and a.reason == "empty_question"
