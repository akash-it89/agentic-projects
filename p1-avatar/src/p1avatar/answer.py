"""Grounded answering. Retrieval always runs first, and the refusal rule is enforced here in code.

If retrieval finds nothing above the threshold, the LLM is never called, so the kiosk cannot make
up an answer. That is the trainer's "no wrong information" rule.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from .lang import LLM_LANGUAGE, REFUSAL, safe_lang
from .llm import LLM, get_llm

NO_ANSWER = "NO_ANSWER"

SYSTEM_PROMPT = f"""You are the information assistant on an airport kiosk.
Rules:
- Answer ONLY with facts stated in the passages. Do not add facts from your own knowledge.
- If the passages do not contain the answer, reply with exactly: {NO_ANSWER}
- Reply in {{language}}, even if the passages are in another language.
- Use at most two short sentences and no lists or Markdown: the answer is spoken aloud."""


@dataclass
class Answer:
    text: str
    lang: str
    refused: bool
    reason: str  # "answered" | "no_passages" | "llm_no_answer" | "empty_question"
    passages: list[dict] = field(default_factory=list)
    llm: str = ""


def build_user_prompt(question: str, passages: list[dict]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    return f"Passages:\n{numbered}\n\nQuestion: {question}"


def answer(question: str, lang: str | None, *, search: Callable[[str], dict] | None = None,
           llm: LLM | None = None) -> Answer:
    lang = safe_lang(lang)
    if not question.strip():
        return Answer(REFUSAL[lang], lang, True, "empty_question")

    if search is None:
        from .rag.search import search_airport_info as search
    ctx = search(question)
    if not ctx.get("found"):
        return Answer(REFUSAL[lang], lang, True, "no_passages")

    llm = llm or get_llm()
    text = llm.complete(SYSTEM_PROMPT.format(language=LLM_LANGUAGE[lang]), build_user_prompt(question, ctx["passages"]))
    if not text or NO_ANSWER in text:
        return Answer(REFUSAL[lang], lang, True, "llm_no_answer", ctx["passages"], llm.name)
    return Answer(text, lang, False, "answered", ctx["passages"], llm.name)
