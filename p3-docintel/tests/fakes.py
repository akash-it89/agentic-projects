"""A fake LLM that behaves like a careful model, so the pipeline can be tested without Ollama or AWS.

It answers each prompt type by reading the numbered lines it was given, just as a real model would,
which means the grounding code (line IDs -> boxes) is exercised for real.
"""

import json
import re

LINE = re.compile(r"^\[(p\d+-L\d+)\] (.*)$", re.MULTILINE)


def _norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", t.casefold())


class FakeLLM:
    name = "fake"

    def __init__(self, values: dict[str, str] | None = None, doc_type: str = "utility_bill",
                 answers: dict[str, str] | None = None):
        self.values = values or {}      # field name -> value to "extract" when it's on a line
        self.doc_type = doc_type
        self.answers = answers or {}    # question keyword -> text that must appear in a line
        self.calls: list[str] = []

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 1500) -> str:
        lines = LINE.findall(user)
        if system.startswith("You extract information"):
            self.calls.append("extract")
            fields = []
            for name, value in self.values.items():
                ids = [i for i, text in lines if _norm(value) in _norm(text)]
                if ids:
                    fields.append({"name": name, "value": value, "type": "text", "line_ids": ids[:1]})
            fields.append({"name": "made_up", "value": "nonsense", "line_ids": ["p9-L999"]})  # must be dropped
            return json.dumps({"fields": fields})
        if system.startswith("Classify"):
            self.calls.append("classify")
            return json.dumps({"doc_type": self.doc_type, "confidence": 0.9})
        if system.startswith("Summarise"):
            self.calls.append("summary")
            return "```json\n" + json.dumps({"summary": "A sample document."}) + "\n```"
        if system.startswith("You answer questions"):
            self.calls.append("chat")
            q = user.rsplit("QUESTION:", 1)[-1].lower()
            for keyword, needle in self.answers.items():
                if keyword in q:
                    ids = [i for i, text in lines if _norm(needle) in _norm(text)]
                    if ids:
                        return json.dumps({"found": True, "answer": needle, "line_ids": ids[:1]})
            return json.dumps({"found": False, "answer": "", "line_ids": []})
        raise AssertionError(f"unexpected prompt: {system[:40]}")
