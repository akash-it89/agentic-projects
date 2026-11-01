"""Chat with one document: retrieve its own chunks, answer only from them, cite line IDs.

Citations work like fields: the model cites line IDs, the code checks that they were in the retrieved
chunks and turns them into boxes, so a chat answer can be highlighted on the page too.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from .citations import UNTRUSTED_TEXT_RULE, parse_ids
from .config import Settings, get_settings
from .llm import LLM, parse_json
from .schema import ChatAnswer, Line, union

log = logging.getLogger(__name__)

NOT_FOUND = "That isn't in this document."

SYSTEM = """You answer questions about one document using only the excerpts below. Each excerpt line starts \
with its ID in brackets, like [p1-L3].
Return JSON with exactly these keys: {"found": true or false, "answer": "<short answer>", "line_ids": [...]}
Example: for the excerpt line "[p1-L7] Due date: 25/10/2026" and the question "When is it due?", return
{"found": true, "answer": "25/10/2026", "line_ids": ["p1-L7"]}
If the excerpts don't contain the answer, return {"found": false, "answer": "", "line_ids": []}.
If the question assumes something the excerpts contradict, say what the excerpts actually state.
Answer in the language of the question.
""" + UNTRUSTED_TEXT_RULE

_ID_IN_TEXT = re.compile(r"\s*[\[(]\s*p\d+\s*-\s*l\d+(?:\s*,\s*p\d+\s*-\s*l\d+)*\s*[\])]", re.IGNORECASE)
_ANSWER_KEYS = ("answer", "response", "result", "text")


def _cited_ids(raw, answer_text: str, allowed: list[str]) -> list[str]:
    """Citations from the line_ids field, plus any IDs written inside the answer itself. Only retrieved IDs
    count; see citations.parse_ids for the formats accepted."""
    items = raw if isinstance(raw, list) else [] if raw in (None, "") else [raw]
    return parse_ids([*items, answer_text], allowed)


_STOP = {"the", "is", "are", "was", "a", "an", "of", "to", "on", "in", "and", "or", "for", "by", "it", "this",
         "that", "be", "at", "as", "with", "your", "you", "will", "has", "have", "which"}


def _ground_answer(answer_text: str, retrieved_lines: list[Line]) -> list[str]:
    """No usable citation: find the retrieved line(s) that contain the answer's content. Most of the answer's
    meaningful words (or all its numbers) must appear in one line, so a made-up answer still finds nothing."""
    tokens = [t for t in tokenize(answer_text) if t not in _STOP]
    if not tokens:
        return []
    numbers = {t for t in tokens if any(c.isdigit() for c in t)}
    best, best_score = [], 0.0
    for ln in retrieved_lines:
        have = set(tokenize(ln.text))
        score = sum(t in have for t in tokens) / len(tokens)
        if numbers and not numbers <= have:
            continue  # a date or amount in the answer must literally be on the line
        if score > best_score:
            best, best_score = [ln.id], score
        elif score == best_score and score > 0:
            best.append(ln.id)
    return best[:3] if best_score >= 0.6 else []


def _is_true(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


@dataclass
class Chunk:
    line_ids: list[str]
    text: str  # "[p1-L3] ...\n[p1-L4] ..."


def make_chunks(lines: list[Line], size: int = 5) -> list[Chunk]:
    """Neighbouring lines on the same page, `size` at a time, overlapping by one line so a fact split
    across a chunk boundary is still retrievable."""
    chunks = []
    step = max(1, size - 1)
    pages: dict[int, list[Line]] = {}
    for ln in lines:
        pages.setdefault(ln.page, []).append(ln)
    for page_lines in pages.values():
        for start in range(0, len(page_lines), step):
            group = page_lines[start:start + size]
            chunks.append(Chunk([ln.id for ln in group], "\n".join(f"[{ln.id}] {ln.text}" for ln in group)))
            if start + size >= len(page_lines):
                break
    return chunks


_TOKEN = re.compile(r"\w+", re.UNICODE)


_LATIN_WORD = re.compile(r"[a-z]+")


def _stem(t: str) -> str:
    """Crude English suffix stripping so "expire", "expires" and "expired" match. Not linguistically correct,
    just consistent: questions and document go through the same function. Non-Latin words are left alone."""
    if not _LATIN_WORD.fullmatch(t):
        return t
    for suffix in ("ing", "ed", "es", "s"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            t = t[: -len(suffix)]
            break
    return t[:-1] if t.endswith("e") and len(t) > 3 else t


def tokenize(text: str) -> list[str]:
    # \w covers Tamil and Arabic letters too; drop the [p1-L3] ID tokens so they don't match questions
    return [_stem(t) for t in _TOKEN.findall(text.casefold()) if not re.fullmatch(r"p\d+|l\d+", t)]


class BM25:
    """Classic keyword ranking. No model to download, any script works; a good baseline before embeddings."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths) if self.docs else 0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(question: str, chunks: list[Chunk], s: Settings) -> list[Chunk]:
    if not chunks:
        return []
    # A card or a bill fits in the prompt whole. Sending everything means retrieval can't miss the answer
    # because the question used different words ("expire" vs "Expires"); only long documents need ranking.
    if len({i for c in chunks for i in c.line_ids}) <= s.full_context_lines:
        return chunks
    if s.retriever == "ollama":
        try:
            import ollama

            client = ollama.Client(host=s.ollama_host)
            vecs = client.embed(model=s.embed_model, input=[question] + [c.text for c in chunks])["embeddings"]
            scores = [_cosine(vecs[0], v) for v in vecs[1:]]
        except Exception as e:  # noqa: BLE001  embeddings are an upgrade; never fail the chat because of them
            log.warning("Embedding retrieval unavailable (%s); using BM25", e)
            scores = BM25([c.text for c in chunks]).scores(question)
    else:
        scores = BM25([c.text for c in chunks]).scores(question)
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])[: s.top_chunks]
    return [chunks[i] for i in ranked if scores[i] > 0] or [chunks[i] for i in ranked[:2]]


def answer(question: str, lines: list[Line], llm: LLM, s: Settings | None = None) -> ChatAnswer:
    s = s or get_settings()
    question = question.strip()
    if not question:
        return ChatAnswer(answer="Ask a question about this document.", found=False)
    retrieved = retrieve(question, make_chunks(lines, s.chunk_lines), s)
    allowed = list(dict.fromkeys(i for c in retrieved for i in c.line_ids))
    excerpts = "\n\n".join(c.text for c in retrieved)
    reply = llm.complete(SYSTEM, f"EXCERPTS:\n{excerpts}\n\nQUESTION: {question}", max_tokens=400)
    try:
        data = parse_json(reply)
    except ValueError:
        log.warning("Chat reply wasn't JSON; refusing. Reply: %.300r", reply)
        return ChatAnswer(answer=NOT_FOUND, found=False, reason="unreadable_reply")

    text = next((str(data[k]).strip() for k in _ANSWER_KEYS if str(data.get(k) or "").strip()), "")
    # A missing "found" with a real answer means found; an explicit false is respected.
    found = _is_true(data["found"]) if "found" in data else bool(text)
    if not found or not text:
        log.info("Model says the answer isn't in the document. Reply: %.300r", reply)
        return ChatAnswer(answer=NOT_FOUND, found=False, reason="not_in_document")

    by_id = {ln.id: ln for ln in lines}
    ids = _cited_ids(data.get("line_ids"), text, allowed)
    if not ids:
        ids = _ground_answer(text, [by_id[i] for i in allowed])
    if not ids:
        # Never show an answer we can't point to on the page: this is the guard against made-up answers.
        log.info("Answer %r has no citation or matching line among the excerpts; refusing", text)
        return ChatAnswer(answer=NOT_FOUND, found=False, reason="no_supporting_line")
    text = _ID_IN_TEXT.sub("", text).strip()

    boxes = []
    for page in sorted({by_id[i].page for i in ids}):
        boxes.append({"page": page, "bbox": union(by_id[i].bbox for i in ids if by_id[i].page == page)})
    return ChatAnswer(answer=text, found=True, line_ids=ids, boxes=boxes)
