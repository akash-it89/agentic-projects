"""Chat with one document: retrieve its own chunks, answer only from them, cite line IDs.

Citations work like fields: the model cites line IDs, the code checks that they were in the retrieved
chunks and turns them into boxes, so a chat answer can be highlighted on the page too.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from .config import Settings, get_settings
from .llm import LLM, parse_json
from .schema import ChatAnswer, Line, union

log = logging.getLogger(__name__)

NOT_FOUND = "That isn't in this document."

SYSTEM = """You answer questions about one document using only the excerpts below. Each excerpt line starts \
with its ID, like [p1-L3].
Return JSON: {"found": true or false, "answer": "<short answer>", "line_ids": ["<IDs of the lines you used>"]}
If the excerpts don't contain the answer, return {"found": false, "answer": "", "line_ids": []}.
Answer in the language of the question."""


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


def tokenize(text: str) -> list[str]:
    # \w covers Tamil and Arabic letters too; drop the [p1-L3] ID tokens so they don't match questions
    return [t for t in _TOKEN.findall(text.casefold()) if not re.fullmatch(r"p\d+|l\d+", t)]


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
    allowed = {i for c in retrieved for i in c.line_ids}
    excerpts = "\n\n".join(c.text for c in retrieved)
    try:
        data = parse_json(llm.complete(SYSTEM, f"EXCERPTS:\n{excerpts}\n\nQUESTION: {question}", max_tokens=400))
    except ValueError as e:
        log.warning("Chat answer returned unusable JSON: %s", e)
        return ChatAnswer(answer=NOT_FOUND, found=False)

    ids = [i for i in data.get("line_ids") or [] if i in allowed]  # cite only what was actually retrieved
    text = str(data.get("answer", "")).strip()
    if not data.get("found") or not ids or not text:
        return ChatAnswer(answer=NOT_FOUND, found=False)

    by_id = {ln.id: ln for ln in lines}
    boxes = []
    for page in sorted({by_id[i].page for i in ids}):
        boxes.append({"page": page, "bbox": union(by_id[i].bbox for i in ids if by_id[i].page == page)})
    return ChatAnswer(answer=text, found=True, line_ids=ids, boxes=boxes)
