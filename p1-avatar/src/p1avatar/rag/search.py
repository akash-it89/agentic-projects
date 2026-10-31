"""Retrieval + reranking. The refusal threshold lives here: below MIN_SCORE nothing is returned."""

import math
from functools import lru_cache
from typing import Any

from ..config import Settings, get_settings


def _to_probability(scores: list[float]) -> list[float]:
    # Depending on the sentence-transformers version, CrossEncoder returns raw logits or sigmoid scores.
    # Normalize so MIN_SCORE always means the same thing.
    if all(0.0 <= s <= 1.0 for s in scores):
        return scores
    return [1 / (1 + math.exp(-s)) for s in scores]


class Retriever:
    def __init__(self, settings: Settings | None = None, embedder: Any = None, reranker: Any = None,
                 client: Any = None):
        self.s = settings or get_settings()
        self._embedder, self._reranker, self._client = embedder, reranker, client

    # Models load on first use, so importing this module stays fast and tests need no GPU.
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.s.embed_model, device=self.s.resolved_device())
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.s.rerank_model, device=self.s.resolved_device())
        return self._reranker

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.s.qdrant_url)
        return self._client

    def search(self, query: str) -> dict:
        """Returns {"found": bool, "passages": [{"text", "file", "lang", "score"}]}.

        Searches every language folder: BGE-M3 is cross-lingual, so a Hindi question can use an English
        passage. The answer step still replies in the user's language.
        """
        query = query.strip()
        if not query:
            return {"found": False, "passages": []}
        vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        hits = self.client.query_points(self.s.collection, query=vec, limit=self.s.retrieve_k).points
        if not hits:
            return {"found": False, "passages": []}

        scores = _to_probability([float(x) for x in self.reranker.predict([(query, h.payload["text"]) for h in hits])])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda x: -x[1])[: self.s.top_passages]
        passages = [
            {"text": h.payload["text"], "file": h.payload.get("file"), "lang": h.payload.get("lang"),
             "score": round(s, 4)}
            for h, s in ranked
            if s >= self.s.min_score
        ]
        return {"found": bool(passages), "passages": passages}


@lru_cache
def get_retriever() -> Retriever:
    return Retriever()


def search_airport_info(query: str) -> dict:
    return get_retriever().search(query)
