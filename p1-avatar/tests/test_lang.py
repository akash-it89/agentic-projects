from p1avatar.lang import detect_lang, normalize_for_tts
from p1avatar.rag.search import Retriever, _to_probability
from p1avatar.voice.asr import pick_language


def test_normalize_for_tts():
    assert normalize_for_tts("**Gate C22** is in `Terminal 2`.") == "Gate C 22 is in Terminal 2."


def test_detect_lang():
    assert detect_lang("गेट C22 टर्मिनल 2 में है।") == "hi"
    assert detect_lang("La puerta C22 está en la Terminal 2.") == "es"
    assert detect_lang("Gate C22 is in Terminal 2.") == "en"
    assert detect_lang("   ") is None


def test_pick_language_prefers_supported():
    assert pick_language("hi", [("hi", 0.8), ("ur", 0.15)]) == ("hi", 0.8)
    assert pick_language("cy", [("cy", 0.5), ("en", 0.4), ("es", 0.1)]) == ("en", 0.4)
    assert pick_language("spanish", None) == ("es", 1.0)
    assert pick_language("fr", []) == ("en", 0.0)


def test_logits_are_mapped_to_probabilities():
    assert _to_probability([0.2, 0.9]) == [0.2, 0.9]
    probs = _to_probability([-3.0, 0.0, 4.0])
    assert probs[1] == 0.5 and probs[0] < 0.1 and probs[2] > 0.9


class _Hit:
    def __init__(self, text):
        self.payload = {"text": text, "file": "en/a.md", "lang": "en"}


class _Client:
    def __init__(self, hits):
        self.hits = hits

    def query_points(self, collection, query, limit):
        return type("R", (), {"points": self.hits})()


class _Vector(list):
    def tolist(self):
        return list(self)


class _Embedder:
    def encode(self, text, normalize_embeddings=True):
        return _Vector([0.0, 1.0])


class _Reranker:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores


def _retriever(scores, min_score=0.35):
    from p1avatar.config import Settings
    s = Settings(min_score=min_score, top_passages=2, _env_file=None)
    hits = [_Hit(f"passage {i}") for i in range(len(scores))]
    return Retriever(s, embedder=_Embedder(), reranker=_Reranker(scores), client=_Client(hits))


def test_threshold_filters_and_ranks():
    r = _retriever([0.1, 0.8, 0.5]).search("gate?")
    assert r["found"] and [p["text"] for p in r["passages"]] == ["passage 1", "passage 2"]


def test_all_below_threshold_means_not_found():
    assert _retriever([0.1, 0.2]).search("pharmacy?") == {"found": False, "passages": []}
