"""The "decider": is block B the continuation of the same article as block A?

Three tiers, cheapest first, so the expensive one only sees the pairs the cheap ones can't settle:
  1. rules on layout features (free, settles most pairs)
  2. a small local LLM on the text either side of the break (only unclear pairs, mostly column jumps)
  3. a fallback heuristic if the LLM is off or unreachable

Every decision records its source, so the run reports how much went to the LLM. That share drives cost.
Decisions are also written to pairs.jsonl: label some by hand and you can train a tiny classifier to
replace tier 2 at the scale of millions of pages.
"""

import json
import logging
from collections import Counter

from .layout import block_text
from .schema import Block, PairDecision

log = logging.getLogger(__name__)


def rule_decision(f: dict) -> tuple[bool, float] | None:
    if f["rule_between"]:
        return False, 0.95
    if (f["same_column"] and f["a_is_headline"] and f["b_is_headline"] and f["gap_lines"] is not None
            and f["gap_lines"] <= 1.5):
        return True, 0.8  # stacked headline lines, or headline + deck: one article
    if f["b_is_headline"]:
        return False, 0.9
    if f["same_column"]:
        gap = f["gap_lines"]
        if gap is None or gap < -0.5:
            return None
        if f["a_is_headline"] and gap <= 2.0:
            return True, 0.9
        if gap <= 1.5:  # body to body (a headline b was ruled out above): a paragraph break
            return True, 0.85
        if gap > 4.0:
            return False, 0.7
    return None


def fallback_decision(f: dict) -> tuple[bool, float]:
    if f["column_jump"] and not f["b_is_headline"] and (
            f["a_ends_hyphen"] or (not f["a_ends_sentence"] and f["b_starts_lower"])):
        return True, 0.6
    return False, 0.5


DECIDER_PROMPT = """You are reading an old newspaper page. Decide whether text block B continues the same \
article as text block A (B comes right after A in reading order).

End of block A:
\"\"\"{a}\"\"\"

Start of block B:
\"\"\"{b}\"\"\"

Layout: {layout}

Answer with JSON only: {{"same_article": true or false, "confidence": a number from 0 to 1}}"""


class LLMDecider:
    def __init__(self, host: str, model: str):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model
        self.available = True

    def decide(self, a: Block, b: Block, f: dict) -> tuple[bool, float] | None:
        if not self.available:
            return None
        layout = ", ".join(f"{k}={v}" for k, v in f.items())
        prompt = DECIDER_PROMPT.format(a=block_text(a)[-400:], b=block_text(b)[:400], layout=layout)
        try:
            r = self.client.chat(model=self.model, format="json", options={"temperature": 0},
                                 messages=[{"role": "user", "content": prompt}])
            data = json.loads(r["message"]["content"])
            return bool(data["same_article"]), float(data.get("confidence", 0.7))
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Decider LLM gave an unusable answer (%s); using the fallback for this pair", e)
            return None
        except Exception as e:  # noqa: BLE001  connection refused, model not pulled, ...
            log.warning("Decider LLM unavailable (%s); using fallback rules for the rest of the run", e)
            self.available = False
            return None


class CascadeDecider:
    def __init__(self, llm: LLMDecider | None = None):
        self.llm = llm
        self.counts: Counter[str] = Counter()

    def decide(self, a: Block, b: Block, f: dict) -> PairDecision:
        for source, fn in (("rule", lambda: rule_decision(f)),
                           ("llm", lambda: self.llm.decide(a, b, f) if self.llm else None),
                           ("fallback", lambda: fallback_decision(f))):
            result = fn()
            if result is not None:
                self.counts[source] += 1
                return PairDecision(a.id, b.id, result[0], result[1], source, f)
        raise AssertionError("fallback_decision always returns a decision")
