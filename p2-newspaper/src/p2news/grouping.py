"""Pairwise "same article" decisions -> articles (union-find), plus each article's box and text."""

from .geometry import union
from .schema import Article, Block, PairDecision


def group(n: int, same_pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in same_pairs:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def dehyphenate(lines: list[str]) -> str:
    """Join OCR lines, rejoining words split across lines: 'infor-' + 'mation' -> 'information'."""
    out = ""
    for line in (ln.strip() for ln in lines):
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return out


def build_articles(ordered: list[Block], decisions: list[PairDecision], page_id: str,
                   engine: str = "paddleocr") -> list[Article]:
    by_id = {b.id: b for b in ordered}
    position = {b.id: i for i, b in enumerate(ordered)}
    groups = group(max(by_id) + 1 if by_id else 0, [(d.a, d.b) for d in decisions if d.same_article])
    articles = []
    groups = [sorted((i for i in g if i in by_id), key=position.__getitem__) for g in groups]
    groups.sort(key=lambda g: position[g[0]] if g else 0)  # number articles in reading order
    for g in (g for g in groups if g):
        blocks = [by_id[i] for i in g]
        lines = [ln for b in blocks for ln in b.lines]
        text = "\n\n".join(b.text or dehyphenate([ln.text for ln in b.lines]) for b in blocks)
        articles.append(Article(
            article_id=f"{page_id}-a{len(articles) + 1:03d}",
            bbox=union(b.box for b in blocks),
            text=text,
            confidence=round(sum(ln.score for ln in lines) / len(lines), 4),
            block_ids=g,
            regions=[b.box for b in blocks],
            source_engine=engine,
        ))
    return articles
