"""Synthetic newspaper pages with known answers (gold JSON), for testing before you have labelled scans.

Pages have columns, headlines, paragraphs, printed rules between articles, and articles that continue
into the next column, which is the hard case for the decider. They are cleaner than real 1950s scans,
so scores here are an upper bound. Use them to check the code works; judge quality on real pages.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORDS = (  # noqa: SIM905  a sentence is easier to edit than a 150-item list
    "the council met on tuesday to discuss a new plan for the harbor and the railway station while "
    "residents asked about rising prices of bread milk and coal during the long winter months the mayor "
    "said that the city would build a bridge across the river before the spring floods and that the "
    "school board had approved funds for two new classrooms farmers reported a strong wheat harvest in "
    "the northern district and the market opened early on saturday with fresh apples and potatoes the "
    "police department warned drivers about icy roads near the old mill a local orchestra will perform "
    "at the town hall next friday evening tickets are available at the post office the weather bureau "
    "expects light rain followed by clear skies the hospital welcomed a new doctor from the capital "
    "shipping companies announced additional routes to the southern ports"
).split()

HEADLINE_WORDS = ("Council Approves New Bridge Plan", "Harvest Breaks Record in North", "Rail Station To Expand",
                  "Icy Roads Close Mill Street", "Orchestra Returns To Town Hall", "Prices of Coal Rise Again",
                  "School Board Funds Classrooms", "New Doctor Joins Hospital", "Port Adds Southern Routes",
                  "Floods Expected This Spring", "Market Opens Early Saturday", "Mayor Speaks On Budget")


def _sentence(rng: random.Random) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
    return " ".join(words).capitalize() + "."


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float, indent: float = 0) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        limit = max_width - (indent if not lines else 0)
        if current and font.getlength(trial) > limit:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def generate_page(page_id: str, seed: int, width: int = 2400, height: int = 3300, columns: int = 4,
                  noisy: bool = False) -> tuple[Image.Image, dict]:
    rng = random.Random(seed)
    img = Image.new("L", (width, height), 245)
    draw = ImageDraw.Draw(img)
    body, head = ImageFont.load_default(size=26), ImageFont.load_default(size=44)
    margin, gutter = 90, 50
    col_w = (width - 2 * margin - (columns - 1) * gutter) / columns
    body_step, head_step, top, bottom = 36, 56, margin, height - margin

    for c in range(1, columns):  # vertical rules between columns
        x = margin + c * col_w + (c - 0.5) * gutter
        draw.line([(x, top), (x, bottom)], fill=40, width=3)

    articles: list[dict] = []
    gold_lines: list[dict] = []
    col, y = 0, top

    def col_x(c: int) -> float:
        return margin + c * (col_w + gutter)

    def put(text: str, font, x: float, aid: str) -> None:
        box = draw.textbbox((x, y), text, font=font)
        draw.text((x, y), text, font=font, fill=15)
        gold_lines.append({"box": [float(v) for v in box], "text": text, "article_id": aid})

    while col < columns:
        if y + head_step + 3 * body_step > bottom:  # no room for a headline and a few lines: next column
            col, y = col + 1, top
            continue
        aid = f"{page_id}-g{len(articles) + 1:03d}"
        articles.append({"article_id": aid})
        for text in _wrap(rng.choice(HEADLINE_WORDS), head, col_w):
            put(text, head, col_x(col), aid)
            y += head_step
        y += 8
        for _ in range(rng.randint(2, 5)):
            para = " ".join(_sentence(rng) for _ in range(rng.randint(2, 4)))
            for i, text in enumerate(_wrap(para, body, col_w, indent=30)):
                if y + body_step > bottom:  # article continues at the top of the next column
                    col, y = col + 1, top
                    if col >= columns:
                        break
                put(text, body, col_x(col) + (30 if i == 0 else 0), aid)
                y += body_step
            if col >= columns:
                break
            y += 14
        if col >= columns:
            break
        y += 20
        if y < bottom - 3 * body_step:  # rule under the article
            draw.line([(col_x(col), y), (col_x(col) + col_w, y)], fill=40, width=3)
        y += 24

    for a in articles:
        mine = [ln for ln in gold_lines if ln["article_id"] == a["article_id"]]
        a["bbox"] = [min(ln["box"][0] for ln in mine), min(ln["box"][1] for ln in mine),
                     max(ln["box"][2] for ln in mine), max(ln["box"][3] for ln in mine)]
        a["text"] = " ".join(ln["text"] for ln in mine)
    articles = [a for a in articles if "bbox" in a]

    if noisy:  # crude "old scan": blur, grain, uneven contrast
        img = img.filter(ImageFilter.GaussianBlur(0.8))
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(seed).normal(0, 12, arr.shape)
        arr *= np.linspace(0.85, 1.0, arr.shape[1])[None, :]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    gold = {"page_id": page_id, "width": width, "height": height, "language": "en",
            "articles": articles, "lines": gold_lines}
    return img, gold


def generate(out_dir: Path, n: int = 5, seed: int = 7, noisy: bool = False) -> list[Path]:
    pages, gold_dir = out_dir / "pages", out_dir / "gold"
    pages.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        page_id = f"synth-{i + 1:03d}"
        img, gold = generate_page(page_id, seed + i, noisy=noisy)
        img.save(pages / f"{page_id}.png")
        (gold_dir / f"{page_id}.json").write_text(json.dumps(gold, indent=1), encoding="utf-8")
        paths.append(pages / f"{page_id}.png")
    return paths
