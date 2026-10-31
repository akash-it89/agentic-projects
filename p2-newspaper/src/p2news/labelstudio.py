"""Convert a Label Studio JSON export into gold files for evaluate.py.

Label each article with one rectangle (label "Article"). Optionally attach a per-region text area with the
article's text to also measure CER. Export with Export -> JSON, then:

    uv run python -m p2news labelstudio export.json data/real/gold
"""

import json
from pathlib import Path
from urllib.parse import unquote


def _page_id(image_ref: str) -> str:
    name = unquote(image_ref).replace("\\", "/").rsplit("/", 1)[-1]
    name = name.split("-", 1)[1] if name[:8].isalnum() and "-" in name[:9] else name  # strip upload prefix "1a2b3c4d-"
    return Path(name).stem


def convert(export_path: Path, out_dir: Path, label: str = "Article") -> list[Path]:
    tasks = json.loads(Path(export_path).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task in tasks:
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        results = annotations[0]["result"]
        texts = {r["id"]: " ".join(r["value"]["text"]) for r in results if r.get("type") == "textarea"}
        page_id = _page_id(next(iter(task["data"].values())))
        articles, width, height = [], None, None
        for r in results:
            if r.get("type") != "rectanglelabels" or label not in r["value"].get("rectanglelabels", []):
                continue
            width, height = r["original_width"], r["original_height"]
            v = r["value"]  # Label Studio stores percentages of the image size
            x0, y0 = v["x"] / 100 * width, v["y"] / 100 * height
            articles.append({"article_id": f"{page_id}-g{len(articles) + 1:03d}",
                             "bbox": [x0, y0, x0 + v["width"] / 100 * width, y0 + v["height"] / 100 * height],
                             "text": texts.get(r["id"], "")})
        gold = {"page_id": page_id, "width": width, "height": height, "articles": articles, "lines": []}
        path = out_dir / f"{page_id}.json"
        path.write_text(json.dumps(gold, indent=1, ensure_ascii=False), encoding="utf-8")
        written.append(path)
    return written
