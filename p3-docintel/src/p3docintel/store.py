"""One folder per document: runs/docs/<doc_id>/{source.<ext>, pages/page-N.png, result.json}.

A folder on disk is enough for a single-user app. For many users, swap this module for S3 + a database;
nothing else needs to change.
"""

import re
import uuid
from pathlib import Path

from .config import get_settings
from .schema import DocResult

_ID = re.compile(r"^[0-9a-f]{12}$")


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]


def doc_dir(doc_id: str) -> Path:
    if not _ID.match(doc_id):  # doc_id comes from URLs: never let it point outside runs/docs
        raise KeyError(doc_id)
    return get_settings().docs_dir / doc_id


def source_path(doc_id: str) -> Path:
    matches = sorted(doc_dir(doc_id).glob("source.*"))
    if not matches:
        raise KeyError(doc_id)
    return matches[0]


def pages_dir(doc_id: str) -> Path:
    return doc_dir(doc_id) / "pages"


def save_upload(doc_id: str, filename: str, data: bytes) -> Path:
    d = doc_dir(doc_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"source{Path(filename).suffix.lower()}"
    path.write_bytes(data)
    return path


def save(result: DocResult) -> None:
    path = doc_dir(result.doc_id) / "result.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(result.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(path)  # atomic, so the UI never reads a half-written file while polling


def load(doc_id: str) -> DocResult:
    path = doc_dir(doc_id) / "result.json"
    if not path.exists():
        raise KeyError(doc_id)
    return DocResult.model_validate_json(path.read_text(encoding="utf-8"))


def list_docs() -> list[DocResult]:
    root = get_settings().docs_dir
    if not root.exists():
        return []
    docs = []
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            docs.append(load(d.name))
        except (KeyError, ValueError):
            continue
    return docs
