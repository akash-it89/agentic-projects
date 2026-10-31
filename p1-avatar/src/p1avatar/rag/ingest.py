"""Index the airport Markdown files into Qdrant.

    uv run python -m p1avatar.rag.ingest            # uses DATA_DIR from settings
    uv run python -m p1avatar.rag.ingest --data-dir path/to/airport
"""

import argparse
import uuid
from collections.abc import Iterator
from pathlib import Path

from qdrant_client import QdrantClient, models

from ..config import get_settings
from ..lang import SUPPORTED
from .chunking import chunk_markdown


def lang_from_path(md: Path, root: Path) -> str:
    """The first folder under the data root names the language: en/, hi/, es/ (or e.g. 'HI_terminals')."""
    first = md.relative_to(root).parts[0].lower()[:2]
    return first if first in SUPPORTED else "unknown"


def iter_records(data_dir: Path) -> Iterator[dict]:
    for md in sorted(data_dir.rglob("*.md")):
        rel = md.relative_to(data_dir).as_posix()
        lang = lang_from_path(md, data_dir)
        for i, chunk in enumerate(chunk_markdown(md.read_text(encoding="utf-8"))):
            yield {
                # Stable IDs make re-ingesting idempotent instead of duplicating points.
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel}#{i}")),
                "text": chunk.text,
                "heading": chunk.heading,
                "lang": lang,
                "file": rel,
            }


def ingest(data_dir: Path, recreate: bool = True) -> int:
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    records = list(iter_records(data_dir))
    if not records:
        raise SystemExit(f"No .md files found under {data_dir}")

    embedder = SentenceTransformer(s.embed_model, device=s.resolved_device())
    vectors = embedder.encode([r["text"] for r in records], normalize_embeddings=True, batch_size=16,
                              show_progress_bar=True)

    client = QdrantClient(url=s.qdrant_url)
    if recreate and client.collection_exists(s.collection):
        client.delete_collection(s.collection)
    if not client.collection_exists(s.collection):
        client.create_collection(
            s.collection,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
    client.upsert(
        s.collection,
        points=[
            models.PointStruct(id=r.pop("id"), vector=v.tolist(), payload=r)
            for r, v in zip(records, vectors, strict=True)
        ],
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=get_settings().data_dir)
    parser.add_argument("--keep", action="store_true", help="add to the existing collection instead of rebuilding")
    args = parser.parse_args()
    n = ingest(args.data_dir, recreate=not args.keep)
    print(f"Indexed {n} chunks from {args.data_dir}")


if __name__ == "__main__":
    main()
