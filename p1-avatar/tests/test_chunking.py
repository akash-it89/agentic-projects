from pathlib import Path

from p1avatar.rag.chunking import chunk_markdown
from p1avatar.rag.ingest import iter_records, lang_from_path

MD = """# Airport

Intro line.

## Terminals

### Terminal 2
Gate C22 is in Concourse C.

## Wi-Fi
Free Wi-Fi everywhere.
"""


def test_heading_path_is_kept():
    chunks = chunk_markdown(MD)
    by_heading = {c.heading: c for c in chunks}
    assert "Airport > Terminals > Terminal 2" in by_heading
    assert "Gate C22" in by_heading["Airport > Terminals > Terminal 2"].text
    assert by_heading["Airport > Wi-Fi"].body == "Free Wi-Fi everywhere."


def test_empty_sections_are_dropped():
    assert all(c.body for c in chunk_markdown(MD))
    assert not any(c.heading == "Airport > Terminals" for c in chunk_markdown(MD))


def test_long_sections_split_on_paragraphs():
    long_md = "## Big\n\n" + "\n\n".join("x" * 500 for _ in range(4))
    chunks = chunk_markdown(long_md, max_chars=1200)
    assert len(chunks) == 2
    assert all(len(c.body) <= 1200 for c in chunks)


def test_lang_from_folder():
    root = Path("data/airport")
    assert lang_from_path(root / "en" / "a.md", root) == "en"
    assert lang_from_path(root / "HI_terminals" / "x" / "b.md", root) == "hi"
    assert lang_from_path(root / "fr" / "c.md", root) == "unknown"


def test_sample_data_ingests_all_languages():
    data = Path(__file__).resolve().parents[1] / "data" / "airport"
    records = list(iter_records(data))
    assert {r["lang"] for r in records} == {"en", "hi", "es"}
    assert len({r["id"] for r in records}) == len(records)  # stable, unique IDs
