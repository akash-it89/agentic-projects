import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    heading: str  # "Terminals > Terminal 2"
    body: str

    @property
    def text(self) -> str:
        # The heading path goes into the embedded text: "Gate C22 ..." alone loses the "Terminal 2" context.
        return f"{self.heading}\n{self.body}" if self.heading else self.body


def chunk_markdown(text: str, max_chars: int = 1200) -> list[Chunk]:
    """Split Markdown on headings, keeping the heading path. Long sections are split on blank lines."""
    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path), body.split("\n\n")))

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2)]
            buf = []
        else:
            buf.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, paragraphs in sections:
        current = ""
        for para in (p.strip() for p in paragraphs):
            if not para:
                continue
            if current and len(current) + len(para) + 2 > max_chars:
                chunks.append(Chunk(heading, current))
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(Chunk(heading, current))
    return chunks
