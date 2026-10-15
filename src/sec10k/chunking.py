"""Section-aware chunking with a context header on every chunk.

Each chunk is prefixed with ``[TICKER FORM FY | Item N. Title]`` so that a chunk
read on its own (by BM25, an embedder, or an LLM) still says where it came from -
a zero-cost take on contextual retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Filing


@dataclass
class Chunk:
    item: str
    index: int
    header: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.header}\n{self.body}"


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Greedy paragraph packing; paragraphs longer than max_chars are split on sentences."""
    pieces: list[str] = []
    for para in _paragraphs(text):
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.!?;])\s+", para):
            while len(sent) > max_chars:  # pathological: a table row or run-on text
                pieces.append(sent[:max_chars])
                sent = sent[max_chars:]
            if len(buf) + len(sent) + 1 > max_chars and buf:
                pieces.append(buf)
                buf = ""
            buf = f"{buf} {sent}".strip()
        if buf:
            pieces.append(buf)

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        # Flush only reasonably sized buffers so a one-line intro doesn't become its own chunk.
        if buf and len(buf) + len(piece) + 1 > max_chars and len(buf) >= max_chars // 3:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            # Start the overlap at a word boundary.
            buf = (tail.split(" ", 1)[1] if " " in tail else "") + "\n" + piece if tail else piece
            buf = buf.strip()
        else:
            buf = f"{buf}\n{piece}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def chunk_filing(filing: Filing, max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    ref = filing.ref
    fy = f"FY{ref.fiscal_year}" if ref.fiscal_year else ref.filed_at
    out: list[Chunk] = []
    for key, section in filing.sections.items():
        header = f"[{ref.ticker} {ref.form} {fy} | Item {key}. {section.title}]"
        for i, body in enumerate(chunk_text(section.text, max_chars, overlap)):
            out.append(Chunk(item=key, index=i, header=header, body=body))
    return out
