"""Two-layer overlapping chunker.

Layer 1 (outer): coarse, ~900 chars, broad context. Preferred for retrieval - more semantic
weight per chunk.
Layer 2 (inner): finer, ~350 chars, narrow facts. Better for short factual queries.

Both layers are stored side by side. Within a layer we slide a window with the configured
overlap so the boundary cases are covered. We try to cut on sentence-ish boundaries when one
is nearby; otherwise we hard-cut to keep the size predictable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    layer: str           # "outer" | "inner"
    index: int           # position within its layer
    char_start: int
    char_end: int


_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])")


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Approximate sentence split. Returns (start, end, sentence) triples."""
    pieces: list[tuple[int, int, str]] = []
    cursor = 0
    for m in _SENT_BOUNDARY.finditer(text):
        end = m.start()
        sent = text[cursor:end]
        if sent.strip():
            pieces.append((cursor, end, sent))
        cursor = m.end()
    if cursor < len(text):
        sent = text[cursor:]
        if sent.strip():
            pieces.append((cursor, len(text), sent))
    return pieces


def _slide(text: str, size: int, overlap: int, layer: str) -> list[Chunk]:
    if size <= 0:
        return []
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []

    if not sentences:
        return chunks

    i = 0
    pos = 0
    idx = 0
    while pos < len(text):
        end = min(pos + size, len(text))

        # Try to extend `end` backward to the nearest sentence end within the window so we cut
        # on a natural boundary instead of mid-word.
        snap = end
        for s_start, s_end, _ in sentences:
            if s_end <= end and s_end > pos + size // 2:
                snap = s_end
        end = snap

        snippet = text[pos:end].strip()
        if snippet:
            chunks.append(Chunk(text=snippet, layer=layer, index=idx,
                                char_start=pos, char_end=end))
            idx += 1

        if end >= len(text):
            break
        pos = max(pos + step, end - overlap)
        i += 1
        if i > 10000:  # paranoia
            break
    return chunks


def chunk_document(text: str, *,
                   outer_size: int, outer_overlap: int,
                   inner_size: int, inner_overlap: int) -> list[Chunk]:
    """Produce both layers concatenated. Caller embeds and stores them with metadata."""
    outer = _slide(text, outer_size, outer_overlap, "outer")
    inner = _slide(text, inner_size, inner_overlap, "inner")
    return outer + inner
