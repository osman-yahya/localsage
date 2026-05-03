"""End-to-end ingest: WikiPage -> chunks -> embeddings -> Chroma upsert."""
from __future__ import annotations

from dataclasses import dataclass

from .chunker import Chunk, chunk_document
from .embedder import Embedder
from .vectorstore import VectorStore, make_id
from .wiki import WikiPage


@dataclass
class IngestReport:
    title: str
    url: str
    type: str
    chunks: int


def ingest_pages(pages: list[WikiPage], *,
                 entity_type: str,
                 store: VectorStore,
                 embedder: Embedder,
                 outer_size: int, outer_overlap: int,
                 inner_size: int, inner_overlap: int) -> list[IngestReport]:
    reports: list[IngestReport] = []
    for page in pages:
        if not page.text.strip():
            continue
        chunks: list[Chunk] = chunk_document(
            page.text,
            outer_size=outer_size, outer_overlap=outer_overlap,
            inner_size=inner_size, inner_overlap=inner_overlap,
        )
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        embs = embedder.encode(texts)
        ids = [make_id(page.url, c.layer, c.index) for c in chunks]
        metas = [
            {
                "title": page.title,
                "url": page.url,
                "type": entity_type,
                "lang": page.lang,
                "layer": c.layer,
                "chunk_index": c.index,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ]
        store.upsert(ids=ids, texts=texts, embeddings=embs.tolist(), metadatas=metas)
        reports.append(IngestReport(title=page.title, url=page.url,
                                    type=entity_type, chunks=len(chunks)))
    return reports
