"""Chroma persistent client wrapper.

We use Option B from the spec: one collection, with metadata `type` (person | place | unknown)
plus document title, url, layer, and chunk index. The retriever applies a `where` filter when
the query router is confident; otherwise it queries unfiltered.

Why Option B: a single index keeps memory and disk usage lower, lets us answer "compare
person and place" questions in a single shot, and metadata filtering in Chroma is essentially
free. The cost is that the router has to be reasonable - with two stores you can hard-route at
the index level. We accept that tradeoff because the keyword router below is sufficient for the
homework's entity set, and unknown queries simply fall back to mixed retrieval.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

import chromadb
from chromadb.config import Settings


@dataclass
class StoredChunk:
    text: str
    metadata: dict
    score: float


class VectorStore:
    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection_name = collection
        # Distance is cosine; embeddings are pre-normalized so the result equals 1 - similarity.
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, *, ids: list[str], texts: list[str],
               embeddings: list[list[float]], metadatas: list[dict]) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )

    def query(self, embedding: list[float], top_k: int,
              where: dict | None = None) -> list[StoredChunk]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )
        out: list[StoredChunk] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            sim = 1.0 - float(dist)  # cosine distance -> cosine similarity
            out.append(StoredChunk(text=doc, metadata=dict(meta or {}), score=sim))
        return out

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def known_documents(self) -> list[dict]:
        """Distinct (title, url, type) currently stored. Used by the CLI 'list' command."""
        # Chroma has no DISTINCT; pull a bounded sample of metadatas and dedup in Python.
        try:
            res = self.collection.get(limit=10000, include=["metadatas"])
        except Exception:
            return []
        seen: dict[str, dict] = {}
        for meta in res.get("metadatas", []) or []:
            url = (meta or {}).get("url")
            if not url or url in seen:
                continue
            seen[url] = {
                "title": meta.get("title"),
                "url": url,
                "type": meta.get("type", "unknown"),
            }
        return list(seen.values())


def make_id(url: str, layer: str, index: int) -> str:
    """Stable per-chunk id so re-ingesting the same page upserts instead of duplicating."""
    base = f"{url}::{layer}::{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))
