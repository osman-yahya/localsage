"""Embed query, route, retrieve, rerank lightly."""
from __future__ import annotations

from dataclasses import dataclass

from .embedder import Embedder
from .router import Router, Routed
from .vectorstore import StoredChunk, VectorStore


@dataclass
class Retrieved:
    chunks: list[StoredChunk]
    route: Routed


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, router: Router,
                 top_k: int = 5, oversample: int = 4, min_similarity: float = 0.25):
        self.store = store
        self.embedder = embedder
        self.router = router
        self.top_k = top_k
        self.oversample = oversample
        self.min_similarity = min_similarity

    def retrieve(self, query: str) -> Retrieved:
        route = self.router.classify(query)
        emb = self.embedder.encode_one(query).tolist()

        where = None
        if route.category in ("person", "place"):
            where = {"type": route.category}

        # Oversample then filter by score floor + light dedup on (url, char_start).
        n = self.top_k * max(1, self.oversample)
        raw = self.store.query(embedding=emb, top_k=n, where=where)

        if not raw and where is not None:
            # Router was wrong about type (or filter excluded everything). Retry unfiltered.
            raw = self.store.query(embedding=emb, top_k=n, where=None)

        deduped: list[StoredChunk] = []
        seen: set[tuple] = set()
        for c in raw:
            key = (c.metadata.get("url"), c.metadata.get("char_start"))
            if key in seen:
                continue
            seen.add(key)
            if c.score >= self.min_similarity:
                deduped.append(c)

        # Prefer outer (coarse) chunks when scores are close, since they carry more context.
        deduped.sort(key=lambda c: (-c.score, 0 if c.metadata.get("layer") == "outer" else 1))
        return Retrieved(chunks=deduped[: self.top_k], route=route)
