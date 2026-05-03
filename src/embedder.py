"""Local sentence-transformers embedder. BERT-family encoder, runs on CPU.

Default is all-MiniLM-L6-v2 (~22M params, 384-dim, fast on CPU). The model is downloaded
on first use into HF_HOME (mapped to /data/hf_cache via the Dockerfile) so it persists
across container restarts.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model: SentenceTransformer | None = None

    def _ensure(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return self._ensure().get_sentence_embedding_dimension()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        model = self._ensure()
        vecs = model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
