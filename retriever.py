from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import faiss
import numpy as np

from .chunking import chunk_text


Modality = Literal["text", "image"]


@dataclass
class Chunk:
    text: str
    source: str
    modality: Modality
    score: float = 0.0


class MultiModalRetriever:
    def __init__(self, embedder, chunk_size: int = 300, chunk_overlap: int = 60) -> None:
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.items: List[Chunk] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.emb_matrix: Optional[np.ndarray] = None

    @property
    def total_chunks(self) -> int:
        return len(self.items)

    def add_text(self, text: str, source: str) -> None:
        for ch in chunk_text(text, self.chunk_size, self.chunk_overlap):
            self.items.append(Chunk(text=ch, source=source, modality="text"))

    def add_image_caption(self, caption: str, source: str) -> None:
        for ch in chunk_text(caption, self.chunk_size, self.chunk_overlap):
            self.items.append(Chunk(text=ch, source=source, modality="image"))

    def build(self) -> None:
        if not self.items:
            raise ValueError("No chunks available to build index.")

        vecs = np.array(self.embedder.embed([i.text for i in self.items]), dtype="float32")
        if vecs.ndim != 2 or vecs.shape[0] == 0:
            raise ValueError("Invalid embeddings shape from provider.")

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)

        dim = int(vecs.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)
        self.emb_matrix = vecs

    def search(self, query: str, top_k: int = 4, modality_filter: str = "both") -> List[Chunk]:
        if self.index is None:
            raise RuntimeError("Index is not built.")

        q = np.array(self.embedder.embed([query]), dtype="float32")
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)

        scan_k = min(max(top_k * 4, top_k), len(self.items))
        scores, idxs = self.index.search(q, scan_k)

        out: List[Chunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[int(idx)]
            if modality_filter in ("text", "image") and item.modality != modality_filter:
                continue
            out.append(Chunk(text=item.text, source=item.source, modality=item.modality, score=float(score)))
            if len(out) >= top_k:
                break
        return out
