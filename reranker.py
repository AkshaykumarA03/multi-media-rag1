from __future__ import annotations

from typing import List

from .retriever import Chunk


class SimpleReranker:
    """Heuristic reranker: cosine score + token overlap bonus."""

    def rank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        q_tokens = set(query.lower().split())
        ranked: List[Chunk] = []
        for ch in chunks:
            c_tokens = set(ch.text.lower().split())
            overlap = len(q_tokens.intersection(c_tokens))
            bonus = min(0.25, overlap * 0.01)
            ranked.append(Chunk(text=ch.text, source=ch.source, modality=ch.modality, score=ch.score + bonus))
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked
