from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import requests


@dataclass
class JinaV4Embedder:
    api_key: str
    model: str = "jina-embeddings-v4"
    endpoint: str = "https://api.jina.ai/v1/embeddings"

    def embed(self, texts: Iterable[str], batch_size: int = 64) -> List[List[float]]:
        items = [t for t in texts if t and t.strip()]
        if not items:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        vectors: List[List[float]] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            payload = {"model": self.model, "input": batch}
            res = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            if res.status_code >= 400:
                raise RuntimeError(f"Jina embedding API error: {res.status_code} {res.text}")
            data = res.json().get("data", [])
            ordered = sorted(data, key=lambda x: x.get("index", 0))
            vectors.extend([row["embedding"] for row in ordered])

        return vectors
