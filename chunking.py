from __future__ import annotations

from pathlib import Path
from typing import List

import pdfplumber


def extract_text_from_file(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    raise ValueError("Unsupported file type. Use .txt or .pdf")


def extract_text_from_pdf(pdf_path: Path) -> str:
    pages: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= len(words):
            break
    return chunks
