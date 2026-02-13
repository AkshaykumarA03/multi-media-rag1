from __future__ import annotations

from typing import Dict, List

from groq import Groq

from .retriever import Chunk


class GroqAnswerGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY for answer generation.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def answer(self, query: str, contexts: List[Chunk], history: List[Dict[str, str]], max_history: int = 4) -> str:
        if not contexts:
            return "I do not know based on the provided context."

        context_block = "\n\n".join(
            [f"[{i+1}] ({c.modality}|{c.source}|score={c.score:.4f}) {c.text}" for i, c in enumerate(contexts)]
        )

        convo = []
        for msg in history[-max_history * 2 :]:
            if msg.get("role") in {"user", "assistant"} and msg.get("content"):
                convo.append({"role": msg["role"], "content": msg["content"]})

        system = (
            "You are a grounded enterprise assistant. "
            "Only answer from retrieved context. "
            "If context is insufficient, reply exactly: 'I do not know based on the provided context.' "
            "Cite sources inline using [n] where possible."
        )

        user_prompt = (
            f"Conversation:\n{convo}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            f"Question: {query}\n"
            "Answer with concise, factual statements grounded in the context."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content.strip()
