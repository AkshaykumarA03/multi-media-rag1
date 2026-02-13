import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    jina_api_key: str
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    llm_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "jina-embeddings-v4"

    @staticmethod
    def from_env() -> "AppConfig":
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        jina_api_key = os.getenv("JINA_API_KEY", "").strip()
        return AppConfig(groq_api_key=groq_api_key, jina_api_key=jina_api_key)
