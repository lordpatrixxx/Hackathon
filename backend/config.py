import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name)
    if value is None:
        return default or ""
    return value


class Settings:
    DATA_DIR: str = get_setting("DATA_DIR", "./data")
    VECTOR_DB_DIR: str = get_setting("VECTOR_DB_DIR", "./chroma_db")
    COLLECTION_NAME: str = get_setting("COLLECTION_NAME", "finance_rag")
    EMBEDDING_MODEL: str = get_setting("EMBEDDING_MODEL", "nomic-embed-text")
    LLM_PROVIDER: str = get_setting("LLM_PROVIDER", "ollama")
    LLM_MODEL: str = get_setting("LLM_MODEL", "llama3.2")
    CHUNK_SIZE: int = int(get_setting("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(get_setting("CHUNK_OVERLAP", "150"))
    RETRIEVAL_TOP_K: int = int(get_setting("RETRIEVAL_TOP_K", "5"))
    EMBEDDING_BATCH_SIZE: int = int(get_setting("EMBEDDING_BATCH_SIZE", "64"))
    LLM_TEMPERATURE: float = float(get_setting("LLM_TEMPERATURE", "0"))
    RELEVANCE_THRESHOLD: float = float(get_setting("RELEVANCE_THRESHOLD", "0.2"))
    OCR_ENGINE: str = get_setting("OCR_ENGINE", "tesseract")
    HOST: str = get_setting("HOST", "0.0.0.0")
    PORT: int = int(get_setting("PORT", "8000"))


settings = Settings()


def ensure_paths() -> None:
    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)
