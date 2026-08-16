import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


def get_setting(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name)
    if value is None:
        return default or ""
    return value


def _detect_default_llm_provider() -> str:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def _detect_default_llm_model() -> str:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini-flash-latest"
    if os.getenv("GROQ_API_KEY"):
        return "llama3-8b-8192"
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    return "llama3.2"


def _detect_default_embedding_model() -> str:
    if os.getenv("GEMINI_API_KEY"):
        return "text-embedding-004"
    if os.getenv("OPENAI_API_KEY"):
        return "text-embedding-3-small"
    return "nomic-embed-text"


class Settings:
    DATA_DIR: str = get_setting("DATA_DIR", "/tmp/data" if IS_VERCEL else "./data")
    VECTOR_DB_DIR: str = get_setting("VECTOR_DB_DIR", "/tmp/chroma_db" if IS_VERCEL else "./chroma_db")
    COLLECTION_NAME: str = get_setting("COLLECTION_NAME", "finance_rag")
    EMBEDDING_MODEL: str = get_setting("EMBEDDING_MODEL", _detect_default_embedding_model())
    LLM_PROVIDER: str = get_setting("LLM_PROVIDER", _detect_default_llm_provider())
    LLM_MODEL: str = get_setting("LLM_MODEL", _detect_default_llm_model())
    CHUNK_SIZE: int = int(get_setting("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(get_setting("CHUNK_OVERLAP", "150"))
    RETRIEVAL_TOP_K: int = int(get_setting("RETRIEVAL_TOP_K", "5"))
    EMBEDDING_BATCH_SIZE: int = int(get_setting("EMBEDDING_BATCH_SIZE", "64"))
    LLM_TEMPERATURE: float = float(get_setting("LLM_TEMPERATURE", "0.0"))
    RELEVANCE_THRESHOLD: float = float(get_setting("RELEVANCE_THRESHOLD", "0.2"))
    OCR_ENGINE: str = get_setting("OCR_ENGINE", "tesseract")
    HOST: str = get_setting("HOST", "0.0.0.0")
    PORT: int = int(get_setting("PORT", "8000"))
    ALLOWED_ORIGINS: str = get_setting("ALLOWED_ORIGINS", "*")


settings = Settings()


def ensure_paths() -> None:
    try:
        Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        Path(settings.VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
