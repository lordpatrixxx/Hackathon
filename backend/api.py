import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from backend.config import ensure_paths, settings
from backend.ingest import ingest_documents
from backend.models.embeddings import EmbeddingModel
from backend.models.llm import LLMClient
from backend.rag.prompt import build_grounded_prompt
from backend.vectorstore import VectorStoreManager

ensure_paths()

app = FastAPI(
    title="Finance RAG Intelligence API",
    description="High-performance grounded RAG platform for large multi-gigabyte financial datasets.",
    version="1.0.0",
)

allowed_origins_raw = getattr(settings, "ALLOWED_ORIGINS", "*")
if allowed_origins_raw == "*" or not allowed_origins_raw:
    origins = ["*"]
else:
    origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStoreManager()
embedding_model = EmbeddingModel()
llm_client = LLMClient()


class ChatRequest(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None
    conversation_id: Optional[str] = None
    top_k: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def validate_message_or_query(cls, values: Any) -> Any:
        if isinstance(values, dict):
            msg = values.get("message") or values.get("query")
            if not msg or not str(msg).strip():
                raise ValueError("A non-empty 'message' or 'query' field is required.")
            values["message"] = str(msg).strip()
        return values


class SourceCitation(BaseModel):
    file: str
    page: Optional[int] = 1
    document_type: Optional[str] = "financial_document"
    company_name: Optional[str] = None
    symbol: Optional[str] = None
    excerpt: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    latency_seconds: float
    retrieved_count: int


class HealthResponse(BaseModel):
    status: str
    vector_store: bool
    index_ready: bool
    indexed_chunks: int
    embedding_model: bool
    llm: bool


@app.get("/health", response_model=HealthResponse)
def health():
    chunks_count = vector_store.count()
    vector_ok = chunks_count >= 0
    index_ready = chunks_count > 0
    has_llm = bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or settings.LLM_PROVIDER == "ollama"
    )

    return HealthResponse(
        status="ok" if (vector_ok and index_ready) else "degraded",
        vector_store=vector_ok,
        index_ready=index_ready,
        indexed_chunks=chunks_count,
        embedding_model=True,
        llm=has_llm,
    )


@app.get("/api/stats")
def stats():
    count = vector_store.count()
    return {
        "total_chunks": count,
        "collection_name": settings.COLLECTION_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query_text = (req.message or req.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Message or query cannot be empty")

    t_start = time.time()

    # Verify vector store is ready
    if vector_store.count() == 0:
        return ChatResponse(
            answer="No documents are currently indexed in the vector store. Please run offline ingestion via `python backend/ingest.py` before querying.",
            sources=[],
            latency_seconds=round(time.time() - t_start, 2),
            retrieved_count=0,
        )

    # 1. Embed query
    try:
        query_embedding = embedding_model.embed_query(query_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding query failed: {exc}")

    # 2. Hybrid search (dense vector similarity + exact entity/term boost)
    top_k = req.top_k or settings.RETRIEVAL_TOP_K
    try:
        hits = vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            query_text=query_text,
        )
    except Exception as exc:
        return ChatResponse(
            answer=f"Vector retrieval error: {exc}",
            sources=[],
            latency_seconds=round(time.time() - t_start, 2),
            retrieved_count=0,
        )

    if not hits:
        return ChatResponse(
            answer="The requested information was not found in the provided dataset.",
            sources=[],
            latency_seconds=round(time.time() - t_start, 2),
            retrieved_count=0,
        )

    # 3. Build grounded prompt and generate answer
    context_chunks = [hit["text"] for hit in hits]
    prompt = build_grounded_prompt(query_text, context_chunks)

    try:
        answer = llm_client.generate(prompt)
    except Exception as exc:
        answer = f"Language model error: {exc}"

    # 4. Extract structured source citations
    sources: List[SourceCitation] = []
    seen: set = set()
    for hit in hits:
        meta = hit.get("metadata", {})
        fname = meta.get("file_name", "unknown")
        doc_type = meta.get("document_type", "document")
        page = meta.get("page", 1)
        company = meta.get("company_name")
        symbol = meta.get("symbol")

        sig = f"{fname}:{page}:{company}:{symbol}"
        if sig not in seen:
            seen.add(sig)
            raw_text = hit.get("text", "")
            excerpt = raw_text[:220] + "..." if len(raw_text) > 220 else raw_text
            sources.append(
                SourceCitation(
                    file=fname,
                    page=page,
                    document_type=doc_type,
                    company_name=company,
                    symbol=symbol,
                    excerpt=excerpt,
                )
            )

    latency = round(time.time() - t_start, 2)
    return ChatResponse(
        answer=answer,
        sources=sources,
        latency_seconds=latency,
        retrieved_count=len(hits),
    )


@app.get("/")
def root():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "frontend", "index.html")
    if not os.path.exists(index_path):
        index_path = os.path.join(os.getcwd(), "frontend", "index.html")
    if not os.path.exists(index_path):
        index_path = os.path.join(os.getcwd(), "index.html")

    if os.path.exists(index_path):
        return FileResponse(index_path)

    return {
        "service": "Finance RAG Intelligence Backend",
        "status": "operational",
        "docs_url": "/docs",
        "health_url": "/health",
        "chat_url": "/api/chat",
    }
