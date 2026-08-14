import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStoreManager()
embedding_model = EmbeddingModel()
llm_client = LLMClient()


def check_and_ensure_index() -> bool:
    count = vector_store.count()
    if count > 0:
        return True
    try:
        ingested = ingest_documents()
        return ingested > 0
    except Exception as exc:
        print(f"Auto-ingestion error: {exc}")
        return False


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    top_k: Optional[int] = None


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
    indexed_chunks: int
    embedding_model: bool
    llm: bool


@app.get("/health", response_model=HealthResponse)
def health():
    chunks_count = vector_store.count()
    vector_ok = chunks_count >= 0
    
    try:
        embedding_model.embed_query("health")
        embed_ok = True
    except Exception:
        embed_ok = False

    try:
        llm_client.generate("test")
        llm_ok = True
    except Exception:
        llm_ok = False

    return HealthResponse(
        status="ok" if (vector_ok and embed_ok) else "degraded",
        vector_store=vector_ok,
        indexed_chunks=chunks_count,
        embedding_model=embed_ok,
        llm=llm_ok,
    )


@app.get("/api/stats")
def stats():
    count = vector_store.count()
    return {
        "total_chunks": count,
        "collection_name": settings.COLLECTION_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    t_start = time.time()
    
    # Ensure vector store is available
    if vector_store.count() == 0:
        if not check_and_ensure_index():
            return ChatResponse(
                answer="No documents are currently indexed. Please run ingestion via `python backend/ingest.py` first.",
                sources=[],
                latency_seconds=time.time() - t_start,
                retrieved_count=0,
            )

    # 1. Embed query
    try:
        query_embedding = embedding_model.embed_query(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding query failed: {exc}")

    # 2. Hybrid search (vector similarity + exact entity/term boost)
    top_k = req.top_k or settings.RETRIEVAL_TOP_K
    try:
        hits = vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            query_text=query,
        )
    except Exception as exc:
        return ChatResponse(
            answer=f"Vector retrieval error: {exc}",
            sources=[],
            latency_seconds=time.time() - t_start,
            retrieved_count=0,
        )

    if not hits:
        return ChatResponse(
            answer="No relevant evidence could be retrieved from the finance dataset for your question.",
            sources=[],
            latency_seconds=time.time() - t_start,
            retrieved_count=0,
        )

    # 3. Build grounded prompt and generate answer
    context_chunks = [hit["text"] for hit in hits]
    prompt = build_grounded_prompt(query, context_chunks)
    
    try:
        answer = llm_client.generate(prompt)
    except Exception as exc:
        answer = f"Language model error: {exc}"

    # 4. Extract structured source citations
    sources = []
    seen = set()
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
            excerpt = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
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


import os
from fastapi.responses import FileResponse

@app.get("/")
def root():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "frontend", "index.html")
    if not os.path.exists(index_path):
        index_path = os.path.join(os.getcwd(), "frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "service": "Finance RAG Intelligence Backend",
        "status": "operational",
        "docs_url": "/docs",
        "health_url": "/health",
        "chat_url": "/api/chat",
    }
