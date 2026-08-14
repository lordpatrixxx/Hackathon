import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend.config import settings
from backend.models.embeddings import EmbeddingModel
from backend.models.llm import LLMClient
from backend.rag.prompt import build_grounded_prompt
from backend.vectorstore import VectorStoreManager


def run_query(question: str, top_k: int = None):
    question = question.strip()
    if not question:
        print("Please provide a non-empty question.")
        return

    vector_store = VectorStoreManager()
    total_docs = vector_store.count()
    if total_docs == 0:
        print("Vector database is empty! Please run `python backend/ingest.py` first to ingest data.")
        return

    embedding_model = EmbeddingModel()
    llm_client = LLMClient()

    print(f"\nSearching vector store ({total_docs} chunks indexed)...")
    t0 = time.time()
    query_emb = embedding_model.embed_query(question)
    hits = vector_store.query(query_emb, top_k=top_k or settings.RETRIEVAL_TOP_K, query_text=question)
    retrieval_time = time.time() - t0

    if not hits:
        print("No relevant chunks found for the query.")
        return

    print(f"Retrieved {len(hits)} relevant evidence chunks in {retrieval_time:.2f}s.\n")

    context_chunks = [hit["text"] for hit in hits]
    prompt = build_grounded_prompt(question, context_chunks)
    
    print("Generating grounded answer from local LLM...\n")
    t1 = time.time()
    answer = llm_client.generate(prompt)
    generation_time = time.time() - t1

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print("\n" + "-" * 60)
    print("SOURCES & CITATIONS:")
    print("-" * 60)
    for idx, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        fname = meta.get("file_name", "unknown")
        doc_type = meta.get("document_type", "document")
        page = meta.get("page", "n/a")
        comp = meta.get("company_name") or meta.get("symbol") or ""
        extra = f" | {comp}" if comp else ""
        print(f"[{idx}] {fname} (Page {page}, Type: {doc_type}{extra}) - Distance: {hit.get('distance', 0.0):.4f}")
    
    print(f"\n[Stats: Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s | Total: {retrieval_time + generation_time:.2f}s]\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        run_query(q)
    else:
        print("=== Finance RAG Interactive CLI Query ===")
        print("Type your financial questions below (or 'exit' to quit):")
        while True:
            try:
                user_q = input("\nQuery > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in {"exit", "quit", "q"}:
                    break
                run_query(user_q)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye.")
