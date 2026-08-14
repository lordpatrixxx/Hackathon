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
from backend.vectorstore import VectorStoreManager


TEST_QUERIES = [
    "What was the Net Profit and Operating Profit Margin (OPM) for Reliance Industries in Mar 2023?",
    "What is the historical price range and latest close price for ticker AAPL or MSFT?",
    "Find details on credit card fraud transactions, merchants, and categories in the dataset.",
    "What is the user profile, yearly income, and credit score for Hazel Robinson?",
    "What are the peer comparison metrics or market quotes for Indian companies in the BSE dataset?",
]


def debug_query(question: str, top_k: int = 5):
    vector_store = VectorStoreManager()
    total = vector_store.count()
    if total == 0:
        print("Vector database is empty.")
        return

    model = EmbeddingModel()
    t0 = time.time()
    query_embedding = model.embed_query(question)
    hits = vector_store.query(query_embedding, top_k=top_k, query_text=question)
    duration = time.time() - t0

    print(f"\n{'='*70}")
    print(f"QUERY: {question}")
    print(f"Total Indexed Chunks in DB: {total} | Retrieval Time: {duration:.3f}s")
    print(f"{'='*70}")
    
    if not hits:
        print("No hits found.")
        return

    for idx, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        print(f"\n[Hit {idx}] Cosine Distance: {hit.get('distance', 0.0):.4f}")
        print(f"  Source: {meta.get('file_name', 'unknown')} (Page: {meta.get('page', 1)}, Type: {meta.get('document_type', 'doc')})")
        if meta.get("company_name"):
            print(f"  Company: {meta.get('company_name')} (Scrip: {meta.get('scrip_code', '')})")
        if meta.get("symbol"):
            print(f"  Security: {meta.get('symbol')}")
        content = hit.get("text", "")
        preview = content if len(content) <= 300 else content[:300] + "..."
        print(f"  Excerpt: {preview}")


def run_all_evaluations():
    print("=== Running Finance RAG System Evaluation ===")
    for q in TEST_QUERIES:
        debug_query(q, top_k=3)
    print("\n=== Evaluation Suite Completed ===")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_query(" ".join(sys.argv[1:]))
    else:
        run_all_evaluations()
