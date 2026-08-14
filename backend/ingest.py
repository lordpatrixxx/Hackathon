import argparse
import os
import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import ensure_paths, settings
from backend.ingestion.pipeline import chunk_documents, discover_documents, extract_document_text
from backend.models.embeddings import EmbeddingModel
from backend.vectorstore import VectorStoreManager


def ingest_documents(data_dir: str = None, force: bool = False, batch_size: int = 64):
    ensure_paths()
    data_dir = data_dir or settings.DATA_DIR
    print(f"=== Starting Finance RAG Ingestion on: {data_dir} ===")
    start_time = time.time()

    files = discover_documents(data_dir)
    if not files:
        print(f"No supported files found in {data_dir}.")
        return 0

    print(f"Discovered {len(files)} dataset targets (including top-level datasets and security directories).")

    embedding_model = EmbeddingModel()
    vector_store = VectorStoreManager()

    if force:
        print("Force flag set. Clearing existing vector collection...")
        try:
            vector_store.client.delete_collection(vector_store.collection_name)
            vector_store.collection = vector_store.client.get_or_create_collection(
                name=vector_store.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print("Vector collection cleared.")
        except Exception as exc:
            print(f"Note on clearing collection: {exc}")

    total_chunks_indexed = 0
    total_files_processed = 0

    for idx, file_path in enumerate(files, start=1):
        target_name = file_path.name
        print(f"[{idx}/{len(files)}] Extracting: {target_name}...", end=" ", flush=True)
        t0 = time.time()
        
        try:
            documents = extract_document_text(file_path)
            chunks = chunk_documents(documents)
        except Exception as exc:
            print(f"FAILED extraction: {exc}")
            continue

        if not chunks:
            print("0 chunks (skipped)")
            continue

        print(f"{len(chunks)} chunks extracted ({time.time() - t0:.2f}s). Embedding & indexing...", end=" ", flush=True)
        t_emb = time.time()

        # Batch embed and upsert
        all_texts = [chunk["text"] for chunk in chunks]
        try:
            embeddings = embedding_model.embed_documents(all_texts)
            vector_store.upsert_chunks(chunks, embeddings)
            total_chunks_indexed += len(chunks)
            total_files_processed += 1
            print(f"Done ({time.time() - t_emb:.2f}s). Total indexed so far: {total_chunks_indexed}")
        except Exception as exc:
            print(f"FAILED indexing: {exc}")

    total_duration = time.time() - start_time
    final_count = vector_store.count()
    print("\n" + "=" * 60)
    print(f"INGESTION COMPLETE in {total_duration:.2f} seconds!")
    print(f"Files/Targets Processed: {total_files_processed}")
    print(f"Total Chunks Ingested: {total_chunks_indexed}")
    print(f"ChromaDB Persistent Collection Count: {final_count}")
    print("=" * 60 + "\n")
    return total_chunks_indexed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest finance dataset into ChromaDB vector store")
    parser.add_argument("--data-dir", default=settings.DATA_DIR, help="Path to data directory")
    parser.add_argument("--force", action="store_true", help="Clear existing index before ingesting")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()
    ingest_documents(data_dir=args.data_dir, force=args.force, batch_size=args.batch_size)
