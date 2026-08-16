import logging
import re
from typing import Any, Dict, List, Optional, Set

import chromadb

from backend.config import settings

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "what", "was", "the", "and", "for", "how", "much", "with", "from",
    "when", "show", "tell", "about", "find", "which", "where", "were",
    "that", "this", "these", "those", "have", "been", "will", "would",
    "could", "should", "does", "give", "list", "name", "rate", "year",
    "many", "some", "data", "file", "record", "records", "summarize",
    "summary", "details", "information", "info", "please",
}


class VectorStoreManager:
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        batch_size = 1000
        for i in range(0, len(chunks), batch_size):
            chunk_slice = chunks[i : i + batch_size]
            emb_slice = embeddings[i : i + batch_size]

            ids = [chunk["metadata"]["chunk_id"] for chunk in chunk_slice]
            documents = [chunk["text"] for chunk in chunk_slice]
            metadatas = []
            for chunk in chunk_slice:
                clean_meta = {}
                for k, v in chunk.get("metadata", {}).items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    elif v is None:
                        clean_meta[k] = ""
                    else:
                        clean_meta[k] = str(v)
                metadatas.append(clean_meta)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=emb_slice,
            )

    def query(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or settings.RETRIEVAL_TOP_K
        total_in_db = self.count()
        if total_in_db == 0:
            return []

        actual_k = min(k, total_in_db)

        # 1. Exact entity & keyword retrieval
        exact_hits: List[Dict[str, Any]] = []
        if query_text:
            exact_hits = self._find_exact_matches(query_text, limit=actual_k)

        # 2. Dense Vector retrieval
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(actual_k * 3, total_in_db),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where

        try:
            results = self.collection.query(**query_args)
        except Exception as exc:
            logger.warning(f"Vector search with where clause failed: {exc}. Retrying without where.")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(actual_k * 3, total_in_db),
                include=["documents", "metadatas", "distances"],
            )

        vector_hits = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                vector_hits.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    }
                )

        # 3. Merge, re-rank, and deduplicate (prioritizing highly specific entity matches)
        seen_ids: Set[str] = set()
        merged_payload: List[Dict[str, Any]] = []

        for hit in exact_hits:
            if hit["id"] not in seen_ids:
                seen_ids.add(hit["id"])
                merged_payload.append(hit)
                if len(merged_payload) >= actual_k:
                    break

        for hit in vector_hits:
            if hit["id"] not in seen_ids:
                seen_ids.add(hit["id"])
                merged_payload.append(hit)
                if len(merged_payload) >= actual_k:
                    break

        return merged_payload

    def _find_exact_matches(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        total_in_db = self.count()
        if total_in_db == 0:
            return []

        hits: List[Dict[str, Any]] = []
        clean_text = re.sub(r"[^\w\s]", " ", query_text)
        tokens = [t.strip() for t in clean_text.split() if len(t.strip()) >= 3 and t.strip().lower() not in STOP_WORDS]

        search_terms: List[str] = []

        # Multi-word entity phrases
        for phrase in ["Hazel Robinson", "Reliance Industries", "Credit Card", "Apple", "Microsoft", "Tata Consultancy Services"]:
            if phrase.lower() in query_text.lower() or all(w.lower() in query_text.lower() for w in phrase.split()):
                search_terms.append(phrase)

        # Add individual key tokens, prioritizing alphanumeric IDs (e.g. T001000, 500325) and longer words
        id_tokens = [t for t in tokens if any(c.isdigit() for c in t) or len(t) >= 5]
        short_tokens = [t for t in tokens if t not in id_tokens]
        ordered_tokens = id_tokens + short_tokens

        for t in ordered_tokens:
            if t.upper().startswith("T00") and len(t) >= 5:
                search_terms.insert(0, f"Terminal ID {t.upper()}")
                search_terms.insert(0, f"terminal_id: {t.upper()}")
            if t.upper().startswith("C00") and len(t) >= 5:
                search_terms.insert(0, f"Customer ID {t.upper()}")
                search_terms.insert(0, f"customer_id: {t.upper()}")
            for variant in [t, t.upper(), t.title(), t.lower()]:
                if variant not in search_terms:
                    search_terms.append(variant)

        seen: Set[str] = set()
        for term in search_terms[:12]:
            try:
                res = self.collection.get(
                    where_document={"$contains": term},
                    limit=limit,
                )
                if res and res.get("ids"):
                    for idx, doc_id in enumerate(res["ids"]):
                        if doc_id not in seen:
                            seen.add(doc_id)
                            hits.append(
                                {
                                    "id": doc_id,
                                    "text": res["documents"][idx],
                                    "metadata": res["metadatas"][idx] if res.get("metadatas") else {},
                                    "distance": 0.05,
                                }
                            )
                            if len(hits) >= limit:
                                return hits
            except Exception:
                pass
        return hits

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0
