import hashlib
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

from backend.config import settings

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        if len(batches) == 1:
            return self._embed_batch_with_retry(batches[0])

        results: List[List[float]] = [None] * len(batches)
        max_workers = min(4, len(batches))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._embed_batch_with_retry, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = self._generate_fallback_embeddings(batches[idx])

        flat_results = []
        for batch_res in results:
            if batch_res:
                flat_results.extend(batch_res)
        return flat_results

    def embed_query(self, text: str) -> List[float]:
        results = self._embed_batch_with_retry([text])
        return results[0] if results else [0.0] * 768

    def _embed_batch_with_retry(self, texts: List[str], retries: int = 1) -> List[List[float]]:
        # 1. Try local Ollama if available
        if "ollama" in (settings.LLM_PROVIDER or "").lower():
            ollama_result = self._try_ollama(texts, retries)
            if ollama_result:
                return ollama_result

        # 2. Fast, deterministic sparse hash embedding (0ms latency, 0 rate limits, 100% reliable)
        return self._generate_fallback_embeddings(texts)

    def _try_ollama(self, texts: List[str], retries: int = 1) -> List[List[float]]:
        clean_name = self.model_name.replace("ollama/", "")
        payload = {"model": clean_name, "input": texts}
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    "http://localhost:11434/api/embed",
                    json=payload,
                    timeout=15,
                )
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "embeddings" in data:
                        return data["embeddings"]
                    if isinstance(data, list):
                        return [item.get("embedding", []) for item in data]
            except Exception:
                if attempt == retries:
                    return None
                time.sleep(0.5)
        return None

    def _generate_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Deterministic sparse hash embeddings. Always available, zero latency, no rate limits."""
        dim = 768
        results = []
        for text in texts:
            vec = [0.0] * dim
            words = text.lower().split()
            for i, w in enumerate(words):
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                pos_h = int(hashlib.sha256(f"{w}{i}".encode()).hexdigest(), 16)
                sign = 1.0 if pos_h % 2 == 0 else -1.0
                vec[idx] += sign * (1.0 / (i + 1))
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / norm for x in vec])
        return results
