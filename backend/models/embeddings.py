import hashlib
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests

from backend.config import settings

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class EmbeddingModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = (model_name or settings.EMBEDDING_MODEL).lower()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # If using fast deterministic fallback or local, embed directly in high-speed batches
        if not (self.model_name.startswith("gemini") or self.model_name.startswith("text-embedding") or self.model_name.startswith("openai")):
            return self._embed_batch_with_retry(texts)

        batch_size = max(64, settings.EMBEDDING_BATCH_SIZE)
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
                except Exception as exc:
                    logger.debug(f"Batch embedding error: {exc}, using fallback")
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
        # 1. Cloud Gemini embeddings if explicitly configured
        if GEMINI_API_KEY and (self.model_name.startswith("gemini") or self.model_name == "text-embedding-004"):
            gemini_res = self._try_gemini_embed(texts, retries)
            if gemini_res:
                return gemini_res

        # 2. Cloud OpenAI embeddings if explicitly configured
        if OPENAI_API_KEY and (self.model_name.startswith("openai") or self.model_name == "text-embedding-3-small"):
            openai_res = self._try_openai_embed(texts, retries)
            if openai_res:
                return openai_res

        # 3. Local Ollama if configured
        if "ollama" in self.model_name:
            ollama_result = self._try_ollama(texts, retries)
            if ollama_result:
                return ollama_result

        # 4. Zero-Dependency Deterministic Sparse Hash Embedding
        # Guarantees 100% uptime with zero external API failure, 0 latency, 0 rate limits.
        return self._generate_fallback_embeddings(texts)

    def _try_gemini_embed(self, texts: List[str], retries: int = 1) -> Optional[List[List[float]]]:
        for attempt in range(retries + 1):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={GEMINI_API_KEY}"
                requests_payload = {
                    "requests": [
                        {
                            "model": "models/text-embedding-004",
                            "content": {"parts": [{"text": t[:2048]}]},
                        }
                        for t in texts
                    ]
                }
                resp = requests.post(url, json=requests_payload, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    if "embeddings" in data:
                        return [e.get("values", []) for e in data["embeddings"]]
            except Exception as exc:
                logger.debug(f"Gemini embed attempt {attempt} error: {exc}")
                if attempt < retries:
                    time.sleep(0.5)
        return None

    def _try_openai_embed(self, texts: List[str], retries: int = 1) -> Optional[List[List[float]]]:
        for attempt in range(retries + 1):
            try:
                url = "https://api.openai.com/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "text-embedding-3-small",
                    "input": [t[:2048] for t in texts],
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    if "data" in data:
                        return [item["embedding"] for item in data["data"]]
            except Exception as exc:
                logger.debug(f"OpenAI embed attempt {attempt} error: {exc}")
                if attempt < retries:
                    time.sleep(0.5)
        return None

    def _try_ollama(self, texts: List[str], retries: int = 1) -> Optional[List[List[float]]]:
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
        """High-speed deterministic sparse hash embeddings. Always available, zero latency, 100% reliable."""
        dim = 768
        results = []
        for text in texts:
            vec = [0.0] * dim
            words = text.lower().split()
            for i, w in enumerate(words[:80]):  # Fast bounded token hashing
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % dim
                sign = 1.0 if (h & 1) == 0 else -1.0
                vec[idx] += sign * (1.0 / (math.sqrt(i + 1)))
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / norm for x in vec])
        return results
