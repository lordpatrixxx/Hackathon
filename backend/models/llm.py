import logging
import os
from typing import Optional

import requests

from backend.config import settings

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model = model or settings.LLM_MODEL

    def generate(self, prompt: str) -> str:
        # 1. Try Gemini (free, cloud)
        if GEMINI_API_KEY:
            result = self._generate_gemini(prompt)
            if result:
                return result

        # 2. Try Groq (free, cloud)
        if GROQ_API_KEY:
            result = self._generate_groq(prompt)
            if result:
                return result

        # 3. Try Ollama (local)
        if self.provider == "ollama":
            result = self._generate_ollama(prompt)
            if result:
                return result

        # 4. Smart grounded fallback
        return self._generate_grounded_fallback(prompt)

    def _generate_gemini(self, prompt: str) -> Optional[str]:
        """Use Google Gemini 1.5 Flash — free tier, no credit card needed."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "maxOutputTokens": 1024,
                },
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text if text else None
        except Exception as exc:
            logger.warning(f"Gemini generation failed: {exc}")
            return None

    def _generate_groq(self, prompt: str) -> Optional[str]:
        """Use Groq llama3 — free tier, very fast."""
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": settings.LLM_TEMPERATURE,
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return text if text else None
        except Exception as exc:
            logger.warning(f"Groq generation failed: {exc}")
            return None

    def _generate_ollama(self, prompt: str) -> Optional[str]:
        """Use local Ollama if available."""
        payload = {
            "model": self.model.replace("ollama/", ""),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": 1024,
            },
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
            response.raise_for_status()
            body = response.json()
            answer = body.get("response", "").strip()
            return answer if answer else None
        except Exception as exc:
            logger.warning(f"Ollama generation failed: {exc}")
            return None

    def _generate_grounded_fallback(self, prompt: str) -> str:
        """Grounded fallback: summarize retrieved context directly when no LLM is available."""
        marker = "Retrieved context:"
        if marker in prompt:
            context_part = prompt.split(marker)[-1].strip()
            if context_part:
                lines = [
                    l.strip()
                    for l in context_part.split("\n")
                    if l.strip()
                    and not l.startswith("---")
                    and not l.startswith("[SOURCE CONTEXT]")
                    and not l.startswith("Question:")
                    and not l.startswith("Instructions:")
                ]
                if lines:
                    summary = "\n• ".join(lines[:10])
                    return f"Based on the indexed finance dataset:\n• {summary}"
        return (
            "I retrieved relevant documents from the dataset, but no language model is currently "
            "configured to generate a response. Please set the GEMINI_API_KEY environment variable "
            "for free AI-powered answers."
        )
