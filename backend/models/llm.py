import logging
from typing import Optional

import requests

from backend.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model = model or settings.LLM_MODEL

    def generate(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._generate_ollama(prompt)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _generate_ollama(self, prompt: str) -> str:
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
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=90)
            response.raise_for_status()
            body = response.json()
            answer = body.get("response", "").strip()
            if answer:
                return answer
        except Exception as exc:
            logger.warning(f"Ollama generation failed: {exc}")

        # Intelligent grounded fallback if Ollama is unavailable
        return self._generate_grounded_fallback(prompt)

    def _generate_grounded_fallback(self, prompt: str) -> str:
        # Extract the context from prompt
        marker = "Retrieved context:"
        if marker in prompt:
            context_part = prompt.split(marker)[-1].strip()
            # Clean up instructions
            if context_part:
                lines = [l.strip() for l in context_part.split("\n") if l.strip() and not l.startswith("---") and not l.startswith("[SOURCE CONTEXT]")]
                summary = "\n• ".join(lines[:8])
                return f"Based on the dataset records:\n• {summary}"
        return "I could not generate an answer because the local LLM is temporarily unreachable."
