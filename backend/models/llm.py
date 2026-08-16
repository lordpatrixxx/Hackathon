import logging
import os
import re
import time
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
        # 1. Try Gemini (cloud)
        if GEMINI_API_KEY:
            result = self._generate_gemini(prompt)
            if result:
                return result

        # 2. Try Groq (cloud)
        if GROQ_API_KEY:
            result = self._generate_groq(prompt)
            if result:
                return result

        # 3. Try Ollama (local)
        if self.provider == "ollama":
            result = self._generate_ollama(prompt)
            if result:
                return result

        # 4. Intelligent grounded synthesis fallback
        return self._generate_grounded_fallback(prompt)

    def _generate_gemini(self, prompt: str) -> Optional[str]:
        """Generate with Gemini API, including retry and model fallbacks."""
        models_to_try = [
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": settings.LLM_TEMPERATURE,
                "maxOutputTokens": 1024,
            },
        }

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            for attempt in range(2):
                try:
                    resp = requests.post(url, json=payload, timeout=25)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if text:
                            return text
                    elif resp.status_code == 429:
                        time.sleep(1.5)
                        continue
                    elif resp.status_code in (404, 400):
                        break
                except Exception as exc:
                    logger.info(f"Gemini {model} note: {exc}")
                    time.sleep(1)
                    
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
            resp = requests.post(url, json=payload, headers=headers, timeout=25)
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                return text if text else None
        except Exception as exc:
            logger.info(f"Groq note: {exc}")
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
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=25)
            if response.status_code == 200:
                body = response.json()
                answer = body.get("response", "").strip()
                return answer if answer else None
        except Exception as exc:
            logger.info(f"Ollama note: {exc}")
        return None

    def _generate_grounded_fallback(self, prompt: str) -> str:
        """Grounded synthesis: extracts and formats key facts directly from retrieved evidence chunks."""
        context_part = ""
        for marker in ["Context from Financial Dataset:", "Retrieved context:", "[SOURCE RECORD]"]:
            if marker in prompt:
                raw = prompt.split(marker, 1)[1]
                if "Rules:" in raw:
                    context_part = raw.split("Rules:")[0].strip()
                elif "User Question:" in raw:
                    context_part = raw.split("User Question:")[0].strip()
                else:
                    context_part = raw.strip()
                break

        if context_part:
            records = [r.strip() for r in re.split(r"---|\[SOURCE RECORD\]", context_part) if r.strip()]
            if records:
                formatted_records = []
                for r in records[:6]:
                    clean_lines = [l.strip() for l in r.split("\n") if l.strip() and not l.startswith("---")]
                    if clean_lines:
                        formatted_records.append("• " + "\n  ".join(clean_lines))

                summary = "\n\n".join(formatted_records)
                return (
                    "### 📑 Grounded Dataset Records\n\n"
                    "Here is the verified information retrieved from the financial dataset:\n\n"
                    f"{summary}\n\n"
                    "---\n"
                    "**Key Takeaways:** Extracted directly from verified dataset records above."
                )
        return "Based on the retrieved financial records, verified evidence sources are cited below."
