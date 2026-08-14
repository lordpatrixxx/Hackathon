from typing import List


def build_grounded_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(f"[SOURCE RECORD]\n{chunk}" for chunk in context_chunks)
    return f"""You are a professional Finance RAG Assistant.

Analyze the user's question and provide an accurate, fact-based response using ONLY the retrieved context from the provided dataset below.

Context from Financial Dataset:
{context_text}

Rules:
1. Ground your answer strictly on the provided context facts, figures, tables, and records.
2. Quote exact numbers, dates, company names, ticker symbols, financial metrics, and transaction details as they appear.
3. If the context does not contain enough evidence to answer the question, clearly state: "The requested information was not found in the provided dataset."
4. Structure your response with clean Markdown headers, bullet points, or tables for readability.
5. End your response with a brief summary of key takeaways.

User Question:
{question}
"""
