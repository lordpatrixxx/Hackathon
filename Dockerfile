FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including OCR and document utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and seed dataset
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/seed/ ./data/seed/
COPY index.html .
COPY README.md .

ENV PORT=8000
ENV HOST=0.0.0.0
ENV VECTOR_DB_DIR=/app/chroma_db
ENV DATA_DIR=./data/seed
ENV COLLECTION_NAME=finance_rag

# One-time pre-build of persistent vector store inside container image
RUN python backend/ingest.py --data-dir ./data/seed --force

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
