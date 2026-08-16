FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and seed data
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/seed/ ./data/seed/
COPY index.html .
COPY README.md .

ENV PORT=7860
ENV HOST=0.0.0.0
ENV VECTOR_DB_DIR=/tmp/chroma_db
ENV DATA_DIR=./data/seed

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
