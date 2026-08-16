# 🚀 Enterprise Full-Stack Finance RAG Pipeline

A production-ready, resilient, and fully grounded Retrieval-Augmented Generation (RAG) platform built for large, heterogeneous financial and enterprise datasets (multi-gigabyte tabular CSVs, JSON directories, PDFs, Excel sheets, and transaction logs).

---

## 🌟 Key Capabilities & Features

1. **Universal Multi-Format Ingestion**:
   - Tabular semantic chunking for multi-period financial statements (Balance Sheets, Cash Flows, Quarterly Results, YoY/QoQ Growth) preserving scrip codes, company names, and metric relationships.
   - Profile & Transaction extractors for demographics, FICO credit scores, user card limits, terminal MCC codes, and fraud logs.
   - Document extractors for PDFs, DOCX, XLSX, TXT, and OCR fallback for scanned pages.
   - Lightweight self-contained seed bundle (`data/seed/`) for 1-click cloud boot.

2. **Hybrid Search (Dense Cosine Vectors + Exact Entity Booster)**:
   - Dense vector retrieval (`top_k * 3`) with Cosine distance indexing in ChromaDB.
   - High-specificity exact entity matching for alphanumeric IDs (e.g. `T001000`, `C001000`), scrip codes (`500325`), stock tickers (`AAPL`), and multi-word names (`Hazel Robinson`).
   - Deduplication and priority re-ranking for verifiable evidence retrieval.

3. **Resilient Multi-Provider LLM Orchestration & Zero-Downtime Fallback**:
   - Multi-cloud orchestration supporting **Google Gemini** (`gemini-flash-latest`, `gemini-2.5-flash`), **Groq** (`llama3-8b-8192`), **OpenAI** (`gpt-4o-mini`), and local **Ollama** (`llama3.2`).
   - **Zero-Error Grounded Fallback Synthesizer**: If cloud APIs hit rate limits (HTTP 429/503) or offline states, the system parses retrieved context chunks and formats clean structured facts, bullet points, and citations with zero error screens.

4. **Zero-Hallucination Prompt Grounding Contract**:
   - Strictly enforces responses based only on retrieved evidence.
   - Quotes exact figures, currency metrics, dates, and identifiers.
   - Explicitly returns *"The requested information was not found in the provided dataset."* when queried on unrecorded entities.

5. **Single-Port Modern Glassmorphic Web Application**:
   - Single-port architecture: FastAPI serves the frontend at `/` and REST API at `/api/*`.
   - Modern Glassmorphic Dark UI with dynamic responsive typography.
   - Marked.js for Markdown and Table rendering.
   - Interactive Expandable Source Citation Drawer with exact document excerpts.
   - Real-time latency and chunk monitoring badges.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               Frontend: Glassmorphic Web UI                 │
│   (Marked.js Markdown, Citation Accordion, Latency Badges)  │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / WebSocket (Port :8000)
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Router                    │
│   GET / (UI), GET /health, GET /api/stats, POST /api/chat    │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│   Hybrid Vector Store        │ │   Resilient LLM Engine       │
│   • Dense Cosine Search      │ │   • Primary: Gemini 2.5/Flash│
│   • Exact Entity/ID Booster  │ │   • Secondary: Groq/OpenAI   │
│   • ChromaDB Persistent      │ │   • Local: Ollama            │
│   • Auto-Boot Seed Indexing  │ │   • Fallback: Grounded Synth │
└──────────────▲───────────────┘ └──────────────────────────────┘
               │
┌──────────────┴───────────────┐
│ Ingestion & Chunking Engine  │
│ • Tabular Statements / CSVs  │
│ • Profiles / Fraud / News    │
│ • PDF, DOCX, XLSX, TXT       │
│ • Seed Dataset (data/seed/)  │
└──────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/lordpatrixxx/Hackathon.git
cd Hackathon
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file (or copy `.env.example`):
```env
GEMINI_API_KEY=your_gemini_api_key_here
# GROQ_API_KEY=your_groq_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here

DATA_DIR=./data/seed
VECTOR_DB_DIR=./chroma_db
COLLECTION_NAME=finance_rag
EMBEDDING_MODEL=nomic-embed-text
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-latest
```

### 3. Ingest Dataset
```bash
# Ingest lightweight seed bundle (takes ~20 seconds)
python backend/ingest.py --data-dir ./data/seed

# Ingest full multi-GB dataset
python backend/ingest.py --data-dir ./data
```

### 4. Run Web Application
```bash
python run_app.py
# Or directly with Uvicorn:
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

## 🧪 Testing & Evaluation

### Run Test Suite
```bash
python -m pytest tests/ -v
```

### Interactive CLI Query Tool
```bash
python query.py "What was the Net Profit for Reliance Industries in Mar 2023?"
python query.py "What is the user profile, yearly income, and debt for Hazel Robinson?"
python query.py "What is the capital of Mars?"
```

### Run Evaluation Suite
```bash
python evaluate.py
```

---

## 🚢 Cloud & Container Deployment

- **Docker**:
  ```bash
  docker build -t finance-rag .
  docker run -p 8000:7860 finance-rag
  ```
- **Render**: Connect repository; `render.yaml` automatically configures the web service.
- **Vercel**: Deploy directly; `vercel.json` and `api/index.py` handle serverless routing with `/tmp/chroma_db` auto-fallback.
