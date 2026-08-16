import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.ingestion.pipeline import (
    discover_documents,
    chunk_documents,
    extract_document_text,
)
from backend.models.embeddings import EmbeddingModel
from backend.models.llm import LLMClient
from backend.rag.prompt import build_grounded_prompt
from backend.vectorstore import VectorStoreManager


def test_discover_documents_finds_supported_files(tmp_path):
    text_file = tmp_path / "finance.txt"
    text_file.write_text("Alpha Bank revenue rose 12% in FY2024.", encoding="utf-8")
    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy")

    files = discover_documents(str(tmp_path))

    assert any(f.name == "finance.txt" for f in files)
    assert any(f.name == "report.pdf" for f in files)


def test_chunk_documents_preserves_text_and_metadata():
    text = "Company: Reliance Industries (ScripCode: 500325) | Net Sales Mar 2023: Rs 212,945 Cr | Net Profit: Rs 21,327 Cr"
    chunks = chunk_documents(
        [
            {
                "text": text,
                "source": "financial_stmt.csv",
                "page": 1,
                "metadata": {"document_type": "quarterly_results", "scrip_code": "500325", "company_name": "Reliance Industries"},
                "content_type": "table",
                "extraction_method": "finance_statement_extractor",
            }
        ],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) >= 1
    assert all("chunk_id" in chunk["metadata"] for chunk in chunks)
    assert all("file_name" in chunk["metadata"] for chunk in chunks)
    assert chunks[0]["metadata"]["scrip_code"] == "500325"


def test_grounded_prompt_stresses_evidence_first():
    prompt = build_grounded_prompt(
        "What was the Net Profit for Reliance?",
        ["Company: Reliance Industries | Net Profit: Rs 21,327 Cr in Mar 2023."],
    )

    assert "ONLY the retrieved context" in prompt
    assert "Reliance Industries" in prompt
    assert "Rs 21,327 Cr" in prompt


def test_custom_extractors(tmp_path):
    csv_file = tmp_path / "Statement_Analysis_Quarter.csv"
    csv_file.write_text(
        ",sentence,prefix,suffix,dir,header,ScripCode,dir_status\n"
        "0,Net Sales - QoQ Growth in quarter ended Mar 2023 is -1.94% vs -5.60% in Dec 2022,Net Sales,Growth,Down,Quarterly Growth,500325,Down\n",
        encoding="utf-8",
    )
    docs = extract_document_text(csv_file)
    assert len(docs) == 1
    assert "500325" in docs[0]["text"]
    assert "Net Sales" in docs[0]["text"]


def test_deterministic_fallback_embeddings():
    emb = EmbeddingModel()
    vecs = emb._generate_fallback_embeddings(["Reliance Industries quarterly revenue Mar 2023", "Apple AAPL stock price"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 768
    # Norm should be approximately 1.0
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_grounded_fallback_llm_synthesis():
    llm = LLMClient()
    prompt = build_grounded_prompt(
        "What is the Net Sales for Reliance in Mar 2023?",
        ["Company: Reliance Industries (ScripCode: 500325) | Net Sales Mar 2023: Rs 212,945 Cr"],
    )
    res = llm._generate_grounded_fallback(prompt)
    assert "Reliance Industries" in res or "500325" in res or "212,945" in res
    assert "Verified Grounded Evidence" in res or "Takeaways" in res


def test_vectorstore_upsert_and_hybrid_query(tmp_path):
    test_collection = f"test_col_{os.getpid()}"
    vsm = VectorStoreManager(collection_name=test_collection)
    emb_model = EmbeddingModel()

    chunk_data = [
        {
            "text": "Company: Reliance Industries (ScripCode: 500325) | Net Profit Mar 2023: Rs 21,327 Cr",
            "source": "Hist_BS_Fin_Stmt.csv",
            "page": 1,
            "metadata": {
                "chunk_id": "test_chunk_rel_01",
                "file_name": "Hist_BS_Fin_Stmt.csv",
                "scrip_code": "500325",
                "company_name": "Reliance Industries",
            },
        },
        {
            "text": "User Profile: Hazel Robinson | Age: 42 | Yearly Income: $125,000 | FICO Score: 780",
            "source": "sd254_users.csv",
            "page": 1,
            "metadata": {
                "chunk_id": "test_chunk_hazel_01",
                "file_name": "sd254_users.csv",
                "user_name": "Hazel Robinson",
            },
        }
    ]

    embeddings = emb_model.embed_documents([c["text"] for c in chunk_data])
    vsm.upsert_chunks(chunk_data, embeddings)

    assert vsm.count() >= 2

    # Test exact entity matching
    exact_matches = vsm._find_exact_matches("What was the performance for Reliance in Mar 2023?", limit=2)
    assert len(exact_matches) > 0
    assert "Reliance" in exact_matches[0]["text"]

    # Test hybrid query
    q_emb = emb_model.embed_query("Hazel Robinson credit score")
    hits = vsm.query(query_embedding=q_emb, top_k=2, query_text="Hazel Robinson")
    assert len(hits) > 0
    assert "Hazel Robinson" in hits[0]["text"]


def test_api_endpoints():
    client = TestClient(app)
    
    # Test health endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert "status" in data
    assert "vector_store" in data
    assert "index_ready" in data
    assert "indexed_chunks" in data

    # Test stats endpoint
    stats_resp = client.get("/api/stats")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "total_chunks" in stats_data
    assert "llm_model" in stats_data

    # Test root endpoint returns HTML
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "Finance RAG Intelligence" in root_resp.text
