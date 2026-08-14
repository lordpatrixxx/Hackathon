import pytest
from pathlib import Path

from backend.ingestion.pipeline import (
    discover_documents,
    chunk_documents,
    extract_document_text,
)
from backend.rag.prompt import build_grounded_prompt


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
                "metadata": {"document_type": "quarterly_results", "scrip_code": "500325"},
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
    # Test CSV statement analysis extraction
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


def test_vectorstore_exact_matching_logic():
    from backend.vectorstore import VectorStoreManager
    vsm = VectorStoreManager()
    matches = vsm._find_exact_matches("What was the performance for Reliance in Mar 2021?", limit=2)
    assert len(matches) > 0
    assert "RELIANCE" in matches[0]["text"].upper() or "500325" in matches[0]["text"]

