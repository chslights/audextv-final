"""Tests for router failure handling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from audit_ingestion.router import ingest_one


def test_router_file_not_found():
    result = ingest_one("/nonexistent/file.pdf", api_key=None)
    assert result.status == "failed"
    assert result.evidence is not None
    assert any(f.type == "file_not_found" for f in result.evidence.flags)


def test_router_no_api_key(tmp_path):
    # Create a simple text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document with some content.")
    result = ingest_one(str(test_file), api_key=None)
    # Should succeed with extraction but no canonical analysis
    assert result.evidence is not None
    assert result.evidence.source_file == "test.txt"
    assert result.evidence.raw_text is not None
    assert any(f.type == "no_ai" for f in result.evidence.flags)


def test_router_csv_extraction(tmp_path):
    import pandas as pd
    test_file = tmp_path / "trial_balance.csv"
    df = pd.DataFrame({
        "Account": ["Cash", "AR", "AP"],
        "Debit": [10000, 5000, 0],
        "Credit": [0, 0, 3000],
    })
    df.to_csv(test_file, index=False)
    result = ingest_one(str(test_file), api_key=None)
    assert result.evidence is not None
    assert result.evidence.raw_text is not None
    assert len(result.evidence.tables) > 0


def test_router_returns_ingestion_result(tmp_path):
    test_file = tmp_path / "doc.txt"
    test_file.write_text("Invoice #001 from Vendor Corp for $500 on 2024-01-15")
    result = ingest_one(str(test_file), api_key=None)
    from audit_ingestion.models import IngestionResult
    assert isinstance(result, IngestionResult)
    assert result.status in ("success", "partial", "failed")
    assert result.engine_chain is not None
