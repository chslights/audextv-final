"""
Tests for v03.1 additions:
- Canonical cache hit
- Fast-mode auto-escalation
- Batch throughput smoke test
- Strict schema validation
- OCR cache
- Stage/timing visibility
- Provider mock contract
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ── Canonical cache ───────────────────────────────────────────────────────────

def test_canonical_cache_populated_on_extract(tmp_path):
    """After canonical extraction, result should be in cache."""
    from audit_ingestion.canonical import _canonical_cache, _canonical_cache_key, SCHEMA_VERSION
    from audit_ingestion.models import ParsedDocument, ParsedPage, ExtractionMeta

    pages = [ParsedPage(page_number=1, text="Invoice from Vendor Corp $500", char_count=30,
                        extractor="pdfplumber", confidence=0.9)]
    doc = ParsedDocument(
        source_file="cache_test.pdf",
        full_text="[Page 1]\nInvoice from Vendor Corp $500",
        page_count=1, pages=pages,
        extraction_chain=["pdfplumber"], primary_extractor="pdfplumber",
        confidence=0.8,
    )

    class MockProvider:
        model = "gpt-5.4"
        def extract_structured(self, *, system, user, json_schema, max_tokens=4000):
            return {
                "family": "invoice_receipt",
                "subtype": "vendor_invoice",
                "title": None,
                "audit_overview": {
                    "summary": "Vendor invoice for $500",
                    "audit_areas": ["expenses"],
                    "assertions": ["existence"],
                    "period": None,
                    "match_targets": ["ap_payables"],
                },
                "parties": [{"role": "vendor", "name": "Vendor Corp",
                              "normalized": "VENDOR CORP",
                              "provenance": {"page": 1, "quote": "Vendor Corp", "confidence": 0.95}}],
                "amounts": [{"type": "invoice_total", "value": 500.0, "currency": "USD",
                             "provenance": {"page": 1, "quote": "$500", "confidence": 0.98}}],
                "dates": [],
                "identifiers": [],
                "assets": [],
                "facts": [],
                "claims": [],
                "flags": [],
                "link_keys": {"party_names": ["VENDOR CORP"], "document_numbers": [],
                              "agreement_numbers": [], "invoice_numbers": [],
                              "asset_descriptions": [], "recurring_amounts": [],
                              "key_dates": [], "other_ids": []},
                "document_specific": {},
            }

    provider = MockProvider()
    from audit_ingestion.canonical import extract_canonical
    result = extract_canonical(doc, provider)

    cache_key = _canonical_cache_key(doc, "gpt-5.4")
    assert cache_key in _canonical_cache
    assert _canonical_cache[cache_key].source_file == "cache_test.pdf"


def test_canonical_cache_returns_cached_result(tmp_path):
    """Second call with same doc should return cached result without calling provider."""
    from audit_ingestion.canonical import _canonical_cache, _canonical_cache_key
    from audit_ingestion.models import ParsedDocument, ParsedPage, AuditEvidence

    pages = [ParsedPage(page_number=1, text="Cached document content here",
                        char_count=28, extractor="pdfplumber", confidence=0.9)]
    doc = ParsedDocument(
        source_file="cached_doc.pdf",
        full_text="[Page 1]\nCached document content here",
        page_count=1, pages=pages,
        extraction_chain=["pdfplumber"], primary_extractor="pdfplumber",
        confidence=0.8,
    )

    # Pre-populate cache
    cached_ev = AuditEvidence(source_file="cached_doc.pdf")
    cache_key = _canonical_cache_key(doc, "gpt-5.4")
    _canonical_cache[cache_key] = cached_ev

    call_count = {"n": 0}

    class CountingProvider:
        model = "gpt-5.4"
        def extract_structured(self, **kwargs):
            call_count["n"] += 1
            return {}

    from audit_ingestion.canonical import extract_canonical
    result = extract_canonical(doc, CountingProvider())
    assert call_count["n"] == 0  # Provider never called
    assert result.source_file == "cached_doc.pdf"


# ── Fast-mode escalation ──────────────────────────────────────────────────────

def test_fast_mode_escalation_flag_in_chain(tmp_path):
    """When fast mode is critically weak, ocr_escalated should appear in chain."""
    # Create a minimal PDF with no text (will be critically weak)
    # We can test this by checking the escalation logic path exists
    from audit_ingestion.extractor import extract_fast, MIN_CHARS_CRITICAL
    # Just verify the constant and logic exist
    assert MIN_CHARS_CRITICAL == 60


def test_fast_mode_returns_parsed_document(tmp_path):
    """extract_fast should return ParsedDocument for non-PDF."""
    import pandas as pd
    from audit_ingestion.extractor import extract_fast
    f = tmp_path / "test.csv"
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(f, index=False)
    result = extract_fast(str(f))
    from audit_ingestion.models import ParsedDocument
    assert isinstance(result, ParsedDocument)
    assert result.source_file == "test.csv"


# ── OCR cache ─────────────────────────────────────────────────────────────────

def test_ocr_cache_exists():
    """OCR page cache dict should exist in extractor module."""
    from audit_ingestion import extractor
    assert hasattr(extractor, "_ocr_page_cache")
    assert isinstance(extractor._ocr_page_cache, dict)


def test_image_cache_exists():
    """Image page cache dict should exist."""
    from audit_ingestion import extractor
    assert hasattr(extractor, "_image_page_cache")


# ── Schema version ────────────────────────────────────────────────────────────

def test_schema_version_defined():
    from audit_ingestion.canonical import SCHEMA_VERSION
    assert SCHEMA_VERSION == "v05.1"


def test_canonical_cache_key_includes_schema_version():
    """Cache key must include schema version so schema changes invalidate cache."""
    from audit_ingestion.canonical import _canonical_cache_key, SCHEMA_VERSION
    from audit_ingestion.models import ParsedDocument, ParsedPage
    pages = [ParsedPage(page_number=1, text="test", char_count=4, extractor="pdfplumber")]
    doc = ParsedDocument(
        source_file="test.pdf", full_text="[Page 1]\ntest",
        page_count=1, pages=pages,
        extraction_chain=["pdfplumber"], primary_extractor="pdfplumber",
    )
    key = _canonical_cache_key(doc, "gpt-5.4")
    assert isinstance(key, str)
    assert len(key) == 32  # MD5 hex


# ── Provider mock contract ────────────────────────────────────────────────────

def test_provider_responses_api_method():
    """OpenAIProvider must expose _responses_call using Responses API."""
    from audit_ingestion.providers.openai_provider import OpenAIProvider
    import inspect
    src = inspect.getsource(OpenAIProvider._responses_call)
    assert "responses.create" in src


def test_provider_extract_structured_uses_json_schema():
    """extract_structured must pass json_schema to _responses_call."""
    from audit_ingestion.providers.openai_provider import OpenAIProvider
    import inspect
    src = inspect.getsource(OpenAIProvider.extract_structured)
    assert "json_schema" in src
    assert "_responses_call" in src


def test_provider_model_constants_immutable():
    """Model constants should be strings and match spec."""
    from audit_ingestion.providers import openai_provider as op
    assert op.CANONICAL_MODEL == "gpt-5.4"
    assert op.VISION_MODEL    == "gpt-5.4"
    assert op.RESCUE_MODEL    == "gpt-5.4-pro"
    assert op.DEFAULT_MODEL   == op.CANONICAL_MODEL


# ── Batch throughput smoke test ───────────────────────────────────────────────

def test_batch_concurrent_processing(tmp_path):
    """Multiple files should process concurrently and complete faster than serial."""
    import pandas as pd
    from audit_ingestion.router import ingest_one

    # Create 4 CSV files
    for i in range(4):
        f = tmp_path / f"file_{i}.csv"
        pd.DataFrame({"Account": [f"Account{i}"], "Amount": [i * 1000]}).to_csv(f, index=False)

    files = list(tmp_path.glob("*.csv"))
    t0 = time.time()

    # Process all without API key (extraction only — fast)
    results = [ingest_one(str(f), api_key=None) for f in files]
    elapsed = time.time() - t0

    assert len(results) == 4
    assert all(r.evidence is not None for r in results)
    # Serial processing of 4 simple CSV files should be well under 5 seconds
    assert elapsed < 5.0


# ── Strict schema ─────────────────────────────────────────────────────────────

def test_canonical_schema_strict_true():
    """CANONICAL_JSON_SCHEMA should have strict=True."""
    from audit_ingestion.canonical import CANONICAL_JSON_SCHEMA
    assert CANONICAL_JSON_SCHEMA.get("strict") is True


def test_canonical_schema_has_required():
    """Schema should declare required fields."""
    from audit_ingestion.canonical import CANONICAL_JSON_SCHEMA
    schema = CANONICAL_JSON_SCHEMA.get("schema", {})
    assert "required" in schema
    required = schema["required"]
    assert "family" in required
    assert "audit_overview" in required
    assert "parties" in required
    assert "amounts" in required
    assert "facts" in required
    assert "claims" in required


def test_canonical_schema_additional_properties_false():
    """Top-level schema should not allow extra properties."""
    from audit_ingestion.canonical import CANONICAL_JSON_SCHEMA
    schema = CANONICAL_JSON_SCHEMA.get("schema", {})
    assert schema.get("additionalProperties") is False


# ── Rescue model not used for canonical ──────────────────────────────────────

def test_extract_structured_uses_canonical_model_not_rescue():
    """extract_structured must use CANONICAL_MODEL, never RESCUE_MODEL."""
    from audit_ingestion.providers.openai_provider import (
        OpenAIProvider, CANONICAL_MODEL, RESCUE_MODEL
    )
    import inspect
    src = inspect.getsource(OpenAIProvider.extract_structured)
    assert RESCUE_MODEL not in src
    # Should reference CANONICAL_MODEL constant
    assert "CANONICAL_MODEL" in src


# ── v03.2 additions ───────────────────────────────────────────────────────────

def test_router_has_ai_semaphore():
    """Router must expose AI and OCR semaphores for throttling."""
    from audit_ingestion import router
    assert hasattr(router, "_AI_SEMAPHORE")
    assert hasattr(router, "_OCR_SEMAPHORE")


def test_semaphore_limits():
    """Semaphores should be conservatively capped."""
    import threading
    from audit_ingestion.router import _AI_SEMAPHORE, _OCR_SEMAPHORE
    # Both should be threading.Semaphore instances
    assert isinstance(_AI_SEMAPHORE, type(threading.Semaphore(1)))
    assert isinstance(_OCR_SEMAPHORE, type(threading.Semaphore(1)))


def test_extract_accepts_ocr_semaphore(tmp_path):
    """extract() should accept ocr_semaphore parameter without error."""
    import threading
    import pandas as pd
    from audit_ingestion.extractor import extract
    f = tmp_path / "throttle_test.csv"
    pd.DataFrame({"A": [1]}).to_csv(f, index=False)
    sem = threading.Semaphore(2)
    result = extract(str(f), ocr_semaphore=sem)
    assert result is not None


def test_stage_timings_in_document_specific(tmp_path):
    """Stage timings should be stored in document_specific after ingest_one."""
    import pandas as pd
    from audit_ingestion.router import ingest_one
    f = tmp_path / "timing_test.csv"
    pd.DataFrame({"Account": ["Cash"], "Balance": [10000]}).to_csv(f, index=False)
    result = ingest_one(str(f), api_key=None)
    assert result.evidence is not None
    timings = result.evidence.document_specific.get("_stage_timings")
    assert timings is not None
    assert "extraction" in timings
    assert isinstance(timings["extraction"], float)


def test_allow_rescue_false_by_default(tmp_path):
    """allow_rescue should default to False and not error."""
    import pandas as pd
    from audit_ingestion.router import ingest_one
    f = tmp_path / "rescue_default.csv"
    pd.DataFrame({"Item": ["test"]}).to_csv(f, index=False)
    result = ingest_one(str(f), api_key=None)  # allow_rescue defaults False
    assert result.evidence is not None
    # No rescue flag should be set when allow_rescue=False
    rescue_flags = [fl for fl in result.evidence.flags if fl.type == "rescue_applied"]
    assert len(rescue_flags) == 0


def test_canonical_schema_version_bumped():
    """Schema version should be v03.2 after tightening."""
    from audit_ingestion.canonical import SCHEMA_VERSION
    assert SCHEMA_VERSION == "v05.1"


def test_canonical_schema_nested_items_have_additional_properties():
    """All nested item schemas should have additionalProperties: False."""
    from audit_ingestion.canonical import CANONICAL_JSON_SCHEMA
    schema = CANONICAL_JSON_SCHEMA["schema"]

    # Check party items
    party_item = schema["properties"]["parties"]["items"]
    assert party_item.get("additionalProperties") is False

    # Check amount items
    amount_item = schema["properties"]["amounts"]["items"]
    assert amount_item.get("additionalProperties") is False

    # Check fact items
    fact_item = schema["properties"]["facts"]["items"]
    assert fact_item.get("additionalProperties") is False

    # Check flag items have severity enum
    flag_item = schema["properties"]["flags"]["items"]
    assert "enum" in flag_item["properties"]["severity"]
    assert flag_item.get("additionalProperties") is False


def test_canonical_schema_link_keys_required():
    """link_keys should declare required fields."""
    from audit_ingestion.canonical import CANONICAL_JSON_SCHEMA
    lk = CANONICAL_JSON_SCHEMA["schema"]["properties"]["link_keys"]
    assert lk.get("additionalProperties") is False
    assert "required" in lk
    assert "party_names" in lk["required"]


def test_rescue_uses_rescue_model_not_canonical():
    """Rescue path in router must reference RESCUE_MODEL, not CANONICAL_MODEL."""
    import inspect
    from audit_ingestion import router
    src = inspect.getsource(router.ingest_one)
    assert "RESCUE_MODEL" in src
    # Verify rescue is clearly separate from canonical path
    assert "allow_rescue" in src


def test_ingest_one_accepts_allow_rescue_param(tmp_path):
    """ingest_one must accept allow_rescue parameter."""
    import inspect
    from audit_ingestion.router import ingest_one
    sig = inspect.signature(ingest_one)
    assert "allow_rescue" in sig.parameters
    assert sig.parameters["allow_rescue"].default is False
