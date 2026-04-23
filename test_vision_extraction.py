"""Tests for canonical extraction parsing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from audit_ingestion.canonical import (
    _parse_response, _parse_provenance, build_relevant_page_context,
)
from audit_ingestion.models import (
    ParsedDocument, ParsedPage, ExtractionMeta, DocumentFamily,
)


def make_parsed_doc(pages_text: list[str]) -> ParsedDocument:
    pages = [
        ParsedPage(page_number=i+1, text=t, char_count=len(t), extractor="pdfplumber")
        for i, t in enumerate(pages_text)
    ]
    full_text = "\n\n".join(f"[Page {i+1}]\n{t}" for i, t in enumerate(pages_text))
    return ParsedDocument(
        source_file="test.pdf",
        full_text=full_text,
        page_count=len(pages),
        pages=pages,
        extraction_chain=["pdfplumber"],
        primary_extractor="pdfplumber",
        confidence=0.8,
    )


def test_parse_provenance_valid():
    prov = _parse_provenance({"page": 1, "quote": "test quote", "confidence": 0.95})
    assert prov is not None
    assert prov.page == 1
    assert prov.confidence == 0.95


def test_parse_provenance_none():
    assert _parse_provenance(None) is None
    assert _parse_provenance({}) is None  # Empty dict has no meaningful provenance


def test_build_relevant_page_context_short():
    doc = make_parsed_doc(["Page one content", "Page two content"])
    ctx = build_relevant_page_context(doc, max_chars=5000)
    assert "Page 1" in ctx
    assert "Page 2" in ctx


def test_build_relevant_page_context_prioritizes_first_pages():
    pages = [f"Content of page {i}" for i in range(10)]
    doc = make_parsed_doc(pages)
    ctx = build_relevant_page_context(doc, max_chars=500)
    # First page should always be included
    assert "Page 1" in ctx


def test_build_relevant_page_context_includes_tables():
    doc = make_parsed_doc(["Invoice content"])
    doc.tables = [{"page_number": 1, "headers": ["Item", "Amount"],
                   "rows": [{"Item": "Service", "Amount": "$500"}]}]
    ctx = build_relevant_page_context(doc, max_chars=5000)
    assert "EXTRACTED TABLES" in ctx


def test_parse_response_minimal():
    meta = ExtractionMeta(primary_extractor="pdfplumber")
    doc = make_parsed_doc(["Test content"])
    data = {
        "family": "invoice_receipt",
        "subtype": "vendor_invoice",
        "title": "Test Invoice",
        "audit_overview": {
            "summary": "A test invoice",
            "audit_areas": ["expenses"],
            "assertions": ["existence"],
            "period": None,
            "match_targets": [],
        },
        "parties": [{"role": "vendor", "name": "Test Vendor", "normalized": "TEST VENDOR",
                     "provenance": {"page": 1, "quote": "Test Vendor", "confidence": 0.9}}],
        "amounts": [{"type": "invoice_total", "value": 500.0, "currency": "USD",
                     "provenance": {"page": 1, "quote": "$500.00", "confidence": 0.95}}],
        "dates": [{"type": "invoice_date", "value": "2024-01-15",
                   "provenance": {"page": 1, "quote": "January 15, 2024", "confidence": 0.9}}],
        "identifiers": [],
        "assets": [],
        "facts": [{"label": "invoice_number", "value": "INV-001",
                   "provenance": {"page": 1, "quote": "INV-001", "confidence": 0.99}}],
        "claims": [{"statement": "Invoice of $500 from Test Vendor",
                    "audit_area": "expenses",
                    "basis_fact_labels": ["invoice_total"],
                    "provenance": {"page": 1, "quote": "$500", "confidence": 0.95}}],
        "flags": [],
        "link_keys": {"party_names": ["TEST VENDOR"], "document_numbers": [],
                      "agreement_numbers": [], "invoice_numbers": ["INV-001"],
                      "asset_descriptions": [], "recurring_amounts": [],
                      "key_dates": ["2024-01-15"], "other_ids": []},
        "document_specific": {},
    }

    evidence = _parse_response(data, "test.pdf", doc, meta)
    assert evidence.family == DocumentFamily.INVOICE
    assert evidence.parties[0].name == "Test Vendor"
    assert evidence.amounts[0].value == 500.0
    assert evidence.facts[0].label == "invoice_number"
    assert evidence.claims[0].audit_area == "expenses"
    assert "TEST VENDOR" in evidence.link_keys.party_names
