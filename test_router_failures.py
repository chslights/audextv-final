"""Tests for canonical Pydantic models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from audit_ingestion.models import (
    AuditEvidence, AuditOverview, AuditPeriod, DocumentFamily,
    Party, Amount, DateItem, Identifier, AssetItem,
    Fact, Claim, Flag, Provenance, LinkKeys, ExtractionMeta,
    ParsedPage, ParsedTable, ParsedDocument, IngestionResult,
)


def test_audit_evidence_minimal():
    ev = AuditEvidence(source_file="test.pdf")
    assert ev.source_file == "test.pdf"
    assert ev.family == DocumentFamily.OTHER
    assert ev.parties == []
    assert ev.amounts == []


def test_audit_evidence_full():
    prov = Provenance(page=1, quote="Fixed Charge Per Month: $2,273.00", confidence=0.98)
    ev = AuditEvidence(
        source_file="ryder_lease.pdf",
        family=DocumentFamily.CONTRACT,
        subtype="vehicle_lease",
        parties=[Party(role="lessor", name="Ryder", normalized="RYDER", provenance=prov)],
        amounts=[Amount(type="monthly_fixed_charge", value=2273.00, provenance=prov)],
        dates=[DateItem(type="schedule_date", value="2019-12-10", provenance=prov)],
        facts=[Fact(label="term_months", value=72, provenance=prov)],
        claims=[Claim(
            statement="72-month lease at $2,273/month",
            audit_area="leases",
            basis_fact_labels=["term_months", "monthly_fixed_charge"],
            provenance=prov,
        )],
        flags=[Flag(type="variable_cost",
                    description="Mileage rate creates variable obligation",
                    severity="info")],
    )
    assert ev.family == DocumentFamily.CONTRACT
    assert ev.parties[0].normalized == "RYDER"
    assert ev.amounts[0].value == 2273.00
    assert ev.facts[0].label == "term_months"
    assert ev.claims[0].audit_area == "leases"
    assert ev.flags[0].severity == "info"


def test_parsed_document():
    pg1 = ParsedPage(page_number=1, text="Fixed Charge Per Month: $2,273.00",
                     extractor="pdfplumber", confidence=0.9)
    pg2 = ParsedPage(page_number=2, text="", extractor="pdfplumber", confidence=0.0)
    doc = ParsedDocument(
        source_file="test.pdf",
        full_text="Fixed Charge Per Month: $2,273.00",
        page_count=2,
        pages=[pg1, pg2],
        extraction_chain=["pdfplumber"],
        primary_extractor="pdfplumber",
        confidence=0.7,
        weak_pages=[2],
    )
    assert doc.page_count == 2
    assert len(doc.weak_pages) == 1
    assert doc.chars_per_page == len(doc.full_text) / 2
    assert not doc.is_sufficient  # Too few chars


def test_parsed_page_char_count():
    text = "Hello world this is a test document"
    pg = ParsedPage(page_number=1, text=text, extractor="pdfplumber")
    assert pg.char_count == len(text)


def test_link_keys_defaults():
    lk = LinkKeys()
    assert lk.party_names == []
    assert lk.recurring_amounts == []


def test_extraction_meta_defaults():
    meta = ExtractionMeta(primary_extractor="pdfplumber")
    assert meta.needs_human_review is True
    assert meta.canonical_validated is False


def test_ingestion_result():
    ev = AuditEvidence(source_file="test.pdf")
    result = IngestionResult(evidence=ev, status="success")
    assert result.status == "success"
    assert result.evidence.source_file == "test.pdf"


def test_document_family_values():
    assert DocumentFamily.CONTRACT == "contract_agreement"
    assert DocumentFamily.INVOICE == "invoice_receipt"
    assert DocumentFamily.GRANT == "grant_donor_funding"
