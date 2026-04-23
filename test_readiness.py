"""Tests for legacy compatibility layer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from audit_ingestion.legacy import canonical_to_legacy_fields, canonical_summary_row
from audit_ingestion.models import (
    AuditEvidence, AuditOverview, DocumentFamily,
    Party, Amount, DateItem, Provenance,
)


def make_test_evidence():
    prov = Provenance(page=1, quote="test quote", confidence=0.9)
    return AuditEvidence(
        source_file="invoice_test.pdf",
        family=DocumentFamily.INVOICE,
        subtype="vendor_invoice",
        audit_overview=AuditOverview(
            summary="Vendor invoice from Staples for office supplies",
            audit_areas=["expenses", "payables"],
            assertions=["existence", "accuracy"],
        ),
        parties=[
            Party(role="vendor", name="Staples", normalized="STAPLES", provenance=prov),
            Party(role="client", name="Hope Community Services",
                  normalized="HOPE COMMUNITY SERVICES", provenance=prov),
        ],
        amounts=[Amount(type="invoice_total", value=413.64, provenance=prov)],
        dates=[DateItem(type="invoice_date", value="2024-01-25", provenance=prov)],
    )


def test_canonical_to_legacy_fields():
    ev = make_test_evidence()
    fields = canonical_to_legacy_fields(ev)
    assert "party_a" in fields
    assert fields["party_a"] == "Staples"
    assert "invoice_total" in fields
    assert fields["invoice_total"] == 413.64
    assert "summary" in fields
    assert "expenses" in fields.get("audit_areas", "")


def test_canonical_summary_row():
    ev = make_test_evidence()
    row = canonical_summary_row(ev)
    assert row["file"] == "invoice_test.pdf"
    assert row["family"] == "invoice_receipt"
    assert row["primary_party"] == "Staples"
    assert "$413.64" in row["primary_amount"]
    assert "expenses" in row["audit_areas"]


def test_legacy_fields_empty_evidence():
    ev = AuditEvidence(source_file="empty.pdf")
    fields = canonical_to_legacy_fields(ev)
    assert isinstance(fields, dict)
