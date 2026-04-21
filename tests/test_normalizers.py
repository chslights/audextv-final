"""Tests for normalizers module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from audit_ingestion.normalizers import (
    normalize_party_name, normalize_date, normalize_amount,
    normalize_identifier, build_link_keys, normalize_evidence,
    dedupe_parties, dedupe_amounts,
)
from audit_ingestion.models import (
    AuditEvidence, Party, Amount, DateItem, Identifier,
    Fact, Provenance,
)


def test_normalize_party_name():
    assert normalize_party_name("Ryder Truck Rental, Inc.") == "RYDER TRUCK RENTAL INC"
    assert normalize_party_name("  Hope Community Services  ") == "HOPE COMMUNITY SERVICES"
    assert normalize_party_name("") == ""


def test_normalize_date_iso():
    assert normalize_date("2019-12-10") == "2019-12-10"


def test_normalize_date_slash():
    assert normalize_date("12/10/2019") == "2019-12-10"


def test_normalize_date_written():
    assert normalize_date("December 10, 2019") == "2019-12-10"


def test_normalize_date_invalid():
    assert normalize_date("not a date") is None
    assert normalize_date("") is None


def test_normalize_amount_plain():
    assert normalize_amount(2273.00) == 2273.00
    assert normalize_amount("2273") == 2273.0


def test_normalize_amount_currency_string():
    assert normalize_amount("$2,273.00") == 2273.0
    assert normalize_amount("$88,112") == 88112.0


def test_normalize_amount_none():
    assert normalize_amount(None) is None
    assert normalize_amount("not a number") is None


def test_normalize_identifier():
    assert normalize_identifier("2329-109") == "2329109"
    assert normalize_identifier("inv #0089") == "INV#0089"


def test_dedupe_parties():
    prov = Provenance(page=1, confidence=0.9)
    parties = [
        Party(role="lessor", name="Ryder", normalized="RYDER", provenance=prov),
        Party(role="lessor", name="Ryder", normalized="RYDER", provenance=prov),
        Party(role="lessee", name="Test Co", normalized="TEST CO", provenance=prov),
    ]
    deduped = dedupe_parties(parties)
    assert len(deduped) == 2


def test_dedupe_amounts():
    amounts = [
        Amount(type="monthly_charge", value=2273.0),
        Amount(type="monthly_charge", value=2273.0),
        Amount(type="total", value=88112.0),
    ]
    deduped = dedupe_amounts(amounts)
    assert len(deduped) == 2


def test_build_link_keys():
    prov = Provenance(page=1, confidence=0.95)
    ev = AuditEvidence(
        source_file="test.pdf",
        parties=[Party(role="lessor", name="Ryder", normalized="RYDER", provenance=prov)],
        amounts=[Amount(type="monthly_fixed_charge", value=2273.0, provenance=prov)],
        dates=[DateItem(type="schedule_date", value="2019-12-10", provenance=prov)],
        identifiers=[Identifier(type="schedule_number", value="2329109", provenance=prov)],
    )
    lk = build_link_keys(ev)
    assert "RYDER" in lk.party_names
    assert 2273.0 in lk.recurring_amounts
    assert "2019-12-10" in lk.key_dates
    assert "2329109" in lk.document_numbers


def test_normalize_evidence():
    prov = Provenance(page=1, confidence=0.9)
    ev = AuditEvidence(
        source_file="test.pdf",
        parties=[Party(role="vendor", name="Staples Business Advantage",
                       normalized="", provenance=prov)],
        amounts=[Amount(type="invoice_total", value=89.9, provenance=prov)],
        dates=[DateItem(type="invoice_date", value="January 25, 2024", provenance=prov)],
    )
    normalized = normalize_evidence(ev)
    assert normalized.parties[0].normalized == "STAPLES BUSINESS ADVANTAGE"
    assert normalized.dates[0].value == "2024-01-25"
    assert normalized.link_keys.party_names == ["STAPLES BUSINESS ADVANTAGE"]
