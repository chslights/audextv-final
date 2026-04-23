"""
tests/test_readiness.py
Tests for the evidence readiness engine and question generation.
"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_ingestion.models import AuditEvidence, Flag, ExtractionMeta, ReadinessResult
from audit_ingestion.readiness import (
    compute_readiness, apply_readiness,
    READINESS_VERSION, _FLAG_RULES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ev(*flag_types, chars=5000, financial_data=None):
    """Build a minimal AuditEvidence with the given flag types."""
    flags = [Flag(type=ft, description=f"test {ft}", severity="warning")
             for ft in flag_types]
    ev = AuditEvidence(
        source_file="test.pdf",
        flags=flags,
        extraction_meta=ExtractionMeta(primary_extractor="pdfplumber", total_chars=chars),
        document_specific={"_financial": financial_data} if financial_data else {},
    )
    return ev


# ── Version ───────────────────────────────────────────────────────────────────

def test_version():
    assert READINESS_VERSION == "v05.0"


# ── Ready status ──────────────────────────────────────────────────────────────

def test_no_flags_is_ready():
    ev = make_ev()
    rd = compute_readiness(ev)
    assert rd.readiness_status == "ready"
    assert rd.blocking_state == "non_blocking"
    assert rd.questions == []


def test_info_only_flags_are_ready():
    ev = make_ev("outlier", "unusual_transaction", "bundle_detected")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "ready"
    assert rd.blocking_state == "non_blocking"
    assert rd.questions == []


def test_duplicate_entry_generates_blocking_question_even_when_critical():
    ev = AuditEvidence(
        source_file="gl.csv",
        flags=[Flag(type="duplicate_entry", description="duplicate posting noted", severity="critical")],
        extraction_meta=ExtractionMeta(primary_extractor="csv", total_chars=500),
        document_specific={"_financial": {"doc_type": "general_ledger", "finality_state": "trusted", "rows": [{"a": 1}], "period_start": "2024-01-01"}},
    )
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_client_answer"
    assert rd.blocking_state == "blocking"
    assert rd.questions
    assert rd.questions[0].source_flag == "duplicate_entry"


def test_rescue_applied_is_ready():
    ev = make_ev("rescue_applied", "governance_note", "document_nature")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "ready"


# ── Needs reviewer confirmation ───────────────────────────────────────────────

def test_tb_year_unconfirmed_blocks():
    ev = make_ev("tb_year_unconfirmed")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_reviewer_confirmation"
    assert rd.blocking_state == "blocking"
    assert "tb_year_unconfirmed" in rd.blocking_issues


def test_partial_extraction_blocks():
    ev = make_ev("partial_extraction")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_reviewer_confirmation"
    assert rd.blocking_state == "blocking"


def test_partial_extraction_visibility_blocks():
    ev = make_ev("partial_extraction_visibility")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_reviewer_confirmation"
    assert rd.blocking_state == "blocking"


# ── Needs client answer ───────────────────────────────────────────────────────

def test_missing_period_blocks_as_client():
    ev = make_ev("missing_period")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_client_answer"
    assert rd.blocking_state == "blocking"
    assert "missing_period" in rd.blocking_issues


def test_material_balance_difference_blocks_as_client():
    ev = make_ev("material_balance_difference")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_client_answer"
    assert rd.blocking_state == "blocking"


def test_unsigned_agreement_blocks_as_client():
    ev = make_ev("unsigned_agreement")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_client_answer"
    assert rd.blocking_state == "blocking"


def test_missing_page_blocks_as_client():
    ev = make_ev("missing_page")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_client_answer"
    assert rd.blocking_state == "blocking"


# ── Exception open (non-blocking questions) ───────────────────────────────────

def test_related_party_is_exception_open():
    ev = make_ev("related_party")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "exception_open"
    assert rd.blocking_state == "non_blocking"
    assert len(rd.questions) == 1
    assert rd.questions[0].blocking is False


def test_conditional_funding_is_exception_open():
    ev = make_ev("conditional_funding")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "exception_open"
    assert rd.blocking_state == "non_blocking"


def test_missing_account_number_non_blocking_for_pdf():
    """missing_account_number is non-blocking for PDF documents."""
    ev = make_ev("missing_account_number")
    rd = compute_readiness(ev)
    assert rd.blocking_state == "non_blocking"


# ── Mixed flags ───────────────────────────────────────────────────────────────

def test_reviewer_blocking_wins_over_client_blocking():
    """When both reviewer and client blocking issues exist, reviewer takes precedence."""
    ev = make_ev("tb_year_unconfirmed", "missing_period")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_reviewer_confirmation"
    assert len(rd.blocking_issues) == 2


def test_blocking_plus_nonblocking():
    """Blocking flag dominates — status is blocking even with non-blocking questions."""
    ev = make_ev("missing_period", "related_party", "duplicate_entry")
    rd = compute_readiness(ev)
    assert rd.readiness_status == "needs_client_answer"
    assert rd.blocking_state == "blocking"
    blocking_qs = [q for q in rd.questions if q.blocking]
    nonblocking_qs = [q for q in rd.questions if not q.blocking]
    assert len(blocking_qs) >= 1
    assert len(nonblocking_qs) >= 1


def test_duplicate_flags_produce_one_question():
    """The same flag type appearing twice should produce only one question."""
    ev = AuditEvidence(
        source_file="test.pdf",
        flags=[
            Flag(type="missing_period", description="first", severity="warning"),
            Flag(type="missing_period", description="second", severity="warning"),
        ],
        extraction_meta=ExtractionMeta(primary_extractor="pdfplumber", total_chars=500),
    )
    rd = compute_readiness(ev)
    period_qs = [q for q in rd.questions if q.source_flag == "missing_period"]
    assert len(period_qs) == 1


# ── Question structure ────────────────────────────────────────────────────────

def test_question_fields_populated():
    ev = make_ev("missing_period")
    rd = compute_readiness(ev)
    q = rd.questions[0]
    assert q.question_id.startswith("missing_period_")
    assert q.question_type == "period_confirmation"
    assert len(q.question_text) > 20
    assert q.audience == "client"
    assert q.blocking is True
    assert q.source_flag == "missing_period"
    assert q.resolved is False


def test_reviewer_question_audience():
    ev = make_ev("tb_year_unconfirmed")
    rd = compute_readiness(ev)
    assert rd.questions[0].audience == "reviewer"


def test_client_question_audience():
    ev = make_ev("missing_period")
    rd = compute_readiness(ev)
    assert rd.questions[0].audience == "client"


# ── Unusable ──────────────────────────────────────────────────────────────────

def test_critical_flag_is_unusable():
    ev = AuditEvidence(
        source_file="test.pdf",
        flags=[Flag(type="extraction_error", description="failed", severity="critical")],
        extraction_meta=ExtractionMeta(primary_extractor="none", total_chars=0),
    )
    rd = compute_readiness(ev)
    assert rd.readiness_status == "unusable"
    assert rd.evidence_use_mode == "unusable"


def test_no_text_is_unusable():
    ev = make_ev(chars=0)
    rd = compute_readiness(ev)
    assert rd.readiness_status == "unusable"


# ── Financial files ───────────────────────────────────────────────────────────

def test_financial_file_missing_account_number_blocking():
    """missing_account_number IS blocking for financial files (bank statement)."""
    fin_data = {
        "doc_type": "bank_statement_csv",
        "finality_state": "trusted",
        "period_start": "2024-01-01",
        "totals": {"total_inflows": 100000},
    }
    ev = make_ev("missing_account_number", financial_data=fin_data)
    rd = compute_readiness(ev)
    q = next((q for q in rd.questions if q.source_flag == "missing_account_number"), None)
    assert q is not None
    assert q.blocking is True


def test_population_ready_trusted_with_period_and_totals():
    fin_data = {
        "doc_type": "general_ledger",
        "finality_state": "trusted",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "totals": {"total_debits": 1000000, "total_credits": 900000},
        "rows": [{"account": "5010", "amount": 1000}],  # has rows
    }
    ev = make_ev(financial_data=fin_data)
    rd = compute_readiness(ev)
    assert rd.population_ready is True
    assert rd.evidence_use_mode == "evidence_and_population"


def test_population_not_ready_without_rows():
    fin_data = {
        "doc_type": "general_ledger",
        "finality_state": "trusted",
        "period_start": "2024-01-01",
        "totals": {"total_debits": 1000000},
        # no rows key
    }
    ev = make_ev(financial_data=fin_data)
    rd = compute_readiness(ev)
    assert rd.population_ready is False
    assert rd.evidence_use_mode == "evidence_only"
    assert "row-level data not yet retained" in rd.population_status


def test_population_not_ready_material_balance_diff():
    fin_data = {
        "doc_type": "trial_balance_current",
        "finality_state": "user_confirmed",
        "period_start": "2024",
        "totals": {"total_dr_balances": 2066500, "total_cr_balances": 2042400},
        "rows": [{"account_number": "1010", "balance": 285000}],
        "balance_check": {"flag_level": "material_balance_difference", "difference": 24100},
    }
    ev = make_ev("material_balance_difference", financial_data=fin_data)
    rd = compute_readiness(ev)
    assert rd.population_ready is False
    assert "material balance difference" in rd.population_status


def test_population_not_ready_review_required():
    fin_data = {
        "doc_type": "trial_balance_unknown_year",
        "finality_state": "review_required",
        "period_start": None,
        "totals": {},
    }
    ev = make_ev("tb_year_unconfirmed", financial_data=fin_data)
    rd = compute_readiness(ev)
    assert rd.population_ready is False


def test_non_financial_file_has_no_population_status():
    ev = make_ev()  # no financial_data
    rd = compute_readiness(ev)
    assert rd.population_ready is None
    assert rd.population_status is None


# ── apply_readiness mutates evidence ─────────────────────────────────────────

def test_apply_readiness_sets_evidence_readiness():
    ev = make_ev("missing_period")
    assert ev.readiness is None
    apply_readiness(ev)
    assert ev.readiness is not None
    assert ev.readiness.readiness_status == "needs_client_answer"


def test_apply_readiness_returns_same_object():
    ev = make_ev()
    result = apply_readiness(ev)
    assert result is ev


# ── Flag rules completeness ───────────────────────────────────────────────────

def test_all_real_data_flags_have_rules():
    """Every flag seen in real diagnostic data should be in _FLAG_RULES."""
    real_flags = {
        "conditional_revenue", "duplicate_entry", "legal_review_pending",
        "material_balance_difference", "missing_account_number",
        "missing_beginning_balance", "missing_ending_balance",
        "missing_period", "no_entity_name", "outlier",
        "partial_extraction", "partial_extraction_visibility",
        "related_party_transaction", "tb_year_unconfirmed", "unusual_transaction",
        "amended_award", "approximate_quantity", "attachment_reference",
        "bundle_detected", "conditional_funding", "conditional_payment",
        "date_inconsistency", "document_nature", "duplicate_entry",
        "governance_note", "missing_page", "missing_term_months",
        "reference_to_other_agreement", "reference_to_supporting_document",
        "reference_to_underlying_agreement", "related_party", "rescue_applied",
        # New flags from real batch runs
        "related_party_gift", "document_limitation", "conditional_approval",
        "executive_compensation_approval", "missing_identifier", "missing_balance",
    }
    missing = real_flags - set(_FLAG_RULES.keys())
    assert missing == set(), f"Flags not in _FLAG_RULES: {missing}"


def test_small_balance_difference_generates_non_blocking_reviewer_question():
    ev = AuditEvidence(
        source_file="tb.csv",
        extraction_meta=ExtractionMeta(total_chars=500),
        flags=[Flag(type="balance_difference_detected", description="small diff", severity="warning")],
        document_specific={"_financial": {"doc_type": "trial_balance_current", "finality_state": "trusted", "period_start": "FY2024", "rows": [{"row_index": 1}], "row_diagnostics": {"blocking_reasons": []}, "balance_check": {"flag_level": "balance_difference_detected"}}},
    )
    rd = compute_readiness(ev)
    assert rd.readiness_status == "exception_open"
    assert len(rd.questions) == 1
    q = rd.questions[0]
    assert q.question_type == "balance_difference_review"
    assert q.audience == "reviewer"
    assert q.blocking is False
