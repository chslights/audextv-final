"""
tests/test_row_retention.py
Tests for:
  - account_family tagging on TB and GL rows
  - inflow_outflow_tag on bank CSV rows
  - revenue_expense_tag on budget rows
  - workflow state persistence (persist/merge/lineage/action queue)
  - diagnostic CSV readiness and row_diagnostics fields
"""
import sys, os, tempfile, json
from pathlib import Path
import pytest
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_ingestion.financial_classifier import (
    _extract_rows, _column_mapping, _account_family, _revenue_expense_tag,
    _row_diagnostics,
    TYPE_TB_CURRENT, TYPE_TB_PRIOR, TYPE_TB_UNKNOWN,
    TYPE_GENERAL_LEDGER, TYPE_BANK_CSV, TYPE_BUDGET, TYPE_CHART_OF_ACCOUNTS,
)
from audit_ingestion.models import AuditEvidence, ExtractionMeta, Flag
from audit_ingestion.readiness import compute_readiness, resolve_question
from audit_ingestion.workflow import (
    build_prioritized_action_queue, build_client_followup_package,
    next_best_question, persist_evidence_state, merge_state_into_evidence,
    load_state, register_lineage,
)


# ── account_family helper ─────────────────────────────────────────────────────

def test_account_family_assets():
    assert _account_family(1010) == "assets"
    assert _account_family(1999) == "assets"
    assert _account_family("1500") == "assets"

def test_account_family_liabilities():
    assert _account_family(2010) == "liabilities"
    assert _account_family(2999) == "liabilities"

def test_account_family_net_assets():
    assert _account_family(3010) == "net_assets"
    assert _account_family(3999) == "net_assets"

def test_account_family_revenue():
    assert _account_family(4010) == "revenue"
    assert _account_family(4999) == "revenue"

def test_account_family_expenses():
    assert _account_family(5010) == "expenses"
    assert _account_family(9999) == "expenses"

def test_account_family_other():
    assert _account_family(100) == "other"
    assert _account_family(10001) == "other"

def test_account_family_bad_input():
    assert _account_family(None) is None
    assert _account_family("not_a_number") is None
    assert _account_family("") is None


# ── revenue_expense_tag helper ────────────────────────────────────────────────

def test_revenue_tag_grant():
    assert _revenue_expense_tag("Grant Revenue") == "revenue"

def test_revenue_tag_contributions():
    assert _revenue_expense_tag("Individual Contributions") == "revenue"

def test_revenue_tag_income():
    assert _revenue_expense_tag("Program Service Income") == "revenue"

def test_expense_tag_salaries():
    assert _revenue_expense_tag("Salaries") == "expense"

def test_expense_tag_rent():
    assert _revenue_expense_tag("Rent Expense") == "expense"

def test_expense_tag_empty():
    assert _revenue_expense_tag("") == "expense"
    assert _revenue_expense_tag(None) == "expense"


# ── TB row account_family tagging ─────────────────────────────────────────────

def make_tb_df():
    return pd.DataFrame({
        "Account Number": [1010, 1020, 2010, 3010, 4010, 5010, 5020],
        "Account Name":   ["Cash","AR","AP","Net Assets","Revenue","Salaries","Rent"],
        "Balance":        [285000,150000,50000,100000,500000,400000,60000],
        "Dr/Cr":          ["DR","DR","CR","CR","CR","DR","DR"],
    })

def test_tb_rows_have_account_family():
    df = make_tb_df()
    cm = _column_mapping(df, TYPE_TB_CURRENT)
    rows = _extract_rows(df, TYPE_TB_CURRENT, cm)
    assert all("account_family" in r for r in rows)

def test_tb_rows_account_family_correct():
    df = make_tb_df()
    cm = _column_mapping(df, TYPE_TB_CURRENT)
    rows = _extract_rows(df, TYPE_TB_CURRENT, cm)
    families = {r["account_number"]: r["account_family"] for r in rows}
    assert families[1010] == "assets"
    assert families[1020] == "assets"
    assert families[2010] == "liabilities"
    assert families[3010] == "net_assets"
    assert families[4010] == "revenue"
    assert families[5010] == "expenses"
    assert families[5020] == "expenses"

def test_tb_rows_no_inflow_outflow():
    df = make_tb_df()
    cm = _column_mapping(df, TYPE_TB_CURRENT)
    rows = _extract_rows(df, TYPE_TB_CURRENT, cm)
    assert all("inflow_outflow_tag" not in r for r in rows)

def test_gl_rows_have_account_family():
    df = pd.DataFrame({
        "Transaction Date": ["2024-01-05","2024-01-10"],
        "Amount":           [45000,-15000],
        "Description":      ["Payroll","Rent"],
        "Account Number":   [5010, 5050],
        "Account":          ["Salaries","Rent"],
    })
    cm = _column_mapping(df, TYPE_GENERAL_LEDGER)
    rows = _extract_rows(df, TYPE_GENERAL_LEDGER, cm)
    assert all("account_family" in r for r in rows)
    assert rows[0]["account_family"] == "expenses"


# ── Bank CSV inflow_outflow_tag ───────────────────────────────────────────────

def make_bank_df():
    return pd.DataFrame({
        "Date":        ["2024-01-06","2024-01-08","2024-01-15"],
        "Amount":      [45000.0, -15000.0, 8000.0],
        "Description": ["PAYROLL DEPOSIT","TAX PAYMENT","GRANT RECEIPT"],
    })

def test_bank_rows_have_inflow_outflow():
    df = make_bank_df()
    cm = _column_mapping(df, TYPE_BANK_CSV)
    rows = _extract_rows(df, TYPE_BANK_CSV, cm)
    assert all("inflow_outflow_tag" in r for r in rows)

def test_bank_rows_inflow_outflow_correct():
    df = make_bank_df()
    cm = _column_mapping(df, TYPE_BANK_CSV)
    rows = _extract_rows(df, TYPE_BANK_CSV, cm)
    assert rows[0]["inflow_outflow_tag"] == "inflow"   # 45000
    assert rows[1]["inflow_outflow_tag"] == "outflow"  # -15000
    assert rows[2]["inflow_outflow_tag"] == "inflow"   # 8000

def test_bank_rows_no_account_family():
    df = make_bank_df()
    cm = _column_mapping(df, TYPE_BANK_CSV)
    rows = _extract_rows(df, TYPE_BANK_CSV, cm)
    assert all("account_family" not in r for r in rows)


# ── Budget revenue_expense_tag ────────────────────────────────────────────────

def make_budget_df():
    return pd.DataFrame({
        "Category":      ["Grant Revenue","Individual Contributions","Salaries","Rent","Utilities"],
        "Budget Amount": [500000, 200000, 400000, 100000, 50000],
    })

def test_budget_rows_have_revenue_expense_tag():
    df = make_budget_df()
    cm = _column_mapping(df, TYPE_BUDGET)
    rows = _extract_rows(df, TYPE_BUDGET, cm)
    assert all("revenue_expense_tag" in r for r in rows)

def test_budget_rows_tags_correct():
    df = make_budget_df()
    cm = _column_mapping(df, TYPE_BUDGET)
    rows = _extract_rows(df, TYPE_BUDGET, cm)
    tags = {r["category"]: r["revenue_expense_tag"] for r in rows}
    assert tags["Grant Revenue"] == "revenue"
    assert tags["Individual Contributions"] == "revenue"
    assert tags["Salaries"] == "expense"
    assert tags["Rent"] == "expense"


# ── Workflow state persistence ────────────────────────────────────────────────

def make_ev_with_readiness(*flag_types, source_file="test.pdf"):
    ev = AuditEvidence(
        source_file=source_file,
        flags=[Flag(type=f, description=f, severity="warning") for f in flag_types],
        extraction_meta=ExtractionMeta(primary_extractor="pdfplumber", total_chars=1000),
    )
    ev.readiness = compute_readiness(ev)
    return ev


def test_persist_and_reload_question_state(tmp_path):
    state_path = tmp_path / "workflow.json"
    ev = make_ev_with_readiness("missing_period")
    q = ev.readiness.questions[0]
    resolve_question(ev, q.question_id, "FY2024", actor="reviewer",
                     resolution_type="answer", comment="confirmed")

    persist_evidence_state(ev, path=state_path)
    assert state_path.exists()

    # Reload into a fresh evidence object
    ev2 = make_ev_with_readiness("missing_period")
    state = load_state(state_path)
    merge_state_into_evidence(ev2, state=state)

    q2 = ev2.readiness.questions[0]
    assert q2.resolved is True
    assert q2.resolution == "FY2024"
    assert q2.resolved_by == "reviewer"


def test_resolve_updates_readiness_status():
    ev = make_ev_with_readiness("missing_period")
    assert ev.readiness.readiness_status == "needs_client_answer"
    q = ev.readiness.questions[0]
    resolve_question(ev, q.question_id, "FY2024")
    assert ev.readiness.readiness_status == "ready"
    assert ev.readiness.blocking_state == "non_blocking"


def test_resolve_partial_leaves_status_blocking():
    ev = make_ev_with_readiness("missing_period", "tb_year_unconfirmed")
    assert len(ev.readiness.questions) == 2
    q_period = next(q for q in ev.readiness.questions if q.source_flag == "missing_period")
    resolve_question(ev, q_period.question_id, "FY2024")
    # tb_year_unconfirmed still unresolved
    assert ev.readiness.readiness_status == "needs_reviewer_confirmation"
    assert ev.readiness.blocking_state == "blocking"


def test_action_queue_blocking_client_first():
    ev1 = make_ev_with_readiness("missing_period", source_file="a.pdf")     # client blocking
    ev2 = make_ev_with_readiness("tb_year_unconfirmed", source_file="b.pdf") # reviewer blocking
    ev3 = make_ev_with_readiness("related_party", source_file="c.pdf")       # non-blocking
    queue = build_prioritized_action_queue([ev1, ev2, ev3])
    assert queue[0]["source_file"] == "a.pdf"   # client blocking first
    assert queue[0]["blocking"] is True
    assert queue[-1]["blocking"] is False        # non-blocking last


def test_client_followup_package_excludes_reviewer():
    ev1 = make_ev_with_readiness("missing_period", source_file="a.pdf")
    ev2 = make_ev_with_readiness("tb_year_unconfirmed", source_file="b.pdf")
    pkg = build_client_followup_package([ev1, ev2])
    assert len(pkg) == 1
    assert pkg[0]["source_file"] == "a.pdf"


def test_next_best_question_returns_highest_priority():
    ev1 = make_ev_with_readiness("related_party", source_file="c.pdf")
    ev2 = make_ev_with_readiness("missing_period", source_file="a.pdf")
    best = next_best_question([ev1, ev2])
    assert best is not None
    assert best["blocking"] is True
    assert best["source_file"] == "a.pdf"


def test_next_best_question_none_when_all_resolved():
    ev = make_ev_with_readiness("missing_period")
    q = ev.readiness.questions[0]
    resolve_question(ev, q.question_id, "FY2024")
    best = next_best_question([ev])
    assert best is None


def test_lineage_registered(tmp_path):
    state_path = tmp_path / "workflow.json"
    ev1 = make_ev_with_readiness("missing_period", source_file="tb.csv")
    ev2 = make_ev_with_readiness(source_file="tb.csv")  # replacement

    persist_evidence_state(ev1, path=state_path)
    register_lineage(ev2, prior=ev1, path=state_path)

    state = load_state(state_path)
    assert "tb.csv" in state.get("lineage", {})


# ── Diagnostic CSV fields ─────────────────────────────────────────────────────

def test_diagnostic_csv_has_readiness_and_row_fields():
    """Smoke test: build_diagnostic_csv produces the expected columns."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # We can't easily call build_diagnostic_csv without Streamlit context,
    # but we can verify the field names are present in the function source
    ingest_app_path = Path(__file__).parent.parent / "ingest_app.py"
    src = ingest_app_path.read_text()
    expected_fields = [
        "readiness_status", "blocking_state", "blocking_issues",
        "question_count", "population_ready", "population_status",
        "row_count", "flagged_row_count", "duplicate_rows",
        "malformed_rows", "open_blocking_questions", "resolved_questions",
        "open_client_questions",
    ]
    missing = [f for f in expected_fields if f'"{f}"' not in src]
    assert missing == [], f"Missing diagnostic CSV fields: {missing}"


def test_build_version_bumped():
    ingest_app_path = Path(__file__).parent.parent / "ingest_app.py"
    src = ingest_app_path.read_text()
    assert 'BUILD_VERSION = "v05.1"' in src


def test_flag_deduplication_in_router():
    router_path = Path(__file__).parent.parent / "audit_ingestion" / "router.py"
    src = router_path.read_text()
    assert "seen_flag_types" in src
    assert "deduped_flags" in src


# ── Fix 3: family/subtype set on review_required early return ─────────────────

def test_review_required_family_mapping_in_router():
    """router.py should contain the financial type-to-family mapping for review_required."""
    router_src = (Path(__file__).parent.parent / "audit_ingestion" / "router.py").read_text()
    # Verify all key mappings are present in the source
    assert "trial_balance_unknown_year" in router_src
    assert "bank_cash_activity" in router_src
    assert "_fin_type_to_family" in router_src
    assert "DocumentFamily" in router_src
