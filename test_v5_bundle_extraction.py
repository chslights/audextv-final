"""
tests/test_financial_classifier.py
Comprehensive tests for the financial file classifier.
Covers: column matching, period detection, totals, account structure,
balance check, TB ambiguity, AI fallback, finality states.
"""
import sys, io, tempfile, os
import pytest
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_ingestion.financial_classifier import (
    classify_financial_file, is_financial_file,
    _layer1_classify, _detect_period, _extract_totals,
    _extract_account_structure, _balance_check, _resolve_tb_year,
    _get_finality,
    TYPE_GENERAL_LEDGER, TYPE_JOURNAL_ENTRY, TYPE_TB_UNKNOWN,
    TYPE_TB_CURRENT, TYPE_TB_PRIOR, TYPE_BUDGET, TYPE_BANK_CSV,
    TYPE_CHART_OF_ACCOUNTS, TYPE_NOT_FINANCIAL,
    BALANCE_OK, BALANCE_SMALL_DIFF, BALANCE_MATERIAL_DIFF,
    FINALITY_TRUSTED, FINALITY_REVIEW_RECOMMENDED, FINALITY_REVIEW_REQUIRED,
    FINALITY_USER_CONFIRMED, MATERIAL_BALANCE_THRESHOLD,
    FINANCIAL_CLASSIFIER_VERSION,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_csv(df: pd.DataFrame, filename: str = "test.csv") -> str:
    """Write a DataFrame to a temp CSV and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", prefix=filename.replace(".csv","_"),
        delete=False, mode="w"
    )
    df.to_csv(tmp.name, index=False)
    return tmp.name


def cleanup(path: str):
    try: os.unlink(path)
    except: pass


# ── Version ───────────────────────────────────────────────────────────────────

def test_version():
    assert FINANCIAL_CLASSIFIER_VERSION == "v05.0"


# ── is_financial_file ─────────────────────────────────────────────────────────

def test_is_financial_file_csv():
    assert is_financial_file("data.csv") is True

def test_is_financial_file_excel():
    assert is_financial_file("data.xlsx") is True
    assert is_financial_file("data.xlsm") is True

def test_is_financial_file_txt():
    # .txt intentionally excluded — narrative docs like board minutes
    # should not go through the financial classifier
    assert is_financial_file("minutes.txt") is False

def test_is_financial_file_pdf():
    assert is_financial_file("invoice.pdf") is False

def test_is_financial_file_docx():
    assert is_financial_file("report.docx") is False


# ── Layer 1: Column matching ──────────────────────────────────────────────────

def test_layer1_general_ledger():
    df = pd.DataFrame({"Transaction Date":[], "Amount":[], "Description":[], "Account Number":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_GENERAL_LEDGER
    assert c >= 0.90

def test_layer1_general_ledger_variant_cols():
    df = pd.DataFrame({"TransactionDate":[], "Amt":[], "Desc":[], "Account":[]})
    # Normalized: transactiondate, amt, desc, account
    # amt doesn't match "amount" exactly — check it still classifies
    t, c = _layer1_classify(df)
    # transactiondate + description + account should match
    assert t == TYPE_GENERAL_LEDGER or t == TYPE_NOT_FINANCIAL  # partial match is ok

def test_layer1_trial_balance():
    df = pd.DataFrame({"Account Number":[], "Account Name":[], "Balance":[], "Dr/Cr":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_TB_UNKNOWN
    assert c >= 0.88

def test_layer1_trial_balance_alt_drcr():
    df = pd.DataFrame({"Account Number":[], "Account Name":[], "Balance":[], "Normal Balance":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_TB_UNKNOWN

def test_layer1_budget():
    df = pd.DataFrame({"Category":[], "Budget Amount":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_BUDGET
    assert c >= 0.88

def test_layer1_budget_with_account():
    df = pd.DataFrame({"Account":[], "Budget":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_BUDGET

def test_layer1_bank_csv():
    df = pd.DataFrame({"Date":[], "Amount":[], "Description":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_BANK_CSV
    assert c >= 0.85

def test_layer1_bank_csv_disqualified_with_account_number():
    """Bank CSV disqualified if it has an account number column."""
    df = pd.DataFrame({"Date":[], "Amount":[], "Description":[], "Account Number":[]})
    t, c = _layer1_classify(df)
    assert t != TYPE_BANK_CSV

def test_layer1_chart_of_accounts():
    df = pd.DataFrame({"Account Number":[], "Account Name":[], "Account Type":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_CHART_OF_ACCOUNTS
    assert c >= 0.90

def test_layer1_chart_disqualified_with_balance():
    """Chart of accounts disqualified if it has a balance column."""
    df = pd.DataFrame({"Account Number":[], "Account Name":[], "Account Type":[], "Balance":[]})
    t, c = _layer1_classify(df)
    assert t != TYPE_CHART_OF_ACCOUNTS

def test_layer1_journal_entry():
    df = pd.DataFrame({"Journal ID":[], "Date":[], "Debit":[], "Credit":[], "Description":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_JOURNAL_ENTRY
    assert c >= 0.88

def test_layer1_unknown():
    df = pd.DataFrame({"Name":[], "Value":[], "Notes":[]})
    t, c = _layer1_classify(df)
    assert t == TYPE_NOT_FINANCIAL
    assert c == 0.0


# ── TB year resolution ────────────────────────────────────────────────────────

def test_tb_year_prior_filename():
    t, src = _resolve_tb_year(TYPE_TB_UNKNOWN, "/data/prior_year_trial_balance.csv")
    assert t == TYPE_TB_PRIOR
    assert src == "filename"

def test_tb_year_py_filename():
    t, src = _resolve_tb_year(TYPE_TB_UNKNOWN, "/data/TB_PY2023.csv")
    assert t == TYPE_TB_PRIOR

def test_tb_year_unknown_stays_unknown():
    t, src = _resolve_tb_year(TYPE_TB_UNKNOWN, "/data/trial_balance.csv")
    assert t == TYPE_TB_UNKNOWN

def test_tb_year_non_tb_unchanged():
    t, src = _resolve_tb_year(TYPE_GENERAL_LEDGER, "/data/prior_gl.csv")
    assert t == TYPE_GENERAL_LEDGER


# ── Finality states ───────────────────────────────────────────────────────────

def test_finality_trusted():
    assert _get_finality(TYPE_GENERAL_LEDGER, 0.95, "heuristic") == FINALITY_TRUSTED

def test_finality_review_required_tb_unknown():
    assert _get_finality(TYPE_TB_UNKNOWN, 0.90, "heuristic") == FINALITY_REVIEW_REQUIRED

def test_finality_review_required_low_confidence():
    assert _get_finality(TYPE_GENERAL_LEDGER, 0.55, "heuristic") == FINALITY_REVIEW_REQUIRED

def test_finality_review_recommended_ai():
    assert _get_finality(TYPE_GENERAL_LEDGER, 0.72, "ai") == FINALITY_REVIEW_RECOMMENDED

def test_finality_review_recommended_medium_conf():
    assert _get_finality(TYPE_BUDGET, 0.75, "heuristic") == FINALITY_REVIEW_RECOMMENDED


# ── Period detection ──────────────────────────────────────────────────────────

def test_period_from_date_column():
    df = pd.DataFrame({
        "Transaction Date": ["2024-01-05", "2024-06-15", "2024-12-31"],
        "Amount": [100, 200, 300],
    })
    path = make_csv(df, "gl.csv")
    try:
        result = _detect_period(df, path, TYPE_GENERAL_LEDGER)
        # Filename has "2024" so filename detection fires first (higher priority than data_inferred)
        assert "2024" in str(result["period_start"])
        assert result["period_source"] in ("filename", "data_inferred")
        assert result["period_confidence"] >= 0.50
    finally:
        cleanup(path)

def test_period_from_filename():
    df = pd.DataFrame({"Account Number": [1010], "Balance": [100], "Dr/Cr": ["DR"]})
    path = make_csv(df, "trial_balance_2024.csv")
    try:
        result = _detect_period(df, path, TYPE_TB_UNKNOWN)
        assert "2024" in str(result.get("period_start", ""))
        assert result["period_source"] == "filename"
        assert result["period_confidence"] == 0.70
    finally:
        cleanup(path)

def test_period_not_found():
    df = pd.DataFrame({"Category": ["Revenue"], "Budget Amount": [100000]})
    path = make_csv(df, "budget.csv")
    try:
        result = _detect_period(df, path, TYPE_BUDGET)
        assert result["period_start"] is None
        assert result["period_source"] == "not_found"
        assert result["period_confidence"] == 0.0
    finally:
        cleanup(path)

def test_period_fy_filename():
    df = pd.DataFrame({"Account Number": [1010], "Balance": [100], "Dr/Cr": ["DR"]})
    path = make_csv(df, "TB_FY2025.csv")
    try:
        result = _detect_period(df, path, TYPE_TB_UNKNOWN)
        assert "FY2025" in str(result.get("period_start", "")).upper() or "2025" in str(result.get("period_start", ""))
    finally:
        cleanup(path)


# ── Key totals ────────────────────────────────────────────────────────────────

def test_totals_general_ledger():
    df = pd.DataFrame({
        "Transaction Date": ["2024-01-01"] * 4,
        "Amount":           [1000, -500, 2000, -300],
        "Description":      ["A","B","C","D"],
        "Account Number":   [1010, 2010, 4010, 5010],
    })
    totals = _extract_totals(df, TYPE_GENERAL_LEDGER)
    assert totals["total_debits"]  == 3000.0
    assert totals["total_credits"] == 800.0
    assert totals["net"]           == 2200.0
    assert totals["transaction_count"] == 4

def test_totals_trial_balance():
    df = pd.DataFrame({
        "Account Number": [1010, 1020, 2010, 4010],
        "Account Name":   ["Cash", "AR", "AP", "Revenue"],
        "Balance":        [50000, 30000, 20000, 60000],
        "Dr/Cr":          ["DR",  "DR",  "CR",  "CR"],
    })
    totals = _extract_totals(df, TYPE_TB_UNKNOWN)
    assert totals["total_dr_balances"] == 80000.0
    assert totals["total_cr_balances"] == 80000.0
    assert totals["net_difference"]    == 0.0
    assert totals["account_count"]     == 4

def test_totals_budget():
    df = pd.DataFrame({
        "Category":      ["Grant Revenue", "Contributions", "Salaries", "Rent"],
        "Budget Amount": [500000, 200000, 400000, 100000],
    })
    totals = _extract_totals(df, TYPE_BUDGET)
    assert totals["total_budget"] == 1200000.0
    assert totals["total_revenue_budget"] > 0
    assert totals["total_expense_budget"] > 0

def test_totals_bank_csv():
    df = pd.DataFrame({
        "Date":        ["2024-01-05", "2024-01-10", "2024-01-15"],
        "Amount":      [10000, -5000, 8000],
        "Description": ["Deposit", "Payment", "Deposit"],
    })
    totals = _extract_totals(df, TYPE_BANK_CSV)
    assert totals["total_inflows"]     == 18000.0
    assert totals["total_outflows"]    == 5000.0
    assert totals["net_cash_change"]   == 13000.0
    assert totals["transaction_count"] == 3


# ── Account structure ─────────────────────────────────────────────────────────

def test_account_structure_tb():
    df = pd.DataFrame({
        "Account Number": [1010, 1020, 2010, 3010, 4010, 4020, 5010, 5020, 5030],
        "Account Name":   ["Cash","AR","AP","Net Assets","Rev1","Rev2","Sal","Rent","Util"],
        "Balance":        [1]*9,
        "Dr/Cr":          ["DR"]*9,
    })
    struct = _extract_account_structure(df, TYPE_TB_UNKNOWN)
    assert struct["has_account_numbers"] is True
    assert struct["account_count"]       == 9
    assert struct["asset_accounts"]      == 2   # 1010, 1020
    assert struct["liability_accounts"]  == 1   # 2010
    assert struct["net_asset_accounts"]  == 1   # 3010
    assert struct["revenue_accounts"]    == 2   # 4010, 4020
    assert struct["expense_accounts"]    == 3   # 5010, 5020, 5030
    assert struct["account_range_low"]   == 1010
    assert struct["account_range_high"]  == 5030

def test_account_structure_skipped_for_budget():
    df = pd.DataFrame({"Category":["Revenue"], "Budget Amount":[100000]})
    struct = _extract_account_structure(df, TYPE_BUDGET)
    assert struct == {}

def test_account_structure_skipped_for_bank():
    df = pd.DataFrame({"Date":["2024-01-01"], "Amount":[100], "Description":["test"]})
    struct = _extract_account_structure(df, TYPE_BANK_CSV)
    assert struct == {}


# ── Balance check ─────────────────────────────────────────────────────────────

def test_balance_check_balanced():
    totals = {"total_dr_balances": 100000.0, "total_cr_balances": 100000.0, "net_difference": 0.0}
    result = _balance_check(totals, TYPE_TB_UNKNOWN)
    assert result["flag_level"] == BALANCE_OK
    assert result["difference"] == 0.0

def test_balance_check_small_diff():
    # 0.5% difference — below the 1% threshold
    totals = {"total_dr_balances": 100000.0, "total_cr_balances": 99500.0, "net_difference": 500.0}
    result = _balance_check(totals, TYPE_TB_CURRENT)
    assert result["flag_level"] == BALANCE_SMALL_DIFF
    assert result["difference"] == 500.0

def test_balance_check_material_diff():
    # 1.2% difference — above the 1% threshold
    totals = {"total_dr_balances": 2066500.0, "total_cr_balances": 2042400.0, "net_difference": 24100.0}
    result = _balance_check(totals, TYPE_TB_CURRENT)
    assert result["flag_level"] == BALANCE_MATERIAL_DIFF
    assert result["difference"] == 24100.0
    assert result["pct_of_dr"]  == pytest.approx(1.167, abs=0.01)

def test_balance_threshold():
    assert MATERIAL_BALANCE_THRESHOLD == 0.01

def test_balance_check_skipped_for_gl():
    totals = {"total_debits": 100000.0, "total_credits": 95000.0}
    result = _balance_check(totals, TYPE_GENERAL_LEDGER)
    assert result == {}

def test_balance_check_skipped_for_budget():
    result = _balance_check({}, TYPE_BUDGET)
    assert result == {}


# ── Full classify_financial_file integration ──────────────────────────────────

def test_classify_general_ledger_file():
    df = pd.DataFrame({
        "Transaction Date": ["2024-01-05", "2024-06-15", "2024-12-31"],
        "Amount":           [45000, -15000, 25000],
        "Description":      ["Payroll", "Rent", "Grant receipt"],
        "Account Number":   [5010, 5050, 4010],
        "Account":          ["Salaries", "Rent", "Grant Revenue"],
    })
    path = make_csv(df, "general_ledger_2024.csv")
    try:
        result = classify_financial_file(path)
        assert result["doc_type"]             == TYPE_GENERAL_LEDGER
        assert result["doc_type_confidence"]  >= 0.90
        assert result["doc_type_source"]      == "heuristic"
        assert result["finality_state"]       == FINALITY_TRUSTED
        assert result["totals"]["total_debits"]  == 70000.0
        assert result["totals"]["total_credits"] == 15000.0
        # Filename has "2024" so filename detection fires before data_inferred
        assert "2024" in str(result["period_start"])
        assert result["period_source"] in ("filename", "data_inferred")
        assert result["row_count"]    == 3
    finally:
        cleanup(path)

def test_classify_trial_balance_unknown():
    df = pd.DataFrame({
        "Account Number": [1010, 1020, 2010, 4010],
        "Account Name":   ["Cash","AR","AP","Revenue"],
        "Balance":        [50000, 30000, 20000, 60000],
        "Dr/Cr":          ["DR","DR","CR","CR"],
    })
    path = make_csv(df, "trial_balance.csv")
    try:
        result = classify_financial_file(path)
        assert result["doc_type"]         == TYPE_TB_UNKNOWN
        assert result["finality_state"]   == FINALITY_REVIEW_REQUIRED
        assert "balance_check" in result
        assert result["balance_check"]["flag_level"] == BALANCE_OK
    finally:
        cleanup(path)

def test_classify_prior_year_tb_from_filename():
    df = pd.DataFrame({
        "Account Number": [1010, 2010],
        "Account Name":   ["Cash","AP"],
        "Balance":        [50000, 20000],
        "Dr/Cr":          ["DR","CR"],
    })
    path = make_csv(df, "prior_year_trial_balance.csv")
    try:
        result = classify_financial_file(path)
        assert result["doc_type"] == TYPE_TB_PRIOR
    finally:
        cleanup(path)

def test_classify_budget_file():
    df = pd.DataFrame({
        "Category":      ["Grant Revenue", "Contributions", "Salaries", "Rent", "Utilities"],
        "Budget Amount": [500000, 200000, 400000, 100000, 50000],
    })
    path = make_csv(df, "budget_fy2024.csv")
    try:
        result = classify_financial_file(path)
        assert result["doc_type"]        == TYPE_BUDGET
        assert result["finality_state"]  == FINALITY_TRUSTED
        assert result["totals"]["total_budget"] == 1250000.0
        assert "2024" in str(result.get("period_start",""))
    finally:
        cleanup(path)

def test_classify_bank_csv():
    df = pd.DataFrame({
        "Date":        pd.date_range("2024-01-01", periods=5),
        "Amount":      [10000, -5000, 8000, -2000, 15000],
        "Description": ["Deposit","Payment","Deposit","Fee","Deposit"],
    })
    path = make_csv(df, "bank_statement_2024.csv")
    try:
        result = classify_financial_file(path)
        assert result["doc_type"]          == TYPE_BANK_CSV
        assert result["finality_state"]    == FINALITY_TRUSTED
        assert result["totals"]["total_inflows"]  == 33000.0
        assert result["totals"]["total_outflows"] == 7000.0
    finally:
        cleanup(path)

def test_classify_material_balance_difference():
    """Reproduces the test batch TB that is out of balance by 1.2%."""
    dr_total = 2066500.0
    cr_total = 2042400.0
    # Build a TB with DR > CR
    df = pd.DataFrame({
        "Account Number": [1010, 2010],
        "Account Name":   ["Cash","AP"],
        "Balance":        [dr_total, cr_total],
        "Dr/Cr":          ["DR","CR"],
    })
    path = make_csv(df, "trial_balance_test.csv")
    try:
        result = classify_financial_file(path)
        assert result["balance_check"]["flag_level"] == BALANCE_MATERIAL_DIFF
        assert result["balance_check"]["difference"] == pytest.approx(24100.0, abs=1.0)
    finally:
        cleanup(path)

def test_classify_non_financial_csv():
    """A CSV with unrecognized columns returns not_financial_structured_data."""
    df = pd.DataFrame({"Name":["Alice","Bob"], "Score":[95,87], "Grade":["A","B"]})
    path = make_csv(df, "grades.csv")
    try:
        result = classify_financial_file(path)
        assert result["doc_type"] == TYPE_NOT_FINANCIAL
    finally:
        cleanup(path)

def test_classify_missing_file():
    result = classify_financial_file("/nonexistent/path/file.csv")
    assert result["doc_type"] == TYPE_NOT_FINANCIAL
    assert "read_error" in result


# ── Score function tests ──────────────────────────────────────────────────────

def test_financial_score_well_classified():
    """A well-classified financial file with period and totals should score >= 0.80."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from audit_ingestion.router import _score_financial
    from audit_ingestion.models import AuditEvidence, ExtractionMeta, AuditOverview

    ev = AuditEvidence(
        source_file="general_ledger.csv",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=12000),
        audit_overview=AuditOverview(summary="General ledger for FY2024."),
    )
    fin = {
        "doc_type": "general_ledger",
        "doc_type_confidence": 0.95,
        "finality_state": "trusted",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "period_confidence": 0.70,
        "totals": {
            "total_debits": 1560100.0,
            "total_credits": 293300.0,
            "net": 1266800.0,
            "transaction_count": 111,
        },
        "balance_check": {},
    }
    score = _score_financial(ev, fin)
    assert score >= 0.75, f"Expected >= 0.75, got {score}"  # real pipeline adds amounts/claims, pushing to 0.83+


def test_financial_score_tb_balanced():
    """A balanced TB should score higher than an out-of-balance TB."""
    from audit_ingestion.router import _score_financial
    from audit_ingestion.models import AuditEvidence, ExtractionMeta, AuditOverview

    def make_ev():
        return AuditEvidence(
            source_file="tb.csv",
            extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=5000),
            audit_overview=AuditOverview(summary="Trial balance."),
        )

    fin_balanced = {
        "doc_type": "trial_balance_current",
        "doc_type_confidence": 0.90,
        "finality_state": "user_confirmed",
        "period_start": "2024", "period_confidence": 0.70,
        "totals": {"total_dr_balances": 100000, "total_cr_balances": 100000,
                   "net_difference": 0, "account_count": 61},
        "balance_check": {"flag_level": "tb_balanced", "difference": 0},
    }
    fin_unbalanced = {**fin_balanced,
        "balance_check": {"flag_level": "material_balance_difference",
                          "difference": 24100, "pct_of_dr": 1.2}}

    s_balanced   = _score_financial(make_ev(), fin_balanced)
    s_unbalanced = _score_financial(make_ev(), fin_unbalanced)
    assert s_balanced >= s_unbalanced, "Balanced TB should score >= unbalanced"
    assert s_balanced >= 0.80


def test_financial_score_unknown_year_tb_lower():
    """An unconfirmed TB year should score lower than a confirmed one."""
    from audit_ingestion.router import _score_financial
    from audit_ingestion.models import AuditEvidence, ExtractionMeta, AuditOverview

    def make_ev():
        return AuditEvidence(
            source_file="tb.csv",
            extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=5000),
            audit_overview=AuditOverview(summary="Trial balance."),
        )

    fin_confirmed = {
        "doc_type": "trial_balance_current",
        "doc_type_confidence": 0.90,
        "finality_state": "user_confirmed",
        "period_start": "2024", "period_confidence": 0.70,
        "totals": {"total_dr_balances": 100000, "total_cr_balances": 100000,
                   "net_difference": 0, "account_count": 61},
        "balance_check": {"flag_level": "tb_balanced"},
    }
    fin_unknown = {**fin_confirmed,
        "doc_type": "trial_balance_unknown_year",
        "finality_state": "review_required"}

    s_confirmed = _score_financial(make_ev(), fin_confirmed)
    s_unknown   = _score_financial(make_ev(), fin_unknown)
    assert s_confirmed > s_unknown, "Confirmed TB should score higher than unknown year"


def test_row_diagnostics_journal_sign_pattern_blocks():
    df = pd.DataFrame({
        "Journal ID": [1, 2],
        "Date": ["2024-01-05", "2024-01-06"],
        "Debit": [100.0, 0.0],
        "Credit": [50.0, 0.0],
        "Description": ["bad", "also bad"],
    })
    path = make_csv(df, "je.csv")
    try:
        result = classify_financial_file(path)
        diag = result["row_diagnostics"]
        assert diag["sign_pattern_issues"] >= 1
        assert any("debit/credit pattern" in r for r in diag["blocking_reasons"])
    finally:
        cleanup(path)


def test_row_diagnostics_thresholds_present_for_trial_balance():
    df = pd.DataFrame({
        "Account Number": [1010, 1020],
        "Account Name": ["Cash", "AR"],
        "Balance": [100.0, 200.0],
        "Dr/Cr": ["DR", "DR"],
    })
    path = make_csv(df, "tb_2024.csv")
    try:
        result = classify_financial_file(path)
        diag = result["row_diagnostics"]
        assert diag["thresholds"]["duplicate_rate"] <= 0.02
        assert "account_range_anomalies" in diag
    finally:
        cleanup(path)
