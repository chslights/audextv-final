"""
audit_ingestion_v05/audit_ingestion/financial_classifier.py

Financial file classifier and structured data extractor.

Handles CSV and Excel files through a three-layer approach:
  Layer 1 — Deterministic column-header matching (free, instant)
  Layer 2 — AI fallback for non-standard headers (lightweight call)
  Layer 3 — User confirmation for TB ambiguity and low-confidence results

Extraction priorities (in order):
  1. Document type classification
  2. Period covered (with confidence scoring)
  3. Key totals (deterministic, computed from data)
  4. Account structure (TB and GL only)
  + Balance check with two-level flagging

Output goes into document_specific.financial_data on the AuditEvidence object.
The top-level schema is unchanged.
"""
from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FINANCIAL_CLASSIFIER_VERSION = "v05.0"

# Finality states
FINALITY_TRUSTED            = "trusted"
FINALITY_REVIEW_RECOMMENDED = "review_recommended"
FINALITY_REVIEW_REQUIRED    = "review_required"
FINALITY_USER_CONFIRMED     = "user_confirmed"

# Document type labels
TYPE_GENERAL_LEDGER          = "general_ledger"
TYPE_JOURNAL_ENTRY           = "journal_entry_listing"
TYPE_TB_UNKNOWN              = "trial_balance_unknown_year"
TYPE_TB_CURRENT              = "trial_balance_current"
TYPE_TB_PRIOR                = "trial_balance_prior_year"
TYPE_BUDGET                  = "budget"
TYPE_BANK_CSV                = "bank_statement_csv"
TYPE_CHART_OF_ACCOUNTS       = "chart_of_accounts"
TYPE_NOT_FINANCIAL           = "not_financial_structured_data"

# Balance check flag levels
BALANCE_OK                   = "tb_balanced"
BALANCE_SMALL_DIFF           = "balance_difference_detected"
BALANCE_MATERIAL_DIFF        = "material_balance_difference"
MATERIAL_BALANCE_THRESHOLD   = 0.01   # 1% of total DR


# ── Column normalization ──────────────────────────────────────────────────────

def _norm(header: str) -> str:
    """Normalize a column header for matching: lowercase, strip spaces/punct."""
    return re.sub(r'[^a-z0-9]', '', str(header).lower())


def _norm_headers(df: pd.DataFrame) -> set[str]:
    """Return normalized set of all column headers."""
    return {_norm(c) for c in df.columns}


# ── Layer 1: Deterministic column matching ────────────────────────────────────

# Each signature: (required_columns, optional_columns, type, confidence)
# Required: ALL must be present (normalized)
# Optional: presence boosts confidence but not required
_SIGNATURES = [
    # General Ledger — has transaction date, amount, description, account
    ({"transactiondate", "amount", "description"}, {"account", "accountnumber", "accountno"}, TYPE_GENERAL_LEDGER, 0.95),
    # Journal Entry — has journal/entry ID, debit AND credit columns
    ({"debit", "credit"}, {"journalid", "entryid", "entrynumber", "jedate", "reference"}, TYPE_JOURNAL_ENTRY, 0.90),
    # Trial Balance — account number, name, balance, Dr/Cr indicator
    ({"accountnumber", "balance"}, {"accountname", "drcr", "normalbalance", "drcrind"}, TYPE_TB_UNKNOWN, 0.90),
    ({"accountno", "balance"}, {"accountname", "drcr"}, TYPE_TB_UNKNOWN, 0.88),
    ({"acct", "balance"}, {"name", "drcr"}, TYPE_TB_UNKNOWN, 0.82),
    # Budget — category/account + budget amount, no date column, no dr/cr
    ({"budgetamount"}, {"category", "account", "description"}, TYPE_BUDGET, 0.92),
    ({"budget"}, {"category", "account"}, TYPE_BUDGET, 0.85),
    # Bank statement CSV — date, amount, description, NO account number
    ({"date", "amount", "description"}, set(), TYPE_BANK_CSV, 0.88),
    # Chart of accounts — account number, name, type — no balance
    ({"accountnumber", "accounttype"}, {"accountname", "accountclass"}, TYPE_CHART_OF_ACCOUNTS, 0.95),
    ({"accountno", "accounttype"}, {"accountname"}, TYPE_CHART_OF_ACCOUNTS, 0.90),
]

# Columns that disqualify certain types
_DISQUALIFIERS = {
    TYPE_BANK_CSV:          {"accountnumber", "accountno", "acctno"},   # bank CSV has no account number
    TYPE_CHART_OF_ACCOUNTS: {"balance", "amount", "debit", "credit"},   # CoA has no financial amounts
    TYPE_BUDGET:            {"transactiondate", "date", "drcr"},        # budget has no date or dr/cr
}


def _layer1_classify(df: pd.DataFrame) -> tuple[str, float]:
    """
    Deterministic column-header matching.
    Returns (doc_type, confidence) or (TYPE_NOT_FINANCIAL, 0.0).
    """
    norm = _norm_headers(df)

    best_type = TYPE_NOT_FINANCIAL
    best_conf = 0.0

    for required, optional, doc_type, base_conf in _SIGNATURES:
        # All required columns must be present
        if not required.issubset(norm):
            continue

        # Check disqualifiers
        disq = _DISQUALIFIERS.get(doc_type, set())
        if disq and disq.intersection(norm):
            continue

        # Boost confidence for optional columns present
        optional_found = len(optional.intersection(norm))
        optional_boost = optional_found * 0.01
        conf = min(1.0, base_conf + optional_boost)

        if conf > best_conf:
            best_conf = conf
            best_type = doc_type

    return best_type, best_conf


# ── TB year resolution ────────────────────────────────────────────────────────

_PRIOR_FILENAME_SIGNALS = ["prior", "py", "lastyear", "previousyear", "prev", "ly"]

def _resolve_tb_year(doc_type: str, filepath: str) -> tuple[str, str]:
    """
    Attempt to resolve trial_balance_unknown_year to current or prior
    based on filename signals. Returns (resolved_type, period_source).
    Only strong signals trigger resolution — default stays unknown.
    """
    if doc_type != TYPE_TB_UNKNOWN:
        return doc_type, ""

    fname_lower = _norm(Path(filepath).stem)

    # Strong prior year signals
    for signal in _PRIOR_FILENAME_SIGNALS:
        if signal in fname_lower:
            return TYPE_TB_PRIOR, "filename"

    # Year number in filename — if it looks like a prior year (e.g. 2023 in a 2024 context)
    # We can't know the engagement year here, so leave as unknown for the UI to resolve
    return TYPE_TB_UNKNOWN, ""


# ── Layer 2: AI fallback ──────────────────────────────────────────────────────

_AI_CLASSIFICATION_PROMPT = """You are classifying a structured financial file for an audit pipeline.

Column headers: {headers}
First 10 rows (as CSV):
{sample_rows}
Filename: {filename}

Classify this file as exactly one of:
- general_ledger
- journal_entry_listing
- trial_balance_unknown_year
- trial_balance_current
- trial_balance_prior_year
- budget
- bank_statement_csv
- chart_of_accounts
- not_financial_structured_data

Return JSON only:
{{"doc_type": "...", "confidence": 0.0, "reason": "one short phrase"}}"""


def _layer2_classify(df: pd.DataFrame, filepath: str, provider) -> tuple[str, float, str]:
    """
    AI fallback classification. Only called when Layer 1 finds no match.
    Returns (doc_type, confidence, reason).
    """
    if provider is None:
        return TYPE_NOT_FINANCIAL, 0.0, "no provider"

    try:
        headers = list(df.columns)
        sample  = df.head(10).to_csv(index=False)
        fname   = Path(filepath).name

        prompt = _AI_CLASSIFICATION_PROMPT.format(
            headers=headers,
            sample_rows=sample[:2000],
            filename=fname,
        )

        schema = {
            "name":   "financial_classification",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "doc_type":   {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason":     {"type": "string"},
                }
            }
        }

        result = provider.extract_structured(
            system="You classify financial files for an audit pipeline. Return only valid JSON.",
            user=prompt,
            json_schema=schema,
            max_tokens=200,
        )

        doc_type = result.get("doc_type", TYPE_NOT_FINANCIAL)
        conf     = float(result.get("confidence", 0.5))
        reason   = result.get("reason", "")

        # Validate the returned type
        valid_types = {
            TYPE_GENERAL_LEDGER, TYPE_JOURNAL_ENTRY, TYPE_TB_UNKNOWN,
            TYPE_TB_CURRENT, TYPE_TB_PRIOR, TYPE_BUDGET,
            TYPE_BANK_CSV, TYPE_CHART_OF_ACCOUNTS, TYPE_NOT_FINANCIAL,
        }
        if doc_type not in valid_types:
            doc_type = TYPE_NOT_FINANCIAL
            conf = 0.0

        return doc_type, conf, reason

    except Exception as e:
        logger.warning(f"Layer 2 AI classification failed: {e}")
        return TYPE_NOT_FINANCIAL, 0.0, f"ai_error: {e}"


# ── Finality state ────────────────────────────────────────────────────────────

def _get_finality(doc_type: str, confidence: float, source: str) -> str:
    """Compute finality state from type, confidence, and source."""
    if doc_type == TYPE_TB_UNKNOWN:
        return FINALITY_REVIEW_REQUIRED
    if confidence < 0.60:
        return FINALITY_REVIEW_REQUIRED
    if source == "ai" and confidence < 0.80:
        return FINALITY_REVIEW_RECOMMENDED
    if confidence < 0.80:
        return FINALITY_REVIEW_RECOMMENDED
    return FINALITY_TRUSTED


# ── Period detection ──────────────────────────────────────────────────────────

_PERIOD_HEADER_PATTERNS = [
    r'for\s+the\s+(?:fiscal\s+)?year\s+ended?\s+(.+)',
    r'year\s+ended?\s+(.+)',
    r'period\s+ended?\s+(.+)',
    r'as\s+of\s+(.+)',
    r'through\s+(.+)',
    r'fiscal\s+year\s+(\d{4})',
    r'fy\s*(\d{4})',
]

_YEAR_IN_FILENAME = re.compile(r'(20\d{2}|19\d{2}|fy\d{2,4}|ye\d{4})', re.IGNORECASE)


def _detect_period(df: pd.DataFrame, filepath: str, doc_type: str) -> dict:
    """
    Detect fiscal period from three sources ranked by reliability.
    Returns dict with period_start, period_end, period_confidence, period_source.
    """
    fname = Path(filepath).name

    # Source 1: look for period header in first few rows (non-data header rows)
    # Some CSV exports include a header like "For the Year Ended December 31, 2024"
    try:
        raw = pd.read_csv(filepath, header=None, nrows=5)
        for _, row in raw.iterrows():
            for cell in row:
                cell_str = str(cell).strip()
                for pat in _PERIOD_HEADER_PATTERNS:
                    m = re.search(pat, cell_str, re.IGNORECASE)
                    if m:
                        period_text = m.group(1).strip()
                        return {
                            "period_start": period_text,
                            "period_end":   period_text,
                            "period_confidence": 0.95,
                            "period_source": "header",
                        }
    except Exception:
        pass

    # Source 2: filename signal
    year_match = _YEAR_IN_FILENAME.search(fname)
    if year_match:
        year_str = year_match.group(1).upper()
        # Normalize FY24 → FY2024
        if re.match(r'FY\d{2}$', year_str):
            year_str = "FY20" + year_str[2:]
        return {
            "period_start": year_str,
            "period_end":   year_str,
            "period_confidence": 0.70,
            "period_source": "filename",
        }

    # Source 3: infer from date column in data
    date_cols = [c for c in df.columns if _norm(c) in {"date", "transactiondate", "txndate", "postdate", "period"}]
    if date_cols:
        try:
            dates = pd.to_datetime(df[date_cols[0]], errors="coerce").dropna()
            if len(dates) > 0:
                return {
                    "period_start": str(dates.min().date()),
                    "period_end":   str(dates.max().date()),
                    "period_confidence": 0.50,
                    "period_source": "data_inferred",
                }
        except Exception:
            pass

    return {
        "period_start":      None,
        "period_end":        None,
        "period_confidence": 0.0,
        "period_source":     "not_found",
    }


# ── Key totals ────────────────────────────────────────────────────────────────

def _extract_totals(df: pd.DataFrame, doc_type: str) -> dict:
    """Compute key totals deterministically from the data."""
    totals = {}
    norm_cols = {_norm(c): c for c in df.columns}

    try:
        if doc_type == TYPE_GENERAL_LEDGER:
            amt_col = norm_cols.get("amount") or norm_cols.get("amt")
            if amt_col:
                amounts = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)
                totals["total_debits"]        = round(float(amounts[amounts > 0].sum()), 2)
                totals["total_credits"]       = round(float(abs(amounts[amounts < 0].sum())), 2)
                totals["net"]                 = round(float(amounts.sum()), 2)
                totals["transaction_count"]   = int(len(df))

        elif doc_type in (TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR):
            bal_col  = norm_cols.get("balance") or norm_cols.get("bal") or norm_cols.get("endingbalance")
            drcr_col = norm_cols.get("drcr") or norm_cols.get("normalbalance") or norm_cols.get("drcrind")
            if bal_col:
                balances = pd.to_numeric(df[bal_col], errors="coerce").fillna(0)
                if drcr_col:
                    dr_mask = df[drcr_col].astype(str).str.upper().str.strip().isin(["DR", "D", "DEBIT"])
                    cr_mask = df[drcr_col].astype(str).str.upper().str.strip().isin(["CR", "C", "CREDIT"])
                    totals["total_dr_balances"] = round(float(balances[dr_mask].sum()), 2)
                    totals["total_cr_balances"] = round(float(balances[cr_mask].sum()), 2)
                    totals["net_difference"]    = round(float(totals["total_dr_balances"] - totals["total_cr_balances"]), 2)
                else:
                    totals["total_balances"]    = round(float(balances.sum()), 2)
                totals["account_count"]         = int(len(df))

        elif doc_type == TYPE_JOURNAL_ENTRY:
            dr_col = norm_cols.get("debit") or norm_cols.get("dr")
            cr_col = norm_cols.get("credit") or norm_cols.get("cr")
            if dr_col and cr_col:
                debits  = pd.to_numeric(df[dr_col], errors="coerce").fillna(0)
                credits = pd.to_numeric(df[cr_col], errors="coerce").fillna(0)
                totals["total_debits"]      = round(float(debits.sum()), 2)
                totals["total_credits"]     = round(float(credits.sum()), 2)
                totals["net"]               = round(float(debits.sum() - credits.sum()), 2)
                totals["entry_count"]       = int(len(df))

        elif doc_type == TYPE_BUDGET:
            amt_col = norm_cols.get("budgetamount") or norm_cols.get("budget") or norm_cols.get("amount")
            cat_col = norm_cols.get("category") or norm_cols.get("account") or norm_cols.get("description")
            if amt_col:
                amounts = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)
                totals["total_budget"] = round(float(amounts.sum()), 2)
                # Split into revenue vs expense by sign or category keywords
                if cat_col:
                    rev_keywords = ["revenue", "contribution", "grant", "income", "gift", "donation", "support"]
                    rev_mask = df[cat_col].astype(str).str.lower().str.contains("|".join(rev_keywords), na=False)
                    totals["total_revenue_budget"]  = round(float(amounts[rev_mask].sum()), 2)
                    totals["total_expense_budget"]  = round(float(amounts[~rev_mask].sum()), 2)
                    totals["projected_surplus_deficit"] = round(
                        float(totals["total_revenue_budget"] - totals["total_expense_budget"]), 2)
                totals["category_count"] = int(len(df))

        elif doc_type == TYPE_BANK_CSV:
            amt_col = norm_cols.get("amount") or norm_cols.get("amt")
            if amt_col:
                amounts = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)
                totals["total_inflows"]      = round(float(amounts[amounts > 0].sum()), 2)
                totals["total_outflows"]     = round(float(abs(amounts[amounts < 0].sum())), 2)
                totals["net_cash_change"]    = round(float(amounts.sum()), 2)
                totals["transaction_count"]  = int(len(df))

    except Exception as e:
        logger.warning(f"Totals extraction failed: {e}")
        totals["totals_error"] = str(e)

    return totals


# ── Account structure ─────────────────────────────────────────────────────────

def _extract_account_structure(df: pd.DataFrame, doc_type: str) -> dict:
    """Extract account structure for TB and GL files. Returns empty dict for others."""
    if doc_type not in (TYPE_GENERAL_LEDGER, TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR):
        return {}

    norm_cols = {_norm(c): c for c in df.columns}
    acct_col  = (norm_cols.get("accountnumber") or norm_cols.get("accountno")
                 or norm_cols.get("acct") or norm_cols.get("account"))
    if not acct_col:
        return {"has_account_numbers": False}

    try:
        acct_nums = pd.to_numeric(df[acct_col], errors="coerce").dropna().astype(int)
        if len(acct_nums) == 0:
            return {"has_account_numbers": False}

        structure = {
            "has_account_numbers": True,
            "account_count":       int(len(acct_nums)),
            "account_range_low":   int(acct_nums.min()),
            "account_range_high":  int(acct_nums.max()),
            "asset_accounts":      int(((acct_nums >= 1000) & (acct_nums < 2000)).sum()),
            "liability_accounts":  int(((acct_nums >= 2000) & (acct_nums < 3000)).sum()),
            "net_asset_accounts":  int(((acct_nums >= 3000) & (acct_nums < 4000)).sum()),
            "revenue_accounts":    int(((acct_nums >= 4000) & (acct_nums < 5000)).sum()),
            "expense_accounts":    int((acct_nums >= 5000).sum()),
        }
        return structure

    except Exception as e:
        logger.warning(f"Account structure extraction failed: {e}")
        return {"structure_error": str(e)}


# ── Balance check ─────────────────────────────────────────────────────────────

def _balance_check(totals: dict, doc_type: str) -> dict:
    """
    Compute balance check for trial balance files.
    Returns dict with dr_total, cr_total, difference, flag_level.
    """
    if doc_type not in (TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR):
        return {}

    dr    = totals.get("total_dr_balances")
    cr    = totals.get("total_cr_balances")

    if dr is None or cr is None:
        return {"flag_level": "insufficient_data"}

    diff  = round(abs(dr - cr), 2)
    pct   = diff / dr if dr > 0 else 0

    if diff == 0:
        flag = BALANCE_OK
    elif pct >= MATERIAL_BALANCE_THRESHOLD:
        flag = BALANCE_MATERIAL_DIFF
    else:
        flag = BALANCE_SMALL_DIFF

    return {
        "dr_total":    dr,
        "cr_total":    cr,
        "difference":  diff,
        "pct_of_dr":   round(pct * 100, 2),
        "flag_level":  flag,
    }


# ── Row retention, column mapping, and diagnostics ───────────────────────────

_ROW_PREVIEW_LIMIT = 2500
_POPULATION_DUPLICATE_RATE_THRESHOLD = 0.10
_POPULATION_FLAGGED_RATE_THRESHOLD = 0.20
POPULATION_THRESHOLDS = {
    # Flagged rate thresholds calibrated to real nonprofit financial data.
    # GL and bank routinely have outlier-flagged rows (large grants, payroll)
    # that are legitimate — threshold raised to avoid false population blocks.
    TYPE_GENERAL_LEDGER: {"duplicate_rate": 0.08, "flagged_rate": 0.45},
    TYPE_JOURNAL_ENTRY: {"duplicate_rate": 0.05, "flagged_rate": 0.18},
    TYPE_BANK_CSV: {"duplicate_rate": 0.03, "flagged_rate": 0.25},
    TYPE_TB_UNKNOWN: {"duplicate_rate": 0.02, "flagged_rate": 0.15},
    TYPE_TB_CURRENT: {"duplicate_rate": 0.02, "flagged_rate": 0.15},
    TYPE_TB_PRIOR: {"duplicate_rate": 0.02, "flagged_rate": 0.15},
    TYPE_BUDGET: {"duplicate_rate": 0.10, "flagged_rate": 0.25},
    TYPE_CHART_OF_ACCOUNTS: {"duplicate_rate": 0.05, "flagged_rate": 0.15},
}


def _find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first real DataFrame column matching one of the normalized names."""
    norm_cols = {_norm(c): c for c in df.columns}
    for cand in candidates:
        col = norm_cols.get(cand)
        if col:
            return col
    return None


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _serialize_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, 'isoformat'):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        return value.item()
    except Exception:
        return str(value)


def _column_mapping(df: pd.DataFrame, doc_type: str) -> dict:
    """Map source columns to canonical financial field names."""
    mapping: dict[str, dict] = {}

    def add(canonical: str, *candidates: str):
        source = _find_column(df, *candidates)
        if source:
            mapping[canonical] = {
                "source_column": source,
                "normalized_source": _norm(source),
                "confidence": 0.95,
            }

    if doc_type == TYPE_GENERAL_LEDGER:
        add("transaction_date", "transactiondate", "date", "txndate", "postdate")
        add("amount", "amount", "amt", "netamount")
        add("description", "description", "memo", "details", "narrative")
        add("account_number", "accountnumber", "accountno", "acct", "account")
        add("account_name", "accountname", "name")
        add("journal_id", "journalid", "entryid", "entrynumber")
        add("reference", "reference", "ref", "documentnumber", "docnumber")
    elif doc_type in (TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR):
        add("account_number", "accountnumber", "accountno", "acct", "account")
        add("account_name", "accountname", "name", "description")
        add("balance", "balance", "bal", "endingbalance")
        add("dr_cr", "drcr", "normalbalance", "drcrind")
    elif doc_type == TYPE_JOURNAL_ENTRY:
        add("entry_id", "entryid", "entrynumber", "journalid")
        add("entry_date", "jedate", "date", "transactiondate")
        add("debit", "debit", "dr")
        add("credit", "credit", "cr")
        add("description", "description", "memo", "details")
        add("account_number", "accountnumber", "accountno", "acct", "account")
        add("reference", "reference", "ref", "documentnumber")
    elif doc_type == TYPE_BUDGET:
        add("category", "category", "account", "description", "name")
        add("budget_amount", "budgetamount", "budget", "amount")
    elif doc_type == TYPE_BANK_CSV:
        add("transaction_date", "date", "transactiondate", "postdate")
        add("amount", "amount", "amt")
        add("description", "description", "memo", "details")
        add("reference", "reference", "ref", "checknumber", "documentnumber")
        add("account_number", "accountnumber", "accountno", "acct")
    elif doc_type == TYPE_CHART_OF_ACCOUNTS:
        add("account_number", "accountnumber", "accountno", "acct", "account")
        add("account_name", "accountname", "name", "description")
        add("account_type", "accounttype", "type", "accountclass")

    return mapping


_ACCOUNT_FAMILY_RANGES = [
    (1000, 1999, "assets"),
    (2000, 2999, "liabilities"),
    (3000, 3999, "net_assets"),
    (4000, 4999, "revenue"),
    (5000, 9999, "expenses"),
]


def _account_family(account_number) -> Optional[str]:
    """Derive account family from numeric account code."""
    try:
        n = int(float(str(account_number)))
        for lo, hi, family in _ACCOUNT_FAMILY_RANGES:
            if lo <= n <= hi:
                return family
        return "other"
    except (ValueError, TypeError):
        return None


_REVENUE_KEYWORDS = [
    "revenue", "contribution", "grant", "income", "gift", "donation",
    "support", "award", "fee income", "program service revenue",
]


def _revenue_expense_tag(category_text: str) -> str:
    """Tag a budget category as revenue or expense."""
    if not category_text:
        return "expense"
    lower = str(category_text).lower()
    return "revenue" if any(kw in lower for kw in _REVENUE_KEYWORDS) else "expense"


def _extract_rows(df: pd.DataFrame, doc_type: str, column_mapping: dict) -> list[dict]:
    """Retain row-level structured data for downstream use."""
    if doc_type == TYPE_NOT_FINANCIAL:
        return []

    rows: list[dict] = []
    subset = df.head(_ROW_PREVIEW_LIMIT).copy()

    is_tb   = doc_type in (TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR, TYPE_CHART_OF_ACCOUNTS)
    is_gl   = doc_type == TYPE_GENERAL_LEDGER
    is_bank = doc_type == TYPE_BANK_CSV
    is_budget = doc_type == TYPE_BUDGET

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        retained: dict = {"row_index": idx}
        non_empty = False
        for canonical, meta in column_mapping.items():
            source_col = meta.get("source_column")
            if source_col not in subset.columns:
                continue
            val = _serialize_value(row.get(source_col))
            retained[canonical] = val
            if val not in (None, ""):
                non_empty = True

        if not non_empty:
            continue

        # Derived fields — added after canonical mapping
        if (is_tb or is_gl) and retained.get("account_number") is not None:
            retained["account_family"] = _account_family(retained["account_number"])

        if is_bank and retained.get("amount") is not None:
            try:
                amt = float(retained["amount"])
                retained["inflow_outflow_tag"] = "inflow" if amt >= 0 else "outflow"
            except (ValueError, TypeError):
                retained["inflow_outflow_tag"] = None

        if is_budget:
            category = retained.get("category") or ""
            retained["revenue_expense_tag"] = _revenue_expense_tag(category)

        rows.append(retained)

    return rows


def _extract_top_flagged_rows(rows: list[dict], row_flags: list[dict], limit: int = 5) -> list[dict]:
    by_index = {r.get("row_index"): r for r in rows}
    preview = []
    for flag in row_flags[:limit]:
        item = {
            "row_index": flag.get("row_index"),
            "issues": flag.get("issues", []),
        }
        base = by_index.get(flag.get("row_index"), {})
        for key in ("account_number", "account_name", "transaction_date", "description", "amount", "balance", "debit", "credit", "budget_amount", "category"):
            if key in base and base.get(key) not in (None, ""):
                item[key] = base.get(key)
        preview.append(item)
    return preview


def _row_diagnostics(rows: list[dict], doc_type: str) -> dict:
    """Deterministic row-level diagnostics for structured financial populations."""
    if not rows:
        return {
            "row_count": 0,
            "blank_rows": 0,
            "duplicate_rows": 0,
            "malformed_rows": 0,
            "outlier_rows": 0,
            "weekend_rows": 0,
            "round_number_rows": 0,
            "date_gaps": 0,
            "account_range_anomalies": 0,
            "sign_pattern_issues": 0,
            "duplicate_journal_patterns": 0,
            "missing_key_field_rows": 0,
            "period_coverage_incomplete": False,
            "high_flagged_row_rate": False,
            "flagged_row_count": 0,
            "flagged_row_rate": 0.0,
            "population_ready": False,
            "blocking_reasons": ["no rows retained"],
            "row_flags": [],
            "top_flagged_rows_preview": [],
            "thresholds": POPULATION_THRESHOLDS.get(doc_type, {"duplicate_rate": _POPULATION_DUPLICATE_RATE_THRESHOLD, "flagged_rate": _POPULATION_FLAGGED_RATE_THRESHOLD}),
        }

    thresholds = POPULATION_THRESHOLDS.get(doc_type, {"duplicate_rate": _POPULATION_DUPLICATE_RATE_THRESHOLD, "flagged_rate": _POPULATION_FLAGGED_RATE_THRESHOLD})
    row_flags: list[dict] = []
    duplicate_rows = 0
    malformed_rows = 0
    outlier_rows = 0
    weekend_rows = 0
    round_number_rows = 0
    date_gaps = 0
    blank_rows = 0
    account_range_anomalies = 0
    sign_pattern_issues = 0
    duplicate_journal_patterns = 0
    missing_key_field_rows = 0
    period_coverage_incomplete = False

    frame = pd.DataFrame(rows)
    content_cols = [c for c in frame.columns if c != "row_index"]
    content_frame = frame[content_cols].copy() if content_cols else frame.copy()
    blank_mask = content_frame.replace("", pd.NA).isna().all(axis=1) if not content_frame.empty else pd.Series([], dtype=bool)
    blank_rows = int(blank_mask.sum()) if len(blank_mask) else 0

    if content_cols:
        dedupe_frame = content_frame.fillna("<NA>")
        dup_mask = dedupe_frame.duplicated(keep=False)
        duplicate_rows = int(dup_mask.sum())
    else:
        dup_mask = pd.Series([False] * len(frame))

    numeric_col = None
    for candidate in ("amount", "balance", "budget_amount", "debit", "credit"):
        if candidate in frame.columns:
            numeric_col = candidate
            break

    numeric_series = _to_numeric_series(frame[numeric_col]).fillna(0) if numeric_col else None
    nonzero_numeric = None
    if numeric_series is not None:
        nonzero_numeric = numeric_series[numeric_series != 0]
        if len(nonzero_numeric) >= 4:
            q1 = nonzero_numeric.quantile(0.25)
            q3 = nonzero_numeric.quantile(0.75)
            iqr = q3 - q1
            # Looser IQR multiplier for balance/budget files — large balances are expected
            # and naturally skewed. 1.5x IQR is too aggressive for TB/budget.
            # Transaction files (GL, bank) keep the standard 1.5x Tukey method.
            _iqr_mult = 3.0 if doc_type in (
                TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR,
                TYPE_BUDGET, TYPE_CHART_OF_ACCOUNTS,
            ) else 1.5
            if iqr > 0:
                lower = q1 - _iqr_mult * iqr
                upper = q3 + _iqr_mult * iqr
                outlier_mask = (numeric_series < lower) | (numeric_series > upper)
            else:
                mean = nonzero_numeric.mean()
                std = nonzero_numeric.std()
                _std_mult = 4.5 if doc_type in (
                    TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR,
                    TYPE_BUDGET, TYPE_CHART_OF_ACCOUNTS,
                ) else 3.0
                outlier_mask = abs(numeric_series - mean) > (_std_mult * std if std else float('inf'))
            outlier_rows = int(outlier_mask.sum())
        else:
            outlier_mask = pd.Series([False] * len(frame))

        # round_number_amount only meaningful on transaction files (GL, bank, JE)
        # Account balances on TB/CoA are legitimately round — do not flag them
        _is_balance_file = doc_type in (
            TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR,
            TYPE_CHART_OF_ACCOUNTS, TYPE_BUDGET,
        )
        if _is_balance_file:
            round_mask = pd.Series([False] * len(frame))
        else:
            # Only flag very round numbers: multiples of $10,000 above $50,000.
            # $45,000 payroll and $15,000 taxes are normal nonprofit transactions.
            round_mask = numeric_series.apply(
                lambda v: float(v).is_integer()
                          and abs(float(v)) >= 50000
                          and int(abs(float(v))) % 10000 == 0
            )
        round_number_rows = int(round_mask.sum())
    else:
        outlier_mask = pd.Series([False] * len(frame))
        round_mask = pd.Series([False] * len(frame))

    date_col = None
    for candidate in ("transaction_date", "entry_date"):
        if candidate in frame.columns:
            date_col = candidate
            break

    if date_col:
        date_series = _coerce_date_series(frame[date_col])
        # weekend_transaction only meaningful for journal entries where a
        # weekend posting could indicate a control weakness (e.g. unauthorized JE).
        # GL and bank transactions are routinely dated to weekends — not suspicious.
        if doc_type == TYPE_JOURNAL_ENTRY:
            weekend_mask = date_series.dt.dayofweek.isin([5, 6]).fillna(False)
            weekend_rows = int(weekend_mask.sum())
        else:
            weekend_mask = pd.Series([False] * len(frame))
        valid_dates = date_series.dropna().sort_values()
        if len(valid_dates) > 1:
            gaps = valid_dates.diff().dropna().dt.days
            date_gaps = int((gaps > 31).sum())
    else:
        weekend_mask = pd.Series([False] * len(frame))

    if doc_type in (TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR) and "account_number" in frame.columns:
        acct_num = _to_numeric_series(frame["account_number"])
        malformed_mask = acct_num.isna()
    elif doc_type == TYPE_GENERAL_LEDGER:
        malformed_mask = pd.Series([False] * len(frame))
        if "transaction_date" in frame.columns:
            malformed_mask = malformed_mask | _coerce_date_series(frame["transaction_date"]).isna()
        if numeric_col:
            malformed_mask = malformed_mask | _to_numeric_series(frame[numeric_col]).isna()
        if "description" in frame.columns:
            malformed_mask = malformed_mask | frame["description"].isna()
    else:
        malformed_mask = pd.Series([False] * len(frame))
        if numeric_col:
            malformed_mask = malformed_mask | _to_numeric_series(frame[numeric_col]).isna()

    if doc_type in (TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR, TYPE_GENERAL_LEDGER, TYPE_JOURNAL_ENTRY, TYPE_CHART_OF_ACCOUNTS) and "account_number" in frame.columns:
        acct_text = frame["account_number"].astype(str).str.replace(r"\D", "", regex=True)
        dominant_len = acct_text[acct_text != ""].str.len().mode()
        expected_len = int(dominant_len.iloc[0]) if not dominant_len.empty else None
        account_range_mask = acct_text.eq("")
        if expected_len:
            account_range_mask = account_range_mask | ((acct_text != "") & (acct_text.str.len() < max(2, expected_len - 2)))
        account_range_anomalies = int(account_range_mask.sum())
    else:
        account_range_mask = pd.Series([False] * len(frame))

    if doc_type == TYPE_JOURNAL_ENTRY:
        debit_series = _to_numeric_series(frame["debit"]).fillna(0) if "debit" in frame.columns else pd.Series([0] * len(frame))
        credit_series = _to_numeric_series(frame["credit"]).fillna(0) if "credit" in frame.columns else pd.Series([0] * len(frame))
        sign_pattern_mask = ((debit_series > 0) & (credit_series > 0)) | ((debit_series == 0) & (credit_series == 0))
        sign_pattern_issues = int(sign_pattern_mask.sum())
        dup_cols = [c for c in ["entry_date", "transaction_date", "reference", "description", "debit", "credit"] if c in frame.columns]
        if dup_cols:
            duplicate_journal_patterns = int(frame[dup_cols].fillna("<NA>").duplicated(keep=False).sum())
    else:
        sign_pattern_mask = pd.Series([False] * len(frame))

    key_fields = []
    if doc_type in (TYPE_GENERAL_LEDGER, TYPE_BANK_CSV):
        key_fields = [c for c in [date_col, numeric_col, "description"] if c]
    elif doc_type in (TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR, TYPE_CHART_OF_ACCOUNTS):
        key_fields = [c for c in ["account_number", "account_name"] if c in frame.columns]
    elif doc_type == TYPE_JOURNAL_ENTRY:
        key_fields = [c for c in ["entry_date", "debit", "credit"] if c in frame.columns]
    if key_fields:
        missing_key_field_mask = frame[key_fields].replace("", pd.NA).isna().any(axis=1)
        missing_key_field_rows = int(missing_key_field_mask.sum())
    else:
        missing_key_field_mask = pd.Series([False] * len(frame))

    if date_col:
        valid_dates2 = _coerce_date_series(frame[date_col]).dropna()
        if len(valid_dates2) >= 3:
            month_coverage = valid_dates2.dt.to_period("M").nunique()
            span_months = max(1, ((valid_dates2.max().year - valid_dates2.min().year) * 12) + valid_dates2.max().month - valid_dates2.min().month + 1)
            period_coverage_incomplete = month_coverage < max(1, span_months // 2)

    malformed_rows = int(malformed_mask.sum())

    flagged_count = 0
    for i, retained_row in enumerate(rows):
        issues = []
        if len(blank_mask) and bool(blank_mask.iloc[i]):
            issues.append("blank_row")
        if len(dup_mask) and bool(dup_mask.iloc[i]):
            issues.append("duplicate_row")
        if len(malformed_mask) and bool(malformed_mask.iloc[i]):
            issues.append("malformed_row")
        if len(outlier_mask) and bool(outlier_mask.iloc[i]):
            issues.append("outlier_amount")
        if len(round_mask) and bool(round_mask.iloc[i]):
            issues.append("round_number_amount")
        if len(weekend_mask) and bool(weekend_mask.iloc[i]):
            issues.append("weekend_transaction")
        if len(account_range_mask) and bool(account_range_mask.iloc[i]):
            issues.append("account_range_anomaly")
        if len(sign_pattern_mask) and bool(sign_pattern_mask.iloc[i]):
            issues.append("sign_pattern_issue")
        if len(missing_key_field_mask) and bool(missing_key_field_mask.iloc[i]):
            issues.append("missing_key_field")
        if issues:
            flagged_count += 1
            row_flags.append({"row_index": retained_row.get("row_index"), "issues": issues})

    row_count = len(rows)
    flagged_rate = round(flagged_count / row_count, 4) if row_count else 0.0
    duplicate_rate = duplicate_rows / row_count if row_count else 0.0

    blocking_reasons = []
    if malformed_rows > 0:
        blocking_reasons.append(f"{malformed_rows} malformed row(s)")
    if missing_key_field_rows > 0 and doc_type in (TYPE_TB_UNKNOWN, TYPE_TB_CURRENT, TYPE_TB_PRIOR, TYPE_GENERAL_LEDGER, TYPE_JOURNAL_ENTRY, TYPE_BANK_CSV):
        blocking_reasons.append(f"{missing_key_field_rows} row(s) missing key fields")
    if duplicate_rate >= thresholds["duplicate_rate"]:
        blocking_reasons.append(f"duplicate row rate {duplicate_rate:.1%} exceeds threshold")
    if flagged_rate >= thresholds["flagged_rate"]:
        blocking_reasons.append(f"flagged row rate {flagged_rate:.1%} exceeds threshold")
    if sign_pattern_issues > 0 and doc_type == TYPE_JOURNAL_ENTRY:
        blocking_reasons.append(f"{sign_pattern_issues} journal row(s) have debit/credit pattern issues")
    if period_coverage_incomplete and doc_type in (TYPE_GENERAL_LEDGER, TYPE_JOURNAL_ENTRY, TYPE_BANK_CSV):
        blocking_reasons.append("date coverage appears incomplete for the observed period")

    population_ready = len(blocking_reasons) == 0

    return {
        "row_count": row_count,
        "blank_rows": blank_rows,
        "duplicate_rows": duplicate_rows,
        "malformed_rows": malformed_rows,
        "outlier_rows": outlier_rows,
        "weekend_rows": weekend_rows,
        "round_number_rows": round_number_rows,
        "date_gaps": date_gaps,
        "account_range_anomalies": account_range_anomalies,
        "sign_pattern_issues": sign_pattern_issues,
        "duplicate_journal_patterns": duplicate_journal_patterns,
        "missing_key_field_rows": missing_key_field_rows,
        "period_coverage_incomplete": period_coverage_incomplete,
        "high_flagged_row_rate": flagged_rate >= thresholds["flagged_rate"],
        "flagged_row_count": flagged_count,
        "flagged_row_rate": flagged_rate,
        "population_ready": population_ready,
        "blocking_reasons": blocking_reasons,
        "row_flags": row_flags,
        "top_flagged_rows_preview": _extract_top_flagged_rows(rows, row_flags),
        "thresholds": thresholds,
    }


# ── Multi-sheet Excel handling ───────────────────────────────────────────────

def _load_best_excel_sheet(
    filepath: str,
    sheet_override: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    For multi-sheet Excel files, find the sheet most likely to contain
    structured financial data. Returns (DataFrame, sheet_info_dict).

    sheet_override: if set, use this sheet directly (user confirmed choice).

    Strategy:
    1. If sheet_override is set — use it directly
    2. If only one sheet — use it
    3. Try each sheet against Layer 1 column matching
    4. Pick the highest-confidence match
    5. Fall back to the first sheet if nothing matches
    """
    xl = pd.ExcelFile(filepath)
    sheet_names = xl.sheet_names
    sheet_info  = {
        "sheet_count": len(sheet_names),
        "all_sheets":  sheet_names,
    }

    # User confirmed a specific sheet — use it directly
    if sheet_override and sheet_override in sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_override, nrows=5000)
        sheet_info["sheet_name"]       = sheet_override
        sheet_info["sheet_source"]     = "user_override"
        sheet_info["sheet_confidence"] = 1.0
        return df, sheet_info

    if len(sheet_names) == 1:
        df = pd.read_excel(filepath, sheet_name=sheet_names[0], nrows=5000)
        sheet_info["sheet_name"] = sheet_names[0]
        return df, sheet_info

    # Try each sheet and score it
    best_df   = None
    best_conf = 0.0
    best_name = sheet_names[0]

    for name in sheet_names:
        # Skip obviously irrelevant sheet names
        skip_keywords = ["cover", "readme", "instructions", "notes", "template", "blank"]
        if any(kw in name.lower() for kw in skip_keywords):
            continue
        try:
            df_candidate = pd.read_excel(filepath, sheet_name=name, nrows=5000)
            if df_candidate.empty or len(df_candidate.columns) < 2:
                continue
            _, conf = _layer1_classify(df_candidate)
            if conf > best_conf:
                best_conf = conf
                best_df   = df_candidate
                best_name = name
        except Exception:
            continue

    if best_df is None:
        # Nothing matched — use first sheet
        best_df   = pd.read_excel(filepath, sheet_name=sheet_names[0], nrows=5000)
        best_name = sheet_names[0]

    sheet_info["sheet_name"]      = best_name
    sheet_info["sheet_confidence"] = round(best_conf, 3)
    return best_df, sheet_info


# ── Main entry point ──────────────────────────────────────────────────────────

def classify_financial_file(
    filepath: str,
    provider=None,
    type_override: Optional[str] = None,
    sheet_override: Optional[str] = None,
) -> dict:
    """
    Classify and extract a structured financial file (CSV or Excel).

    type_override: skips Layers 1+2, uses this type directly with user_confirmed
    finality. Totals/structure are still recomputed from actual data.

    Returns a dict for document_specific.financial_data with:
      doc_type, doc_type_confidence, doc_type_source, finality_state,
      period_*, totals, financial_structure, balance_check,
      column_headers, row_count, column_mapping, rows, row_diagnostics,
      top_flagged_rows_preview

    If the file is not a recognized financial type, returns
      {"doc_type": "not_financial_structured_data", "finality_state": "trusted"}
    """
    path = Path(filepath)
    result = {
        "classifier_version": FINANCIAL_CLASSIFIER_VERSION,
        "user_confirmed_type": False,
        "user_override_type":  None,
    }

    # Load the file
    try:
        if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            df, sheet_info = _load_best_excel_sheet(filepath, sheet_override=sheet_override)
            result["excel_sheet_name"]  = sheet_info.get("sheet_name")
            result["excel_sheet_count"] = sheet_info.get("sheet_count", 1)
            if sheet_info.get("sheet_count", 1) > 1:
                result["excel_all_sheets"] = sheet_info.get("all_sheets", [])
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(filepath, nrows=5000)
        else:
            return {**result, "doc_type": TYPE_NOT_FINANCIAL,
                    "doc_type_confidence": 0.0, "doc_type_source": "extension",
                    "finality_state": FINALITY_TRUSTED}
    except Exception as e:
        logger.warning(f"Financial classifier could not read {filepath}: {e}")
        return {**result, "doc_type": TYPE_NOT_FINANCIAL,
                "doc_type_confidence": 0.0, "doc_type_source": "read_error",
                "finality_state": FINALITY_TRUSTED,
                "read_error": str(e)}

    result["column_headers"] = list(df.columns)
    result["row_count"]      = len(df)

    # Apply user type override — skips Layers 1 and 2 entirely
    # Totals and structure are still recomputed from actual data for consistency
    if type_override:
        valid_types = {
            TYPE_GENERAL_LEDGER, TYPE_JOURNAL_ENTRY, TYPE_TB_UNKNOWN,
            TYPE_TB_CURRENT, TYPE_TB_PRIOR, TYPE_BUDGET,
            TYPE_BANK_CSV, TYPE_CHART_OF_ACCOUNTS,
        }
        if type_override in valid_types:
            result["doc_type"]            = type_override
            result["doc_type_confidence"] = 1.0
            result["doc_type_source"]     = "user_override"
            result["finality_state"]      = FINALITY_USER_CONFIRMED
            result["user_confirmed_type"] = True
            result["user_override_type"]  = type_override
            period = _detect_period(df, filepath, type_override)
            result.update(period)
            totals = _extract_totals(df, type_override)
            result["totals"] = totals
            structure = _extract_account_structure(df, type_override)
            if structure:
                result["financial_structure"] = structure
            bal = _balance_check(totals, type_override)
            if bal:
                result["balance_check"] = bal
            column_mapping = _column_mapping(df, type_override)
            result["column_mapping"] = column_mapping
            rows = _extract_rows(df, type_override, column_mapping)
            result["rows"] = rows
            row_diagnostics = _row_diagnostics(rows, type_override)
            result["row_diagnostics"] = row_diagnostics
            result["top_flagged_rows_preview"] = row_diagnostics.get("top_flagged_rows_preview", [])
            logger.info(f"Financial classifier: {Path(filepath).name} → {type_override} (user_override)")
            return result

    # Layer 1: deterministic column matching
    doc_type, confidence = _layer1_classify(df)
    source = "heuristic"

    # Layer 2: AI fallback if no match
    if doc_type == TYPE_NOT_FINANCIAL and provider is not None:
        doc_type, confidence, reason = _layer2_classify(df, filepath, provider)
        source = "ai"
        if doc_type != TYPE_NOT_FINANCIAL:
            result["ai_classification_reason"] = reason

    # TB year resolution from filename
    if doc_type == TYPE_TB_UNKNOWN:
        resolved, res_source = _resolve_tb_year(doc_type, filepath)
        if resolved != TYPE_TB_UNKNOWN:
            doc_type = resolved
            source   = f"heuristic+{res_source}"
            confidence = min(confidence, 0.82)  # filename resolution is weaker

    result["doc_type"]            = doc_type
    result["doc_type_confidence"] = round(confidence, 3)
    result["doc_type_source"]     = source
    result["finality_state"]      = _get_finality(doc_type, confidence, source)

    if doc_type == TYPE_NOT_FINANCIAL:
        return result

    # Period detection
    period = _detect_period(df, filepath, doc_type)
    result.update(period)

    # Key totals
    totals = _extract_totals(df, doc_type)
    result["totals"] = totals

    # Account structure (TB and GL only)
    structure = _extract_account_structure(df, doc_type)
    if structure:
        result["financial_structure"] = structure

    # Balance check (TB only)
    bal = _balance_check(totals, doc_type)
    if bal:
        result["balance_check"] = bal

    # Row retention, mappings, and diagnostics
    column_mapping = _column_mapping(df, doc_type)
    result["column_mapping"] = column_mapping
    rows = _extract_rows(df, doc_type, column_mapping)
    result["rows"] = rows
    row_diagnostics = _row_diagnostics(rows, doc_type)
    result["row_diagnostics"] = row_diagnostics
    result["top_flagged_rows_preview"] = row_diagnostics.get("top_flagged_rows_preview", [])

    logger.info(
        f"Financial classifier: {path.name} → {doc_type} "
        f"(conf={confidence:.2f}, finality={result['finality_state']})"
    )
    return result


def is_financial_file(filepath: str) -> bool:
    """Quick gate: is this file a candidate for financial classification?
    .txt excluded — narrative documents (board minutes, memos) should not
    go through the financial classifier."""
    return Path(filepath).suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xls")
