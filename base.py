"""
audit_ingestion_v04.2/audit_ingestion/normalizers.py
Deterministic normalization for cross-document matching.

All normalization happens here — never scattered across other modules.
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import Optional
from .models import AuditEvidence, LinkKeys, Party, Amount, DateItem, Identifier


# ── Name Normalization ────────────────────────────────────────────────────────

_STRIP_PATTERNS = re.compile(r"[^\w\s]")
_MULTI_SPACE    = re.compile(r"\s+")

# Common suffixes to normalize
_ENTITY_SUFFIXES = [
    r"\bINC\.?\b", r"\bLLC\.?\b", r"\bLLP\.?\b", r"\bCORP\.?\b",
    r"\bCO\.?\b",  r"\bLTD\.?\b", r"\bL\.P\.?\b", r"\bP\.C\.?\b",
    r"\bD/B/A\b",  r"\bDBA\b",
]

def normalize_party_name(name: str) -> str:
    """Normalize a party name for matching. Returns UPPERCASE clean string."""
    if not name:
        return ""
    result = name.upper().strip()
    result = _STRIP_PATTERNS.sub(" ", result)
    result = _MULTI_SPACE.sub(" ", result).strip()
    return result


# ── Date Normalization ────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y",
    "%B %d, %Y", "%b %d, %Y", "%d-%b-%Y",
    "%Y/%m/%d", "%d/%m/%Y", "%B %Y", "%b %Y",
]

def normalize_date(value: str) -> Optional[str]:
    """Parse and normalize a date string to YYYY-MM-DD. Returns None if unparseable."""
    if not value:
        return None
    value = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            if 1990 <= dt.year <= 2040:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ── Amount Normalization ──────────────────────────────────────────────────────

_AMOUNT_CLEAN = re.compile(r"[\$,\s()]")

def normalize_amount(value) -> Optional[float]:
    """Parse and normalize a monetary amount to float. Returns None if unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    try:
        cleaned = _AMOUNT_CLEAN.sub("", str(value))
        if cleaned.startswith("-") or cleaned.endswith("-"):
            cleaned = "-" + cleaned.strip("-")
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ── Identifier Normalization ──────────────────────────────────────────────────

_ID_CLEAN = re.compile(r"[\s\-_]")

def normalize_identifier(value: str) -> str:
    """Normalize an identifier string — uppercase, no spaces/hyphens."""
    if not value:
        return ""
    return _ID_CLEAN.sub("", str(value).upper().strip())


# ── Deduplication ─────────────────────────────────────────────────────────────

def dedupe_parties(parties: list[Party]) -> list[Party]:
    """Remove duplicate parties by normalized name."""
    seen: set[str] = set()
    result: list[Party] = []
    for p in parties:
        key = f"{p.role}:{p.normalized}"
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def dedupe_amounts(amounts: list[Amount]) -> list[Amount]:
    """Remove duplicate amounts by type+value."""
    seen: set[str] = set()
    result: list[Amount] = []
    for a in amounts:
        key = f"{a.type}:{a.value:.2f}"
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result


def dedupe_dates(dates: list[DateItem]) -> list[DateItem]:
    """Remove duplicate dates by type+value."""
    seen: set[str] = set()
    result: list[DateItem] = []
    for d in dates:
        key = f"{d.type}:{d.value}"
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def dedupe_identifiers(identifiers: list[Identifier]) -> list[Identifier]:
    """Remove duplicate identifiers by type+value."""
    seen: set[str] = set()
    result: list[Identifier] = []
    for i in identifiers:
        key = f"{i.type}:{normalize_identifier(i.value)}"
        if key not in seen:
            seen.add(key)
            result.append(i)
    return result


# ── Link Key Builder ──────────────────────────────────────────────────────────

def build_link_keys(evidence: AuditEvidence) -> LinkKeys:
    """
    Build normalized link keys from canonical evidence.
    These keys are used for cross-document matching.
    """
    party_names = sorted({
        normalize_party_name(p.name)
        for p in evidence.parties
        if p.name and normalize_party_name(p.name)
    })

    doc_numbers = sorted({
        normalize_identifier(i.value)
        for i in evidence.identifiers
        if i.type in ("document_number", "schedule_number", "contract_number",
                      "agreement_number", "reference_number")
        and i.value
    })

    agreement_numbers = sorted({
        normalize_identifier(i.value)
        for i in evidence.identifiers
        if i.type in ("agreement_number", "grant_number", "contract_number",
                      "lease_number", "po_number")
        and i.value
    })

    invoice_numbers = sorted({
        normalize_identifier(i.value)
        for i in evidence.identifiers
        if i.type in ("invoice_number", "receipt_number", "check_number",
                      "confirmation_number")
        and i.value
    })

    asset_descriptions = sorted({
        a.description.upper().strip()
        for a in evidence.assets
        if a.description
    })

    # Recurring amounts — amounts that are likely periodic payments
    recurring_keywords = {"monthly", "fixed_charge", "periodic", "recurring",
                          "installment", "rent", "lease_payment", "payment"}
    recurring_amounts = sorted({
        a.value
        for a in evidence.amounts
        if any(kw in a.type.lower() for kw in recurring_keywords)
        and a.value > 0
    })

    key_dates = sorted({
        normalize_date(d.value) or d.value
        for d in evidence.dates
        if d.value
    })

    # Other IDs from facts
    other_ids = sorted({
        normalize_identifier(str(f.value))
        for f in evidence.facts
        if f.label.endswith("_number") or f.label.endswith("_id")
        and f.value
    })

    return LinkKeys(
        party_names=party_names,
        document_numbers=doc_numbers,
        agreement_numbers=agreement_numbers,
        invoice_numbers=invoice_numbers,
        asset_descriptions=asset_descriptions,
        recurring_amounts=recurring_amounts,
        key_dates=key_dates,
        other_ids=other_ids,
    )


def normalize_evidence(evidence: AuditEvidence) -> AuditEvidence:
    """
    Run all normalization passes on a canonical AuditEvidence object.
    - Normalize party names
    - Normalize dates
    - Normalize amounts
    - Deduplicate all collections
    - Build link keys
    Returns a new normalized AuditEvidence.
    """
    # Normalize parties
    normalized_parties = []
    for p in evidence.parties:
        normalized_parties.append(p.model_copy(update={
            "normalized": normalize_party_name(p.name)
        }))
    normalized_parties = dedupe_parties(normalized_parties)

    # Normalize amounts
    normalized_amounts = []
    for a in evidence.amounts:
        v = normalize_amount(a.value)
        if v is not None:
            normalized_amounts.append(a.model_copy(update={"value": v}))
    normalized_amounts = dedupe_amounts(normalized_amounts)

    # Normalize dates
    normalized_dates = []
    for d in evidence.dates:
        nd = normalize_date(d.value)
        normalized_dates.append(d.model_copy(update={"value": nd or d.value}))
    normalized_dates = dedupe_dates(normalized_dates)

    # Normalize identifiers
    normalized_identifiers = dedupe_identifiers(evidence.identifiers)

    updated = evidence.model_copy(update={
        "parties":     normalized_parties,
        "amounts":     normalized_amounts,
        "dates":       normalized_dates,
        "identifiers": normalized_identifiers,
    })

    # Build link keys from normalized evidence
    updated = updated.model_copy(update={
        "link_keys": build_link_keys(updated)
    })

    return updated
