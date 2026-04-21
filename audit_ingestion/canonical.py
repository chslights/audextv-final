"""
audit_ingestion_v04.2/audit_ingestion/canonical.py
Single AI canonical extraction pass.

Uses OpenAI Structured Outputs / JSON Schema for reliable JSON.
Builds relevant-page context instead of blind 6000-char truncation.
Validates with Pydantic and retries once on failure.
"""
from __future__ import annotations
import json
import logging
from typing import Optional
from .models import (
    ParsedDocument, AuditEvidence, AuditOverview, AuditPeriod,
    LinkKeys, ExtractionMeta, DocumentFamily,
    Party, Amount, DateItem, Identifier, AssetItem,
    Fact, Claim, Flag, Provenance,
)

logger = logging.getLogger(__name__)

# ── Canonical result cache ────────────────────────────────────────────────────
# Keyed by file_hash + mode + model + schema_version
# Two levels: in-process memory (fast) + disk (survives restarts)
_canonical_cache: dict[str, "AuditEvidence"] = {}

import os as _os
import json as _json

_DISK_CACHE_DIR = _os.path.join(_os.path.dirname(__file__), "..", ".canonical_cache")

def _disk_cache_path(cache_key: str) -> str:
    _os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
    return _os.path.join(_DISK_CACHE_DIR, f"{cache_key}.json")

def _load_from_disk(cache_key: str) -> "Optional[AuditEvidence]":
    """Load a cached AuditEvidence from disk. Returns None if not found."""
    path = _disk_cache_path(cache_key)
    try:
        if _os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            from .models import AuditEvidence
            ev = AuditEvidence(**data)
            _canonical_cache[cache_key] = ev  # warm memory cache too
            return ev
    except Exception as e:
        logger.debug(f"Disk cache read failed for {cache_key}: {e}")
    return None

def _save_to_disk(cache_key: str, evidence: "AuditEvidence") -> None:
    """Persist a canonical result to disk cache."""
    path = _disk_cache_path(cache_key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(evidence.model_dump(), f)
    except Exception as e:
        logger.debug(f"Disk cache write failed for {cache_key}: {e}")

SCHEMA_VERSION = "v05.1"  # Bump when schema changes to invalidate cache

# Keywords that indicate audit-relevant pages
_AUDIT_KEYWORDS = {
    "agreement", "contract", "invoice", "total", "amount", "payment",
    "term", "effective", "date", "approved", "board", "grant", "award",
    "lease", "rent", "salary", "payroll", "bank", "balance", "account",
    "signature", "signed", "authorized", "hereby", "whereas", "party",
    "vendor", "client", "lessee", "lessor", "grantor", "grantee",
    "monthly", "annual", "fiscal", "period", "cfda", "schedule",
    "fixed charge", "original value", "term in months",
}

MAX_CONTEXT_CHARS = 20000  # Safe limit for structured extraction


def _score_page_relevance(page_text: str) -> float:
    """Score a page by audit keyword density."""
    if not page_text:
        return 0.0
    text_lower = page_text.lower()
    hits = sum(1 for kw in _AUDIT_KEYWORDS if kw in text_lower)
    # Bonus for pages with numbers (amounts, dates)
    import re
    numbers = len(re.findall(r'\$[\d,]+|\d+\.\d+|\d{4}-\d{2}-\d{2}', page_text))
    return hits * 2.0 + numbers * 1.5


def build_relevant_page_context(parsed_doc: ParsedDocument, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Build a context string from the most audit-relevant pages.
    Always includes first 2 pages and last page (often signature/summary).
    Then fills remaining budget with highest-scoring pages.
    """
    pages = parsed_doc.pages
    if not pages:
        return parsed_doc.full_text[:max_chars]

    n = len(pages)

    # Always include: first 2, last 1
    priority_indices = set()
    priority_indices.add(0)
    if n > 1:
        priority_indices.add(1)
    if n > 2:
        priority_indices.add(n - 1)

    # Score remaining pages
    scored = [
        (i, _score_page_relevance(pages[i].text))
        for i in range(n) if i not in priority_indices
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build ordered page list: priority first, then by score
    ordered = list(priority_indices) + [i for i, _ in scored]
    ordered_unique = list(dict.fromkeys(ordered))  # Preserve order, dedupe

    # Fill context up to max_chars
    parts: list[str] = []
    used_chars = 0

    for i in ordered_unique:
        page = pages[i]
        if not page.text.strip():
            continue
        page_header = f"[Page {page.page_number} | extractor: {page.extractor}]"
        chunk = f"{page_header}\n{page.text.strip()}"
        if used_chars + len(chunk) > max_chars:
            # Include partial if we have room
            remaining = max_chars - used_chars
            if remaining > 200:
                parts.append(chunk[:remaining] + "\n...[truncated]")
            break
        parts.append(chunk)
        used_chars += len(chunk)

    # Append table summaries
    if parsed_doc.tables:
        table_parts = ["\n\n--- EXTRACTED TABLES ---"]
        for tbl in parsed_doc.tables[:8]:
            if isinstance(tbl, dict):
                headers = tbl.get("headers", [])
                rows = tbl.get("rows", [])[:5]
                page_n = tbl.get("page_number", "?")
            else:
                headers = tbl.headers
                rows = tbl.rows[:5]
                page_n = tbl.page_number
            if headers:
                table_parts.append(f"\nTable (page {page_n}): {' | '.join(str(h) for h in headers)}")
                for row in rows:
                    table_parts.append("  " + " | ".join(str(v) for v in row.values()))

        table_text = "\n".join(table_parts)
        if used_chars + len(table_text) < max_chars:
            parts.append(table_text)


    # Financial pre-classification hint — injected when file was pre-classified
    _fin_data = getattr(parsed_doc, "_financial_data", None)
    if _fin_data:
        _ftype = _fin_data.get("doc_type", "")
        if _ftype and _ftype != "not_financial_structured_data":
            _fp = _fin_data.get("period_start") or "not found"
            _ft = str(_fin_data.get("totals", {}))[:200]
            _fhint = "\n\n--- FINANCIAL FILE PRE-CLASSIFICATION ---\n"
            _fhint += "This file has been pre-classified as: " + _ftype + "\n"
            _fhint += "Period detected: " + _fp + "\n"
            _fhint += "Key totals (computed): " + _ft + "\n"
            _fhint += "Focus extraction on audit-relevant facts for a " + _ftype + ". Do not reclassify."
            if used_chars + len(_fhint) < max_chars:
                parts.append(_fhint)

    # Form-table truncation hint
    # Scanned two-column form tables often lose right-column values to OCR
    # Detect "Label: | Label:" with no value after second label -> flag for AI
    import re as _re
    full = parsed_doc.full_text or ""
    truncated = _re.findall(
        r'([A-Z][A-Za-z /]+:)\s*\|\s*([A-Z][A-Za-z /]+:)\s*$',
        full, _re.MULTILINE
    )
    if truncated:
        hint = ["\n\n--- FORM TABLE OCR NOTE ---",
                "Two-column form rows detected where right-column value is missing from OCR.",
                "Flag these fields as unextractable rather than null:"]
        for left, right in truncated[:5]:
            hint.append(f"  LEFT: '{left.strip()}' has value | RIGHT: '{right.strip()}' value missing")
        hint_text = "\n".join(hint)
        if used_chars + len(hint_text) < max_chars:
            parts.append(hint_text)

    return "\n\n".join(parts)


# ── JSON Schema for Structured Outputs ───────────────────────────────────────

CANONICAL_JSON_SCHEMA = {
    "name": "audit_evidence",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "family", "subtype", "title", "audit_overview",
            "parties", "amounts", "dates", "identifiers", "assets",
            "facts", "claims", "flags", "link_keys", "document_specific"
        ],
        "properties": {
            "family":  {"type": "string"},
            "subtype": {"type": ["string", "null"]},
            "title":   {"type": ["string", "null"]},

            "audit_overview": {
                "type": "object",
                "additionalProperties": False,
                "required": ["summary", "audit_areas", "assertions", "period", "match_targets"],
                "properties": {
                    "summary":      {"type": "string"},
                    "audit_areas":  {"type": "array", "items": {"type": "string"}},
                    "assertions":   {"type": "array", "items": {"type": "string"}},
                    "period": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["effective_date", "start", "end", "term_months"],
                                "properties": {
                                    "effective_date": {"type": ["string", "null"]},
                                    "start":          {"type": ["string", "null"]},
                                    "end":            {"type": ["string", "null"]},
                                    "term_months":    {"type": ["integer", "null"]}
                                }
                            },
                            {"type": "null"}
                        ]
                    },
                    "match_targets": {"type": "array", "items": {"type": "string"}}
                }
            },

            "parties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "name", "normalized", "provenance"],
                    "properties": {
                        "role":       {"type": "string"},
                        "name":       {"type": "string"},
                        "normalized": {"type": "string"},
                        "provenance": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["page", "quote", "confidence"],
                                    "properties": {
                                        "page":       {"type": ["integer", "null"]},
                                        "quote":      {"type": ["string", "null"]},
                                        "confidence": {"type": "number"}
                                    }
                                },
                                {"type": "null"}
                            ]
                        }
                    }
                }
            },

            "amounts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "value", "currency", "provenance"],
                    "properties": {
                        "type":     {"type": "string"},
                        "value":    {"type": "number"},
                        "currency": {"type": "string"},
                        "provenance": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["page", "quote", "confidence"],
                                    "properties": {
                                        "page":       {"type": ["integer", "null"]},
                                        "quote":      {"type": ["string", "null"]},
                                        "confidence": {"type": "number"}
                                    }
                                },
                                {"type": "null"}
                            ]
                        }
                    }
                }
            },

            "dates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "value", "provenance"],
                    "properties": {
                        "type":       {"type": "string"},
                        "value":      {"type": "string"},
                        "provenance": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["page", "quote", "confidence"],
                                    "properties": {
                                        "page":       {"type": ["integer", "null"]},
                                        "quote":      {"type": ["string", "null"]},
                                        "confidence": {"type": "number"}
                                    }
                                },
                                {"type": "null"}
                            ]
                        }
                    }
                }
            },

            "identifiers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "value", "provenance"],
                    "properties": {
                        "type":       {"type": "string"},
                        "value":      {"type": "string"},
                        "provenance": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["page", "quote", "confidence"],
                                    "properties": {
                                        "page":       {"type": ["integer", "null"]},
                                        "quote":      {"type": ["string", "null"]},
                                        "confidence": {"type": "number"}
                                    }
                                },
                                {"type": "null"}
                            ]
                        }
                    }
                }
            },

            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "description", "value", "provenance"],
                    "properties": {
                        "type":        {"type": "string"},
                        "description": {"type": "string"},
                        "value":       {"type": ["number", "null"]},
                        "provenance":  {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["page", "quote", "confidence"],
                                    "properties": {
                                        "page":       {"type": ["integer", "null"]},
                                        "quote":      {"type": ["string", "null"]},
                                        "confidence": {"type": "number"}
                                    }
                                },
                                {"type": "null"}
                            ]
                        }
                    }
                }
            },

            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "value", "provenance"],
                    "properties": {
                        "label":      {"type": "string"},
                        "value":      {"type": ["string", "number", "integer", "boolean", "null"]},
                        "provenance": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["page", "quote", "confidence"],
                                    "properties": {
                                        "page":       {"type": ["integer", "null"]},
                                        "quote":      {"type": ["string", "null"]},
                                        "confidence": {"type": "number"}
                                    }
                                },
                                {"type": "null"}
                            ]
                        }
                    }
                }
            },

            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["statement", "audit_area", "basis_fact_labels", "provenance"],
                    "properties": {
                        "statement":         {"type": "string"},
                        "audit_area":        {"type": "string"},
                        "basis_fact_labels": {"type": "array", "items": {"type": "string"}},
                        "provenance":        {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["page", "quote", "confidence"],
                                    "properties": {
                                        "page":       {"type": ["integer", "null"]},
                                        "quote":      {"type": ["string", "null"]},
                                        "confidence": {"type": "number"}
                                    }
                                },
                                {"type": "null"}
                            ]
                        }
                    }
                }
            },

            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "description", "severity"],
                    "properties": {
                        "type":        {"type": "string"},
                        "description": {"type": "string"},
                        "severity":    {"type": "string", "enum": ["info", "warning", "critical"]}
                    }
                }
            },

            "link_keys": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "party_names", "document_numbers", "agreement_numbers",
                    "invoice_numbers", "asset_descriptions", "recurring_amounts",
                    "key_dates", "other_ids"
                ],
                "properties": {
                    "party_names":        {"type": "array", "items": {"type": "string"}},
                    "document_numbers":   {"type": "array", "items": {"type": "string"}},
                    "agreement_numbers":  {"type": "array", "items": {"type": "string"}},
                    "invoice_numbers":    {"type": "array", "items": {"type": "string"}},
                    "asset_descriptions": {"type": "array", "items": {"type": "string"}},
                    "recurring_amounts":  {"type": "array", "items": {"type": "number"}},
                    "key_dates":          {"type": "array", "items": {"type": "string"}},
                    "other_ids":          {"type": "array", "items": {"type": "string"}}
                }
            },

            "document_specific": {
                "type": "object",
                "additionalProperties": False,
                "required": [],
                "properties": {}
            }
        }
    }
}


CANONICAL_SYSTEM = """You are a senior CPA auditor extracting canonical audit evidence from financial documents.

The document has already been extracted — your job is to identify and structure the audit-relevant facts.

RULES:
1. Extract ONLY facts explicitly supported by the document text. Never infer or calculate.
2. Every material amount, date, party, and identifier MUST include provenance: page number, short quote (≤15 words), confidence (0.0-1.0).
3. facts[] = atomic matchable items (term_months=72, monthly_charge=2273.00)
4. claims[] = auditor-readable interpretations built from facts ("72-month commitment at $2,273/month")
5. amounts must be numeric values, never strings.
6. dates must be YYYY-MM-DD format.
7. normalized party names must be UPPERCASE with no punctuation.
8. link_keys must be populated — these drive future matching.
9. Use null for any field you cannot confidently extract.
10. document_specific holds extras beyond the universal schema.
11. This PDF may contain a primary agreement plus attached supporting pages. You have been given ONLY the primary document pages. Extract core agreement terms from these pages only. If you see references to attached specifications, proposals, or exhibits, note them in document_specific but do NOT treat them as primary agreement facts. Do not populate core fields (term_months, fixed_charge, schedule_number, original_value) from attachment pages.
12. For bank statements and high-transaction documents: include summary amounts only (beginning balance, ending balance, total deposits, total withdrawals, total fees) in amounts[]. Do NOT list every individual transaction as a separate fact or amount — summarize instead. Include account number and statement period in identifiers[]. Keep facts[] to ≤10 most audit-critical items.
13. Keep the total JSON response as concise as possible while capturing all audit-critical information. Quality over quantity.
14. flags[].description must be self-contained audit context — written so a reviewer can act on it without opening the document. Include: the specific party name or firm, the dollar amount if present, the relationship or role, and whether it was properly approved (vote outcome, abstention). Bad: "Related party transaction identified." Good: "Payment of $18,500 to Thompson Consulting, a firm owned by board member David Thompson (Treasurer). Board approved 6-1 at March meeting with Thompson abstaining."
15. For governance documents (board minutes), always flag: related-party transactions with names and amounts, conditional approvals with the specific condition, any dissenting votes, executive compensation approvals with the amount, and any items noted as tabled or deferred.

DOCUMENT FAMILY OPTIONS:
contract_agreement | invoice_receipt | payment_proof | bank_cash_activity |
payroll_support | accounting_report | governance_approval | grant_donor_funding |
tax_regulatory | correspondence | schedule_listing | other

AUDIT AREAS: cash, receivables, payables, fixed_assets, leases, prepaid,
investments, debt, equity, revenue, expenses, payroll, grants, taxes,
insurance, commitments, contingencies, disclosures, governance

ASSERTIONS: existence, completeness, accuracy, cutoff, classification,
rights_and_obligations, valuation, presentation, disclosure"""


def _parse_provenance(d) -> Optional[Provenance]:
    if not d or not isinstance(d, dict) or not any(d.values()):
        return None
    return Provenance(
        page=d.get("page"),
        quote=d.get("quote"),
        confidence=float(d.get("confidence", 0.0)),
    )


def _parse_response(data: dict, source_file: str, parsed_doc: ParsedDocument,
                    meta: ExtractionMeta) -> AuditEvidence:
    """Parse validated JSON response into AuditEvidence."""

    family_val = data.get("family", "other")
    try:
        family = DocumentFamily(family_val)
    except ValueError:
        family = DocumentFamily.OTHER

    od = data.get("audit_overview") or {}
    pd_data = od.get("period") or {}
    overview = AuditOverview(
        summary=od.get("summary", ""),
        audit_areas=od.get("audit_areas", []),
        assertions=od.get("assertions", []),
        period=AuditPeriod(
            effective_date=pd_data.get("effective_date"),
            start=pd_data.get("start"),
            end=pd_data.get("end"),
            term_months=pd_data.get("term_months"),
        ) if pd_data else None,
        match_targets=od.get("match_targets", []),
    )

    parties = [
        Party(role=p["role"], name=p["name"],
              normalized=p.get("normalized", p["name"].upper()),
              provenance=_parse_provenance(p.get("provenance")))
        for p in (data.get("parties") or []) if p.get("name")
    ]

    amounts = [
        Amount(type=a["type"], value=float(a["value"]),
               currency=a.get("currency", "USD"),
               provenance=_parse_provenance(a.get("provenance")))
        for a in (data.get("amounts") or []) if a.get("value") is not None
    ]

    dates = [
        DateItem(type=d["type"], value=str(d["value"]),
                 provenance=_parse_provenance(d.get("provenance")))
        for d in (data.get("dates") or []) if d.get("value")
    ]

    identifiers = [
        Identifier(type=i["type"], value=str(i["value"]),
                   provenance=_parse_provenance(i.get("provenance")))
        for i in (data.get("identifiers") or []) if i.get("value")
    ]

    assets = [
        AssetItem(type=a["type"], description=a["description"],
                  value=float(a["value"]) if a.get("value") is not None else None,
                  provenance=_parse_provenance(a.get("provenance")))
        for a in (data.get("assets") or []) if a.get("description")
    ]

    facts = [
        Fact(label=f["label"], value=f["value"],
             provenance=_parse_provenance(f.get("provenance")))
        for f in (data.get("facts") or []) if f.get("label") and f.get("value") is not None
    ]

    claims = [
        Claim(statement=c["statement"], audit_area=c.get("audit_area", "other"),
              basis_fact_labels=c.get("basis_fact_labels", []),
              provenance=_parse_provenance(c.get("provenance")))
        for c in (data.get("claims") or []) if c.get("statement")
    ]

    flags = [
        Flag(type=f["type"], description=f["description"],
             severity=f.get("severity", "info"))
        for f in (data.get("flags") or []) if f.get("description")
    ]

    lk_data = data.get("link_keys") or {}
    link_keys = LinkKeys(
        party_names=[str(n).upper() for n in lk_data.get("party_names", [])],
        document_numbers=[str(n) for n in lk_data.get("document_numbers", [])],
        agreement_numbers=[str(n) for n in lk_data.get("agreement_numbers", [])],
        invoice_numbers=[str(n) for n in lk_data.get("invoice_numbers", [])],
        asset_descriptions=[str(n).upper() for n in lk_data.get("asset_descriptions", [])],
        recurring_amounts=[float(a) for a in lk_data.get("recurring_amounts", []) if a],
        key_dates=[str(d) for d in lk_data.get("key_dates", [])],
        other_ids=[str(i) for i in lk_data.get("other_ids", [])],
    )

    return AuditEvidence(
        source_file=source_file,
        family=family,
        subtype=data.get("subtype"),
        title=data.get("title"),
        audit_overview=overview,
        parties=parties,
        amounts=amounts,
        dates=dates,
        identifiers=identifiers,
        assets=assets,
        facts=facts,
        claims=claims,
        flags=flags,
        link_keys=link_keys,
        document_specific=data.get("document_specific") or {},
        raw_text=parsed_doc.full_text,
        tables=[t if isinstance(t, dict) else t.model_dump() for t in parsed_doc.tables],
        extraction_meta=meta,
    )


def _canonical_cache_key(parsed_doc: ParsedDocument, model: str,
                         mode: str = "fast") -> str:
    """
    Build canonical cache key.
    Includes extraction mode so fast/deep results are cached independently.
    Re-uploading in deep mode after a fast failure will run fresh.
    Uses file_hash (MD5 of file content) when available.
    Falls back to MD5 of full_text if file_hash not populated.
    """
    import hashlib
    base = parsed_doc.file_hash or hashlib.md5(
        (parsed_doc.full_text or "").encode("utf-8", errors="ignore")
    ).hexdigest()
    h = hashlib.md5()
    h.update(f"{base}:{model}:{mode}:{SCHEMA_VERSION}".encode())
    return h.hexdigest()


def extract_canonical(
    parsed_doc: ParsedDocument,
    provider,
    mode: str = "fast",
    bypass_cache: bool = False,
) -> AuditEvidence:
    """
    Run single AI canonical extraction pass on a ParsedDocument.
    Uses structured outputs for reliable JSON.
    Validates with Pydantic and retries once on failure.
    Caches result by file content + model + schema version.
    bypass_cache: if True, skip cache lookup and force a fresh AI call.
    """
    # Check in-memory cache first, then disk cache
    model_name = getattr(provider, "model", "unknown")
    cache_key = _canonical_cache_key(parsed_doc, model_name, mode=mode)
    if not bypass_cache:
        if cache_key in _canonical_cache:
            logger.info(f"Canonical memory cache hit: {parsed_doc.source_file}")
            return _canonical_cache[cache_key]
        disk_hit = _load_from_disk(cache_key)
        if disk_hit is not None:
            # Reject stale cache entries that contain critical flags — these are
            # bad results from a previously failed extraction and must not be reused.
            _has_critical = any(f.severity == "critical" for f in (disk_hit.flags or []))
            if _has_critical:
                logger.info(
                    f"Canonical disk cache rejected (has critical flags): "
                    f"{parsed_doc.source_file} — will re-extract"
                )
                _canonical_cache.pop(cache_key, None)
                try:
                    import os as _rej_os
                    _rej_path = _disk_cache_path(cache_key)
                    if _rej_os.path.exists(_rej_path):
                        _rej_os.unlink(_rej_path)
                except Exception:
                    pass
            else:
                logger.info(f"Canonical disk cache hit: {parsed_doc.source_file}")
                return disk_hit
    else:
        # Remove stale cache entry so fresh result is saved after extraction
        _canonical_cache.pop(cache_key, None)
        try:
            import os as _bcos
            _stale = _disk_cache_path(cache_key)
            if _bcos.path.exists(_stale):
                _bcos.unlink(_stale)
        except Exception:
            pass
        logger.info(f"Canonical cache bypassed for: {parsed_doc.source_file}")
    meta = ExtractionMeta(
        primary_extractor=parsed_doc.primary_extractor,
        pages_processed=parsed_doc.page_count,
        weak_pages_count=len(parsed_doc.weak_pages),
        ocr_pages_count=len(parsed_doc.ocr_pages),
        vision_pages_count=len(parsed_doc.vision_pages),
        total_chars=len(parsed_doc.full_text),
        overall_confidence=parsed_doc.confidence,
        warnings=parsed_doc.warnings,
        errors=parsed_doc.errors,
    )

    if not parsed_doc.full_text or not parsed_doc.full_text.strip():
        meta.errors.append("No text available for canonical extraction")
        return AuditEvidence(
            source_file=parsed_doc.source_file,
            extraction_meta=meta,
            flags=[Flag(type="no_text", description="No text extracted", severity="critical")]
        )

    context = build_relevant_page_context(parsed_doc)
    user_prompt = (
        f"Filename: {parsed_doc.source_file}\n"
        f"Pages: {parsed_doc.page_count} | "
        f"Extraction chain: {' → '.join(parsed_doc.extraction_chain)}\n\n"
        f"{context}"
    )

    # Attempt 1: structured extraction
    try:
        raw = provider.extract_structured(
            system=CANONICAL_SYSTEM,
            user=user_prompt,
            json_schema=CANONICAL_JSON_SCHEMA,
            max_tokens=16000,
        )
        data = raw if isinstance(raw, dict) else json.loads(raw)
        evidence = _parse_response(data, parsed_doc.source_file, parsed_doc, meta)
        meta.canonical_validated = True
        _canonical_cache[cache_key] = evidence
        _save_to_disk(cache_key, evidence)
        return evidence

    except Exception as e:
        logger.warning(f"Canonical extraction attempt 1 failed: {e} — retrying")

    # Attempt 2: retry with simplified prompt
    try:
        repair_prompt = (
            f"The previous extraction failed. Try again with this document.\n\n"
            f"Filename: {parsed_doc.source_file}\n\n"
            f"{context[:10000]}"
        )
        raw = provider.extract_structured(
            system=CANONICAL_SYSTEM,
            user=repair_prompt,
            json_schema=CANONICAL_JSON_SCHEMA,
            max_tokens=16000,
        )
        data = raw if isinstance(raw, dict) else json.loads(raw)
        evidence = _parse_response(data, parsed_doc.source_file, parsed_doc, meta)
        meta.canonical_validated = True
        meta.canonical_retried = True
        _canonical_cache[cache_key] = evidence
        _save_to_disk(cache_key, evidence)
        return evidence

    except Exception as e:
        logger.error(f"Canonical extraction failed after retry: {e}")
        meta.errors.append(f"Canonical extraction failed: {e}")
        return AuditEvidence(
            source_file=parsed_doc.source_file,
            raw_text=parsed_doc.full_text,
            tables=[t if isinstance(t, dict) else t.model_dump() for t in parsed_doc.tables],
            extraction_meta=meta,
            flags=[Flag(type="extraction_failed",
                        description=f"AI extraction failed after retry: {e}",
                        severity="critical")]
        )
