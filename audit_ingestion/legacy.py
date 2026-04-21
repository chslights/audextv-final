"""
audit_ingestion_v04.2/audit_ingestion/legacy.py
Compatibility layer — derives legacy field dict from canonical AuditEvidence.
Never the primary output. Useful for downstream systems expecting the old format.
"""
from __future__ import annotations
from .models import AuditEvidence


def canonical_to_legacy_fields(evidence: AuditEvidence) -> dict:
    """
    Derive a flat field dict from canonical AuditEvidence.
    Maps canonical objects to the simple key-value format used in v02.
    """
    fields: dict = {}

    # Parties
    for i, party in enumerate(evidence.parties[:4]):
        if i == 0:
            fields["party_a"] = party.name
            fields["party_a_role"] = party.role
        elif i == 1:
            fields["party_b"] = party.name
            fields["party_b_role"] = party.role

    # Amounts — pick the most significant ones
    for amt in evidence.amounts:
        key = amt.type.lower().replace(" ", "_")
        fields[key] = amt.value

    # Pick "total" amount if available
    total_candidates = [a for a in evidence.amounts
                        if any(kw in a.type.lower() for kw in
                               ["total", "award", "invoice", "payment", "fixed_charge"])]
    if total_candidates:
        fields["total_amount"] = total_candidates[0].value

    # Dates
    for dt in evidence.dates:
        key = dt.type.lower().replace(" ", "_")
        fields[key] = dt.value

    # Identifiers
    for ident in evidence.identifiers:
        key = ident.type.lower().replace(" ", "_")
        fields[key] = ident.value

    # Overview
    if evidence.audit_overview:
        fields["summary"] = evidence.audit_overview.summary
        fields["audit_areas"] = ", ".join(evidence.audit_overview.audit_areas)
        fields["doc_family"] = evidence.family.value
        fields["doc_subtype"] = evidence.subtype or ""

    return fields


def canonical_summary_row(evidence: AuditEvidence) -> dict:
    """
    Build a one-row summary dict for display in a summary table.
    """
    meta = evidence.extraction_meta
    overview = evidence.audit_overview

    # Primary amount
    primary_amount = None
    for amt in evidence.amounts:
        if any(kw in amt.type.lower() for kw in ["total", "award", "fixed_charge", "invoice"]):
            primary_amount = f"${amt.value:,.2f}"
            break

    # Primary party
    primary_party = evidence.parties[0].name if evidence.parties else "—"

    return {
        "file":           evidence.source_file,
        "family":         evidence.family.value,
        "subtype":        evidence.subtype or "—",
        "summary":        overview.summary[:100] + "..." if overview and len(overview.summary) > 100
                          else (overview.summary if overview else "—"),
        "primary_party":  primary_party,
        "primary_amount": primary_amount or "—",
        "audit_areas":    ", ".join(overview.audit_areas[:3]) if overview else "—",
        "confidence":     f"{meta.overall_confidence:.2f}",
        "extractor":      meta.primary_extractor,
        "chars":          meta.total_chars,
        "needs_review":   meta.needs_human_review,
    }
