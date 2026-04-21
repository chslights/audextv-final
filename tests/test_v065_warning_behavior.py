"""
tests/test_v065_warning_behavior.py — v06.5 regression tests.

Four behavior changes landed in v06.5, all covered here:

1. VISION_FIXABLE_FLAGS is exported from readiness and contains the
   three expected flag types.

2. reconcile_readiness_with_warnings (the ingest_app cleanup path)
   cannot leave a Ready evidence object with an unresolved warning-
   severity flag. The PolyRx bug.

3. Auto-rescue CLEARS vision-fixable flags when allow_rescue=True,
   a provider is available, and vision succeeds with strong page
   confidence.

4. Auto-rescue DOWNGRADES vision-fixable flags (warning → info) when
   vision runs but confidence stays weak — preserving provenance
   without artificially blocking readiness.

Tests use either direct function calls on the readiness module or
a source-inspection check on router.ingest_one. None of them need
Streamlit or a live API key.
"""
from __future__ import annotations

import inspect

import pytest

from audit_ingestion.models import (
    AuditEvidence, AuditOverview, AuditPeriod, DocumentFamily,
    ExtractionMeta, Flag, ReadinessResult,
)
from audit_ingestion.readiness import (
    VISION_FIXABLE_FLAGS, apply_readiness, compute_readiness,
    reconcile_readiness_with_warnings,
)


# ── 1. VISION_FIXABLE_FLAGS constant ──────────────────────────────────────

def test_vision_fixable_flags_is_frozen_set_with_correct_members():
    """The canonical list of vision-fixable flag types."""
    assert isinstance(VISION_FIXABLE_FLAGS, frozenset)
    assert VISION_FIXABLE_FLAGS == frozenset({
        "handwriting_detected",
        "ocr_limitations",
        "ocr_quality",
    })


def test_vision_fixable_flags_all_have_readiness_rules():
    """Every fixable type must have a _FLAG_RULES entry so questions generate."""
    from audit_ingestion.readiness import _FLAG_RULES
    for flag_type in VISION_FIXABLE_FLAGS:
        assert flag_type in _FLAG_RULES, f"{flag_type} missing from _FLAG_RULES"
        assert _FLAG_RULES[flag_type] is not None


def test_vision_fixable_flag_rules_are_reviewer_blocking():
    """The three fixable flag types should all be reviewer-blocking so they
    produce needs_reviewer_confirmation status (not client-blocking)."""
    from audit_ingestion.readiness import _FLAG_RULES
    for flag_type in VISION_FIXABLE_FLAGS:
        rule = _FLAG_RULES[flag_type]
        assert rule["audience"] == "reviewer"
        assert rule["blocking"] is True
        assert rule["question_type"] == "ocr_quality_review"


# ── 2. Ready + warning cannot coexist ────────────────────────────────────

def _make_evidence(
    *,
    flags: list[Flag] = None,
    readiness_status: str = "ready",
    total_chars: int = 5000,
    workflow: dict = None,
) -> AuditEvidence:
    """Build a minimal AuditEvidence with the given flags and readiness."""
    doc_specific = {}
    if workflow:
        doc_specific["_workflow"] = workflow
    ev = AuditEvidence(
        source_file="test.pdf",
        family=DocumentFamily.OTHER,
        subtype=None,
        document_specific=doc_specific,
        extraction_meta=ExtractionMeta(
            primary_extractor="pdfplumber",
            pages_processed=1,
            total_chars=total_chars,
            overall_confidence=0.85,
        ),
        flags=flags or [],
        readiness=ReadinessResult(
            readiness_status=readiness_status,
            blocking_state="non_blocking",
            questions=[],
        ),
    )
    return ev


def test_reconcile_demotes_ready_when_fixable_warning_remains():
    """
    v06.5 regression: if readiness says 'ready' but a warning-severity
    vision-fixable flag is still unresolved, demote to
    needs_reviewer_confirmation with blocking state.
    """
    ev = _make_evidence(
        flags=[Flag(type="handwriting_detected",
                    description="handwriting on page 1",
                    severity="warning")],
        readiness_status="ready",
    )
    reconcile_readiness_with_warnings(ev)
    assert ev.readiness.readiness_status == "needs_reviewer_confirmation"
    assert ev.readiness.blocking_state == "blocking"


def test_reconcile_demotes_ready_when_ocr_limitations_remain():
    """Same rule for ocr_limitations."""
    ev = _make_evidence(
        flags=[Flag(type="ocr_limitations",
                    description="3 weak pages",
                    severity="warning")],
        readiness_status="ready",
    )
    reconcile_readiness_with_warnings(ev)
    assert ev.readiness.readiness_status == "needs_reviewer_confirmation"


def test_reconcile_demotes_ready_to_exception_open_for_judgment_warning():
    """
    Non-vision-fixable warnings (related_party, date_inconsistency, etc.)
    demote to exception_open, not needs_reviewer_confirmation.
    """
    ev = _make_evidence(
        flags=[Flag(type="related_party",
                    description="related-party identified",
                    severity="warning")],
        readiness_status="ready",
    )
    reconcile_readiness_with_warnings(ev)
    assert ev.readiness.readiness_status == "exception_open"
    assert ev.readiness.blocking_state == "non_blocking"


def test_reconcile_keeps_ready_when_no_warnings_remain():
    """Control case: ready stays ready when no warning-severity flags remain."""
    ev = _make_evidence(
        flags=[Flag(type="tb_balanced",
                    description="TB balanced",
                    severity="info")],
        readiness_status="ready",
    )
    reconcile_readiness_with_warnings(ev)
    assert ev.readiness.readiness_status == "ready"


def test_reconcile_keeps_ready_when_info_severity_flag_present():
    """Info-severity flags do not block ready. Only warning-severity does."""
    ev = _make_evidence(
        flags=[
            Flag(type="handwriting_detected",
                 description="handwriting present — downgraded by rescue",
                 severity="info"),
        ],
        readiness_status="ready",
    )
    reconcile_readiness_with_warnings(ev)
    assert ev.readiness.readiness_status == "ready"


def test_reconcile_promotes_exception_open_to_ready_when_clean():
    """
    Stuck in exception_open with no open questions and no active warnings
    → promote to ready. This is the v06.4 behavior preserved.
    """
    ev = _make_evidence(
        flags=[],
        readiness_status="exception_open",
    )
    reconcile_readiness_with_warnings(ev)
    assert ev.readiness.readiness_status == "ready"


def test_reconcile_keeps_ready_when_warning_in_resolved_exceptions():
    """
    If a warning's flag type appears in _workflow.resolved_exceptions,
    it's treated as resolved and doesn't block promotion. Stale flag
    gets cleaned up.
    """
    ev = _make_evidence(
        flags=[Flag(type="handwriting_detected",
                    description="resolved handwriting",
                    severity="warning")],
        readiness_status="exception_open",
        workflow={
            "resolved_exceptions": [
                {"source_flag": "handwriting_detected", "resolution": "confirmed"},
            ],
        },
    )
    reconcile_readiness_with_warnings(ev)
    assert ev.readiness.readiness_status == "ready"
    # The stale flag should have been cleaned up.
    assert not any(f.type == "handwriting_detected" for f in ev.flags)


# ── 3 & 4. Auto-rescue in router: source-level guards ─────────────────────

def test_router_auto_rescue_is_guarded_by_allow_rescue():
    """
    The router's auto-rescue block must be gated on allow_rescue so
    'Mode B' users (auto-rescue disabled) don't get silent vision runs.
    Verify by inspecting router.ingest_one source.
    """
    import audit_ingestion.router as router_mod
    src = inspect.getsource(router_mod.ingest_one)
    assert "allow_rescue and provider is not None and _fixable_present" in src, \
        "Auto-rescue must be guarded by allow_rescue"


def test_router_imports_vision_fixable_flags():
    """Router must use the shared constant, not a local hard-coded list."""
    import audit_ingestion.router as router_mod
    src = inspect.getsource(router_mod.ingest_one)
    assert "VISION_FIXABLE_FLAGS" in src
    assert "from .readiness import VISION_FIXABLE_FLAGS" in src


def test_router_strong_rescue_threshold():
    """
    Strong-rescue threshold per the v06.5 spec:
      - pages_succeeded >= 75% of target pages
      - avg page confidence >= 0.70
    Verify both constants appear so the policy isn't silently relaxed.
    """
    import audit_ingestion.router as router_mod
    src = inspect.getsource(router_mod.ingest_one)
    assert "0.75" in src, "Strong-rescue page-success threshold must be 0.75"
    assert "0.70" in src, "Strong-rescue avg-confidence threshold must be 0.70"


def test_router_weak_rescue_downgrades_severity_to_info():
    """
    Weak rescue must downgrade fixable-flag severity to 'info' rather
    than silently removing the flag.
    """
    import audit_ingestion.router as router_mod
    src = inspect.getsource(router_mod.ingest_one)
    assert '_f.severity = "info"' in src, \
        "Weak rescue should downgrade severity to info"


def test_router_logs_rescue_applied_provenance_flag():
    """
    Every rescue (strong or weak) must add a 'rescue_applied' info flag
    so provenance is visible in the evidence record.
    """
    import audit_ingestion.router as router_mod
    src = inspect.getsource(router_mod.ingest_one)
    assert 'type="rescue_applied"' in src


# ── Readiness consistency with fixable warnings ──────────────────────────

def test_compute_readiness_with_fixable_warning_is_not_ready():
    """
    compute_readiness() should NOT return status='ready' when a
    vision-fixable warning is present. These three flag types are
    marked blocking=True/audience=reviewer.
    """
    for flag_type in VISION_FIXABLE_FLAGS:
        ev = _make_evidence(
            flags=[Flag(type=flag_type, description="test", severity="warning")],
            readiness_status="ready",
        )
        result = compute_readiness(ev)
        assert result.readiness_status != "ready", \
            f"{flag_type} should not yield ready status"
        assert result.readiness_status == "needs_reviewer_confirmation"
        assert any(q.blocking and not q.resolved for q in result.questions)
