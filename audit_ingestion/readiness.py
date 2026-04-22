"""
audit_ingestion_v05/audit_ingestion/readiness.py

Evidence readiness engine.

Computes readiness_status and blocking_state from the extracted flags,
then generates structured Question objects for each unresolved issue.

Two separate concerns:
  1. Was the file technically processed? (IngestionResult.status — success/partial/failed)
  2. Is the evidence complete enough for audit use? (ReadinessResult — ready/needs_*/unusable)

A file can be SUCCESS with confidence 1.00 and still not be Ready if it has a
missing period, unconfirmed TB year, or material balance difference.
"""
from __future__ import annotations
import logging
import uuid
from typing import Optional
import re
from .models import AuditEvidence, AuditOverview, AuditPeriod, Flag, Question, ReadinessResult
from .workflow import record_question_event, utc_now_iso

logger = logging.getLogger(__name__)

READINESS_VERSION = "v05.0"

# ── Vision-fixable flag types (v06.5) ─────────────────────────────────────
# The flag types in this set represent extraction-quality problems that a
# vision extraction run can potentially correct. They drive two behaviors:
#
#   1. Auto-rescue (in router.py): when allow_rescue=True and a file emits
#      one of these flags, the router automatically runs vision on the
#      affected pages before finalizing readiness.
#
#   2. Warning-type-aware UI (in ingest_app.py): the Warnings column shows
#      a direct "Run Vision" action for these flag types, instead of
#      routing the user through the generic question-and-answer flow.
#
# Flag types NOT in this set represent judgment calls (related parties,
# date inconsistencies, amendments, legal holds, etc.) — those keep the
# existing reviewer-question flow because vision cannot resolve them.
VISION_FIXABLE_FLAGS = frozenset({
    "handwriting_detected",
    "ocr_limitations",
    "ocr_quality",
})


def reconcile_readiness_with_warnings(evidence: AuditEvidence) -> AuditEvidence:
    """
    v06.5: enforce the "Ready must not coexist with an active warning" rule.

    This is the post-extraction cleanup that runs after apply_readiness().
    It handles two cases:

      (1) Stuck in exception_open with no open questions and no active
          warnings → promote to ready (this is the existing v06.4
          cleanup, preserved).

      (2) Currently marked ready but warning-severity flags are still
          active (not appearing in _workflow.resolved_exceptions) →
          demote so the summary row cannot show a green "Ready" alongside
          a warning chip. Vision-fixable warnings demote to
          needs_reviewer_confirmation (blocking) so the reviewer is
          explicitly prompted. Other warnings demote to exception_open
          (non-blocking) which the summary row still surfaces visually.

    A flag counts as "active" if its type does NOT appear in the
    _workflow.resolved_exceptions list for this evidence. Once a flag has
    been resolved via the workflow (reviewer answered the question, etc.),
    it no longer blocks ready promotion.

    Returns the same evidence object (mutated in place), matching the
    convention used elsewhere in this module.
    """
    if not evidence.readiness:
        return evidence

    open_qs = [q for q in (evidence.readiness.questions or []) if not q.resolved]
    wf = (evidence.document_specific or {}).get("_workflow", {}) or {}
    resolved_keys = {
        item.get("source_flag")
        for item in (wf.get("resolved_exceptions") or [])
        if item.get("source_flag")
    }

    active_warnings = [
        f for f in (evidence.flags or [])
        if getattr(f, "severity", "") == "warning"
        and getattr(f, "type", None) not in resolved_keys
    ]

    if (evidence.readiness.readiness_status == "exception_open"
            and not open_qs and not active_warnings):
        # Case 1: promote to ready and clean up stale resolved flags.
        evidence.readiness.readiness_status = "ready"
        evidence.readiness.blocking_state = "non_blocking"
        evidence.readiness.blocking_issues = []
        if resolved_keys:
            evidence.flags = [
                f for f in (evidence.flags or [])
                if getattr(f, "type", None) not in resolved_keys
            ]
    elif (evidence.readiness.readiness_status == "ready"
            and active_warnings):
        # Case 2: demote so ready + warning cannot coexist.
        if any(f.type in VISION_FIXABLE_FLAGS for f in active_warnings):
            evidence.readiness.readiness_status = "needs_reviewer_confirmation"
            evidence.readiness.blocking_state = "blocking"
        else:
            evidence.readiness.readiness_status = "exception_open"
            evidence.readiness.blocking_state = "non_blocking"

    return evidence

# ── Blocking flag registry ────────────────────────────────────────────────────
# Maps flag type → (audience, question_type, blocking, question_template)
# audience:      "reviewer" | "client"
# blocking:      True = prevents Ready status
# question_text: shown to the user; {details} filled from flag description

_FLAG_RULES: dict[str, dict] = {

    # Financial classification
    "tb_year_unconfirmed": {
        "audience": "reviewer",
        "question_type": "current_vs_prior_year_confirmation",
        "blocking": True,
        "question_text": "Is this the current-year or prior-year trial balance? "
                         "The system could not determine the year from content or filename.",
    },
    "material_balance_difference": {
        "audience": "client",
        "question_type": "balance_difference_explanation",
        "blocking": True,
        "question_text": "This trial balance appears out of balance. "
                         "Please provide an explanation or upload a corrected file.",
    },
    "balance_difference_detected": {
        "audience": "reviewer",
        "question_type": "balance_difference_review",
        "blocking": False,
        "question_text": "A small balance difference was detected. "
                         "Confirm whether this is a rounding difference or requires follow-up.",
    },

    # Period and completeness
    "missing_period": {
        "audience": "client",
        "question_type": "period_confirmation",
        "blocking": True,
        "question_text": "What period does this file cover? "
                         "No fiscal year or date range was found in the file or filename.",
    },
    "missing_page": {
        "audience": "client",
        "question_type": "missing_page_follow_up",
        "blocking": True,
        "question_text": "One or more pages appear to be missing from this document. "
                         "Please provide a complete copy.",
    },
    "partial_extraction": {
        "audience": "reviewer",
        "question_type": "partial_extraction_review",
        "blocking": True,
        "question_text": "Extraction was incomplete for this file. "
                         "Review the extracted content and confirm it is sufficient, "
                         "or rerun with deep extraction mode.",
    },
    "partial_extraction_visibility": {
        "audience": "reviewer",
        "question_type": "partial_extraction_review",
        "blocking": True,
        "question_text": "Some content in this file may not have been fully captured. "
                         "Confirm the extraction is sufficient or rerun.",
    },

    # Document completeness
    "unsigned_agreement": {
        "audience": "client",
        "question_type": "signed_copy_request",
        "blocking": True,
        "question_text": "This agreement does not appear to be fully signed. "
                         "Please provide the fully executed copy.",
    },
    "missing_term_months": {
        "audience": "client",
        "question_type": "missing_term_follow_up",
        "blocking": False,
        "question_text": "The lease term in months could not be extracted. "
                         "Please confirm the full lease term.",
    },
    "reference_to_supporting_document": {
        "audience": "client",
        "question_type": "missing_support_request",
        "blocking": False,
        "question_text": "This document references a supporting schedule or attachment "
                         "that was not included. Please provide the referenced document.",
    },
    "reference_to_underlying_agreement": {
        "audience": "client",
        "question_type": "missing_agreement_request",
        "blocking": False,
        "question_text": "This document references an underlying agreement. "
                         "Please confirm the agreement has been provided separately.",
    },
    "reference_to_other_agreement": {
        "audience": "client",
        "question_type": "missing_agreement_request",
        "blocking": False,
        "question_text": "This document references another agreement. "
                         "Please confirm the referenced agreement is included in the upload.",
    },
    "attachment_reference": {
        "audience": "client",
        "question_type": "missing_attachment_request",
        "blocking": False,
        "question_text": "This document references an attachment. "
                         "Please confirm the attachment is included in the upload.",
    },

    # Bank statement completeness
    "missing_beginning_balance": {
        "audience": "client",
        "question_type": "missing_balance_request",
        "blocking": False,
        "question_text": "The beginning balance is not present in this bank statement export. "
                         "Please provide a version that includes the opening balance.",
    },
    "missing_ending_balance": {
        "audience": "client",
        "question_type": "missing_balance_request",
        "blocking": False,
        "question_text": "The ending balance is not present in this bank statement export. "
                         "Please provide a version that includes the closing balance.",
    },
    "missing_account_number": {
        "audience": "client",
        "question_type": "missing_account_info",
        "blocking": False,
        "question_text": "No bank account number was found in this file. "
                         "Please confirm which account this statement covers.",
    },

    # Reviewer-owned signals (non-blocking by default)
    "related_party": {
        "audience": "reviewer",
        "question_type": "related_party_review",
        "blocking": False,
        "question_text": "A related-party relationship was identified. "
                         "Confirm whether this requires additional disclosure or testing.",
    },
    "related_party_transaction": {
        "audience": "reviewer",
        "question_type": "related_party_review",
        "blocking": False,
        "question_text": "A related-party transaction was identified. "
                         "Confirm whether additional testing or disclosure is required.",
    },
    "conditional_funding": {
        "audience": "reviewer",
        "question_type": "conditional_terms_review",
        "blocking": False,
        "question_text": "This grant or funding contains conditional terms. "
                         "Confirm the conditions have been met or are tracked.",
    },
    "conditional_payment": {
        "audience": "reviewer",
        "question_type": "conditional_terms_review",
        "blocking": False,
        "question_text": "This document contains conditional payment terms. "
                         "Confirm conditions are noted in the workpapers.",
    },
    "conditional_revenue": {
        "audience": "reviewer",
        "question_type": "conditional_terms_review",
        "blocking": False,
        "question_text": "This document contains conditions affecting revenue recognition. "
                         "Confirm the recognition criteria are met.",
    },
    "date_inconsistency": {
        "audience": "reviewer",
        "question_type": "date_discrepancy_review",
        "blocking": False,
        "question_text": "Dates in this document appear inconsistent. "
                         "Review and confirm the correct effective date.",
    },
    "amended_award": {
        "audience": "reviewer",
        "question_type": "amendment_review",
        "blocking": False,
        "question_text": "This appears to be an amended grant award. "
                         "Confirm the original award and amendment are both in the file.",
    },
    "legal_review_pending": {
        "audience": "reviewer",
        "question_type": "legal_review",
        "blocking": False,
        "question_text": "This document appears to have open legal matters. "
                         "Confirm whether legal review has been completed.",
    },
    "related_party_gift": {
        "audience": "reviewer",
        "question_type": "related_party_review",
        "blocking": False,
        "question_text": "A related-party gift or donation was identified. "
                         "Confirm disclosure requirements and whether board approval was obtained.",
    },
    "document_limitation": {
        "audience": "reviewer",
        "question_type": "document_limitation_review",
        "blocking": False,
        "question_text": "This document has a noted limitation in scope or completeness. "
                         "Confirm whether additional documentation is needed.",
    },
    "conditional_approval": {
        "audience": "reviewer",
        "question_type": "conditional_terms_review",
        "blocking": False,
        "question_text": "A conditional approval was identified in this document. "
                         "Confirm the conditions have been met or are tracked in the workpapers.",
    },
    "executive_compensation_approval": {
        "audience": "reviewer",
        "question_type": "governance_review",
        "blocking": False,
        "question_text": "An executive compensation approval was identified. "
                         "Confirm the amount, approval process, and disclosure in the financial statements.",
    },
    "missing_identifier": {
        "audience": "client",
        "question_type": "missing_identifier_request",
        "blocking": False,
        "question_text": "A key identifier (account number, reference number) is missing from this file. "
                         "Please provide a version that includes the identifier.",
    },
    "missing_balance": {
        "audience": "client",
        "question_type": "missing_balance_request",
        "blocking": False,
        "question_text": "An expected balance (beginning or ending) is missing from this file. "
                         "Please provide a version that includes the balance.",
    },

    # Audit signals — duplicate entries should route into workflow, not dead-end as unusable
    "duplicate_entry": {
        "audience": "client",
        "question_type": "corrected_ledger_request",
        "blocking": True,
        "question_text": "A duplicate entry or duplicate-posting note was identified in this file. "
                         "Please provide a corrected ledger or explain whether the entry is valid.",
    },

    # Audit signals — informational, no question generated
    "outlier":                      None,
    "unusual_transaction":          None,
    "bundle_detected":              None,
    "rescue_applied":               None,
    "governance_note":              None,
    "document_nature":              None,
    "approximate_quantity":         None,
    "no_entity_name":               None,
    "tb_balanced":                  None,

    # Bug C fix: OCR quality and handwriting warnings now participate in
    # readiness/summary logic so the summary row cannot show green-ready when
    # handwriting is present or OCR quality is limited.
    "ocr_limitations": {
        "audience":       "reviewer",
        "question_type":  "ocr_quality_review",
        "blocking":       True,
        "question_text":  (
            "This document contains handwriting or has limited OCR quality. "
            "The extracted content may be incomplete or inaccurate. "
            "Run vision extraction to improve accuracy, then confirm the result."
        ),
    },
    "handwriting_detected": {
        "audience":       "reviewer",
        "question_type":  "ocr_quality_review",
        "blocking":       True,
        "question_text":  (
            "Handwriting was detected in this document. "
            "Standard text extraction may have missed handwritten responses. "
            "Run vision extraction to capture handwritten content."
        ),
    },
    # canonical AI emits ocr_quality as a flag type -- alias it so it
    # participates in readiness blocking the same way as ocr_limitations.
    "ocr_quality": {
        "audience":       "reviewer",
        "question_type":  "ocr_quality_review",
        "blocking":       True,
        "question_text":  (
            "This document has OCR quality concerns -- handwritten or low-legibility "
            "content may be incomplete or inaccurate. "
            "Run vision extraction to improve accuracy, then confirm the result."
        ),
    },
}

# Financial file flag overrides — some flags are blocking for financial files
# that would be non-blocking for PDF documents
_FINANCIAL_BLOCKING_OVERRIDES = {
    "missing_period":           True,   # always blocking for financial files
    "missing_beginning_balance":True,   # blocking for bank statements
    "missing_ending_balance":   True,   # blocking for bank statements
    "missing_account_number":   True,   # blocking for bank statements
}


def _format_financial_row_context(row: dict, source_file: str = "") -> str:
    if not row:
        return ""
    row_num = row.get("row_index")
    # rows are usually retained with first data row = 1; convert to CSV line number when possible
    if isinstance(row_num, int):
        display_row = row_num + 1 if row_num >= 1 else row_num
    else:
        display_row = row_num
    parts = []
    if display_row not in (None, ""):
        parts.append(f"row {display_row}")
    for label, key in (("Date", "transaction_date"), ("Account", "account_number"), ("Account", "account_name"), ("Description", "description"), ("Amount", "amount"), ("Debit", "debit"), ("Credit", "credit")):
        val = row.get(key)
        if val not in (None, "", "<NA>"):
            parts.append(f"{label}: {val}")
    return "; ".join(parts)


def _enrich_flag_description(flag: Flag, evidence: AuditEvidence | None = None, fin_data: dict | None = None) -> str:
    desc = (flag.description or "").strip()
    if not evidence:
        return desc
    fin_data = fin_data or (evidence.document_specific or {}).get("_financial", {}) or {}
    rows = fin_data.get("rows") or []
    if flag.type == "duplicate_entry":
        # Find the exact source row when a duplicate-posting note is embedded in GL text.
        for row in rows:
            hay = " ".join(str(row.get(k, "")) for k in ("description", "account_name", "account_number", "transaction_date", "amount")).lower()
            if "duplicate" in hay:
                row_ctx = _format_financial_row_context(row, evidence.source_file)
                specific = f"Exact source row identified: {row_ctx}."
                if desc and specific.lower() not in desc.lower():
                    return f"{specific} {desc}".strip()
                return specific
    return desc


def _make_question(flag: Flag, rule: dict, is_financial: bool = False) -> Question:
    """Build a Question from a flag and its rule."""
    blocking = rule["blocking"]
    if is_financial and flag.type in _FINANCIAL_BLOCKING_OVERRIDES:
        blocking = _FINANCIAL_BLOCKING_OVERRIDES[flag.type]

    return Question(
        question_id   = f"{flag.type}_{uuid.uuid4().hex[:6]}",
        question_type = rule["question_type"],
        question_text = rule["question_text"],
        audience      = rule["audience"],
        blocking      = blocking,
        source_flag   = flag.type,
        resolved      = False,
    )


def build_action_queue(evidence_items: list[AuditEvidence]) -> list[dict]:
    """
    Flatten unresolved questions across files into a prioritised action queue.
    Delegates to workflow.build_prioritized_action_queue — blocking client items first,
    then blocking reviewer items, then non-blocking, then by file name.
    """
    from .workflow import build_prioritized_action_queue
    return build_prioritized_action_queue(evidence_items)





def _record_resolved_exception(evidence: AuditEvidence, question: Question, resolution: str) -> None:
    if not question.source_flag:
        return
    ds = evidence.document_specific or {}
    wf = dict(ds.get("_workflow") or {})
    resolved = list(wf.get("resolved_exceptions") or [])
    flag = next((f for f in (evidence.flags or []) if f.type == question.source_flag), None)
    entry = {
        "source_flag": question.source_flag,
        "question_type": question.question_type,
        "question_text": question.question_text,
        "flag_description": flag.description if flag else "",
        "severity": flag.severity if flag else "warning",
        "resolution": (resolution or "resolved").strip(),
        "resolution_type": question.resolution_type or "answer",
        "resolved_at": question.resolved_at or utc_now_iso(),
        "resolved_by": question.resolved_by or "reviewer",
    }
    resolved = [r for r in resolved if not (r.get("source_flag") == entry["source_flag"] and r.get("question_type") == entry["question_type"])]
    resolved.append(entry)
    wf["resolved_exceptions"] = resolved
    ds["_workflow"] = wf
    evidence.document_specific = ds


def _remove_flag(evidence: AuditEvidence, flag_type: str | None) -> None:
    if not flag_type:
        return
    evidence.flags = [f for f in (evidence.flags or []) if f.type != flag_type]


def _apply_resolution_side_effects(evidence: AuditEvidence, question: Question, resolution: str) -> None:
    ensure = (evidence.document_specific or {}).setdefault("_workflow", {})
    overrides = ensure.setdefault("field_overrides", {})
    answer = (resolution or "").strip()
    remove_flag = False
    if question.question_type == "period_confirmation" and answer:
        if not evidence.audit_overview:
            evidence.audit_overview = AuditOverview(summary=evidence.title or evidence.source_file, period=AuditPeriod())
        elif not evidence.audit_overview.period:
            evidence.audit_overview.period = AuditPeriod()
        fin = (evidence.document_specific or {}).setdefault("_financial", {})
        if len(answer) == 4 and answer.isdigit():
            evidence.audit_overview.period.effective_date = answer
            evidence.audit_overview.period.start = f"{answer}-01-01"
            evidence.audit_overview.period.end = f"{answer}-12-31"
            fin["period_start"] = evidence.audit_overview.period.start
            fin["period_end"] = evidence.audit_overview.period.end
            fin["fiscal_year"] = answer
            overrides["period_effective_date"] = answer
            overrides["financial_period_start"] = fin["period_start"]
            overrides["financial_period_end"] = fin["period_end"]
        elif " to " in answer:
            _start, _end = [x.strip() for x in answer.split(" to ", 1)]
            evidence.audit_overview.period.start = _start
            evidence.audit_overview.period.end = _end
            evidence.audit_overview.period.effective_date = f"{_start} to {_end}"
            fin["period_start"] = _start
            fin["period_end"] = _end
            overrides["period_effective_date"] = evidence.audit_overview.period.effective_date
            overrides["financial_period_start"] = _start
            overrides["financial_period_end"] = _end
        else:
            evidence.audit_overview.period.effective_date = answer
            fin["period_start"] = answer
            fin["period_end"] = answer
            overrides["period_effective_date"] = answer
            overrides["financial_period_start"] = answer
            overrides["financial_period_end"] = answer
        remove_flag = True
    elif question.question_type == "current_vs_prior_year_confirmation" and answer:
        normalized = answer.lower()
        fin = (evidence.document_specific or {}).setdefault("_financial", {})
        doc_type = ""
        if "prior" in normalized:
            doc_type = "trial_balance_prior_year"
        elif "current" in normalized:
            doc_type = "trial_balance_current"
        if doc_type:
            fin["doc_type"] = doc_type
            fin["finality_state"] = "user_confirmed"
            fin["doc_type_source"] = "user_override"
            overrides["financial_doc_type"] = doc_type
            overrides["financial_finality_state"] = "user_confirmed"
            evidence.subtype = doc_type
            overrides["subtype"] = evidence.subtype
            # Keep family/audit metadata in sync for user-confirmed trial balances.
            try:
                from .models import DocumentFamily
                evidence.family = DocumentFamily("accounting_report")
            except Exception:
                pass
            if not evidence.audit_overview:
                evidence.audit_overview = AuditOverview(summary=evidence.title or evidence.source_file, period=AuditPeriod())
            evidence.audit_overview.audit_areas = ["cash", "receivables", "prepaid"]
            evidence.audit_overview.assertions = ["accuracy", "existence", "rights_and_obligations", "classification"]
        remove_flag = True
    elif question.resolution_type in ("override", "dismissed", "reviewer_confirmed") or question.audience == "reviewer":
        remove_flag = True
    elif question.source_flag == "duplicate_entry" and answer:
        remove_flag = True
    elif answer:
        # For non-blocking client confirmations like attachment/agreement references,
        # a direct answer should close the exception instead of leaving it active.
        remove_flag = True

    if remove_flag and question.source_flag:
        _record_resolved_exception(evidence, question, answer or "resolved")
        _remove_flag(evidence, question.source_flag)


def _rebuild_readiness_with_history(evidence: AuditEvidence, resolved_history: list[Question]) -> ReadinessResult:
    new_rd = compute_readiness(evidence)
    unresolved = list(new_rd.questions or [])
    preserved = []
    seen = set()
    for q in resolved_history:
        key = (q.source_flag, q.question_type, q.resolution or "")
        if key in seen:
            continue
        seen.add(key)
        preserved.append(q)
    new_rd.questions = unresolved + preserved
    return new_rd

def resolve_question(
    evidence: AuditEvidence,
    question_id: str,
    resolution: str,
    *,
    actor: str = "reviewer",
    resolution_type: str = "answer",
    comment: str | None = None,
) -> AuditEvidence:
    """Resolve a question in-place, apply field updates, and recalculate readiness."""
    if not evidence.readiness:
        evidence.readiness = compute_readiness(evidence)

    resolved_question = None
    for q in evidence.readiness.questions or []:
        if q.question_id == question_id:
            q.resolved = True
            q.resolution = resolution
            q.status = "resolved" if resolution_type not in ("override", "dismissed", "superseded") else (
                "overridden" if resolution_type == "override" else resolution_type
            )
            q.resolution_type = resolution_type
            q.resolved_by = actor
            q.resolved_at = utc_now_iso()
            q.comments = comment or resolution
            record_question_event(evidence, q, actor=actor, action=q.status, comment=q.comments)
            resolved_question = q.model_copy(deep=True)
            _apply_resolution_side_effects(evidence, q, resolution)
            break

    resolved_history = [q.model_copy(deep=True) for q in (evidence.readiness.questions or []) if q.resolved]
    if resolved_question and all((rq.question_id != resolved_question.question_id) for rq in resolved_history):
        resolved_history.append(resolved_question)

    evidence.readiness = _rebuild_readiness_with_history(evidence, resolved_history)
    return evidence


def compute_readiness(evidence: AuditEvidence) -> ReadinessResult:
    """
    Compute the evidence readiness result from the current flag set.
    Returns a ReadinessResult; does not mutate the evidence object.
    """
    flags      = evidence.flags or []
    fin_data   = (evidence.document_specific or {}).get("_financial", {})
    # Refresh any file-specific flag descriptions so questions can show exact context.
    for _flag in flags:
        _flag.description = _enrich_flag_description(_flag, evidence=evidence, fin_data=fin_data)
    is_financial = bool(
        fin_data and fin_data.get("doc_type") and
        fin_data["doc_type"] != "not_financial_structured_data"
    )

    # Handle unusable / failed state
    meta = evidence.extraction_meta
    critical_flags = [f for f in flags if f.severity == "critical"]
    critical_without_workflow = [f for f in critical_flags if _FLAG_RULES.get(f.type) is None]
    if critical_without_workflow or (meta.total_chars < 50 and not is_financial):
        return ReadinessResult(
            readiness_status  = "unusable",
            blocking_state    = "blocking",
            blocking_issues   = [f.type for f in critical_without_workflow],
            evidence_use_mode = "unusable",
        )

    # Generate questions from flags
    questions: list[Question] = []
    seen_types: set[str] = set()  # one question per flag type

    for flag in flags:
        if flag.type in seen_types:
            continue
        rule = _FLAG_RULES.get(flag.type)
        if rule is None:
            continue   # info-only flag, no question
        seen_types.add(flag.type)
        questions.append(_make_question(flag, rule, is_financial))

    # Blocking issues = questions that are marked blocking and not yet resolved
    blocking_questions = [q for q in questions if q.blocking and not q.resolved]
    blocking_issues    = [q.source_flag for q in blocking_questions if q.source_flag]

    # Determine readiness status
    reviewer_blocking = [q for q in blocking_questions if q.audience == "reviewer"]
    client_blocking   = [q for q in blocking_questions if q.audience == "client"]
    any_non_blocking  = [q for q in questions if not q.blocking]

    if not questions:
        readiness_status = "ready"
    elif blocking_questions:
        if reviewer_blocking:
            readiness_status = "needs_reviewer_confirmation"
        else:
            readiness_status = "needs_client_answer"
    elif any_non_blocking:
        readiness_status = "exception_open"
    else:
        readiness_status = "ready"

    blocking_state = "blocking" if blocking_questions else "non_blocking"

    # Financial population readiness
    population_ready  = None
    population_status = None
    evidence_use_mode = "evidence_and_population"

    if is_financial:
        finality   = fin_data.get("finality_state", "")
        bal        = fin_data.get("balance_check", {})
        bal_flag   = bal.get("flag_level", "")
        has_rows   = bool(fin_data.get("rows"))
        has_period = bool(fin_data.get("period_start"))
        row_diag   = fin_data.get("row_diagnostics") or {}

        pop_blockers = []
        if finality not in ("trusted", "user_confirmed"):
            pop_blockers.append("classification not confirmed")
        if not has_period:
            pop_blockers.append("period not detected")
        if bal_flag == "material_balance_difference":
            pop_blockers.append("material balance difference unresolved")
        if not has_rows:
            pop_blockers.append("row-level data not yet retained")

        diag_reasons = row_diag.get("blocking_reasons") or []
        pop_blockers.extend(diag_reasons)

        population_ready  = len(pop_blockers) == 0
        population_status = "; ".join(dict.fromkeys(pop_blockers)) if pop_blockers else "ready"

        if not population_ready:
            evidence_use_mode = "evidence_only"

    return ReadinessResult(
        readiness_status  = readiness_status,
        blocking_state    = blocking_state,
        blocking_issues   = blocking_issues,
        questions         = questions,
        population_ready  = population_ready,
        population_status = population_status,
        evidence_use_mode = evidence_use_mode,
    )


def apply_readiness(evidence: AuditEvidence) -> AuditEvidence:
    """
    Compute readiness and attach it to the evidence object.
    Returns the same object (mutated in place).
    """
    evidence.readiness = compute_readiness(evidence)
    return evidence
