from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AuditEvidence, Question

WORKFLOW_STATE_VERSION = "v05.1"
STATE_PATH = Path(__file__).resolve().parent.parent / ".workflow_state.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_bytes_signature(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def load_state(path: Path | None = None) -> dict[str, Any]:
    path = path or STATE_PATH
    if not path.exists():
        return {"version": WORKFLOW_STATE_VERSION, "files": {}, "lineage": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("workflow state must be an object")
    except Exception:
        return {"version": WORKFLOW_STATE_VERSION, "files": {}, "lineage": {}}
    data.setdefault("version", WORKFLOW_STATE_VERSION)
    data.setdefault("files", {})
    data.setdefault("lineage", {})
    return data


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = path or STATE_PATH
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _question_to_dict(q: Question) -> dict[str, Any]:
    return {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "question_text": q.question_text,
        "audience": q.audience,
        "blocking": q.blocking,
        "source_flag": q.source_flag,
        "resolved": q.resolved,
        "resolution": q.resolution,
        "status": q.status,
        "resolution_type": q.resolution_type,
        "resolved_by": q.resolved_by,
        "resolved_at": q.resolved_at,
        "comments": q.comments,
    }


def ensure_workflow_metadata(evidence: AuditEvidence, file_signature: str | None = None) -> AuditEvidence:
    ds = evidence.document_specific or {}
    wf = dict(ds.get("_workflow") or {})
    if file_signature and not wf.get("file_signature"):
        wf["file_signature"] = file_signature
    wf.setdefault("version_id", wf.get("file_signature") or evidence.source_file)
    wf.setdefault("source_file", evidence.source_file)
    wf.setdefault("created_at", utc_now_iso())
    wf.setdefault("lineage", {})
    wf.setdefault("question_history", [])
    wf.setdefault("field_overrides", {})
    wf.setdefault("resolved_exceptions", [])
    ds["_workflow"] = wf
    evidence.document_specific = ds
    return evidence


def merge_state_into_evidence(evidence: AuditEvidence, file_signature: str | None = None, state: dict[str, Any] | None = None) -> AuditEvidence:
    state = state or load_state()
    ensure_workflow_metadata(evidence, file_signature=file_signature)
    wf = evidence.document_specific.get("_workflow", {})
    sig = wf.get("file_signature") or file_signature or evidence.source_file
    saved = state.get("files", {}).get(sig)
    if not saved:
        return evidence
    q_state_by_key: dict[tuple[str | None, str], dict[str, Any]] = {}
    for item in saved.get("questions", []):
        q_state_by_key[(item.get("source_flag"), item.get("question_type"))] = item
    if evidence.readiness:
        for q in evidence.readiness.questions or []:
            item = q_state_by_key.get((q.source_flag, q.question_type))
            if not item:
                continue
            q.resolved = bool(item.get("resolved"))
            q.resolution = item.get("resolution")
            q.status = item.get("status") or ("resolved" if q.resolved else "open")
            q.resolution_type = item.get("resolution_type")
            q.resolved_by = item.get("resolved_by")
            q.resolved_at = item.get("resolved_at")
            q.comments = item.get("comments")
    wf.setdefault("question_history", saved.get("question_history", []))
    wf.setdefault("lineage", saved.get("lineage", {}))
    wf.setdefault("resolved_exceptions", saved.get("resolved_exceptions", []))
    overrides = saved.get("field_overrides", {}) or wf.get("field_overrides", {})
    wf["field_overrides"] = overrides
    evidence.document_specific["_workflow"] = wf
    _apply_field_overrides(evidence, overrides)
    if overrides and evidence.readiness:
        from .readiness import compute_readiness
        resolved_history = []
        for item in saved.get("questions", []):
            if item.get("resolved"):
                try:
                    resolved_history.append(Question(**item))
                except Exception:
                    pass
        new_rd = compute_readiness(evidence)
        unresolved = list(new_rd.questions or [])
        for rq in resolved_history:
            unresolved.append(rq)
        new_rd.questions = unresolved
        evidence.readiness = new_rd
    return evidence




def _apply_field_overrides(evidence: AuditEvidence, overrides: dict[str, Any]) -> AuditEvidence:
    if not overrides:
        return evidence
    if overrides.get("period_effective_date"):
        if not evidence.audit_overview:
            from .models import AuditOverview, AuditPeriod
            evidence.audit_overview = AuditOverview(summary=evidence.title or evidence.source_file, period=AuditPeriod())
        elif not evidence.audit_overview.period:
            from .models import AuditPeriod
            evidence.audit_overview.period = AuditPeriod()
        evidence.audit_overview.period.effective_date = overrides.get("period_effective_date")
        evidence.flags = [f for f in (evidence.flags or []) if f.type != "missing_period"]
    fin = (evidence.document_specific or {}).setdefault("_financial", {})
    if overrides.get("financial_period_start"):
        fin["period_start"] = overrides.get("financial_period_start")
    if overrides.get("financial_period_end"):
        fin["period_end"] = overrides.get("financial_period_end")
    if overrides.get("financial_finality_state"):
        fin["finality_state"] = overrides.get("financial_finality_state")
        evidence.flags = [f for f in (evidence.flags or []) if f.type != "tb_year_unconfirmed"]
    if overrides.get("financial_doc_type"):
        fin["doc_type"] = overrides.get("financial_doc_type")
        fin["doc_type_source"] = "user_override"
    if overrides.get("subtype"):
        evidence.subtype = overrides.get("subtype")
    return evidence

def persist_evidence_state(evidence: AuditEvidence, state: dict[str, Any] | None = None, path: Path | None = None) -> dict[str, Any]:
    state = deepcopy(state or load_state(path))
    ensure_workflow_metadata(evidence)
    wf = evidence.document_specific.get("_workflow", {})
    sig = wf.get("file_signature") or evidence.source_file
    state.setdefault("files", {})[sig] = {
        "source_file": evidence.source_file,
        "file_signature": sig,
        "updated_at": utc_now_iso(),
        "questions": [_question_to_dict(q) for q in (evidence.readiness.questions if evidence.readiness else [])],
        "question_history": wf.get("question_history", []),
        "lineage": wf.get("lineage", {}),
        "field_overrides": wf.get("field_overrides", {}),
        "resolved_exceptions": wf.get("resolved_exceptions", []),
    }
    save_state(state, path)
    return state


def register_lineage(current: AuditEvidence, prior: AuditEvidence | None, state: dict[str, Any] | None = None, path: Path | None = None) -> dict[str, Any]:
    state = deepcopy(state or load_state(path))
    ensure_workflow_metadata(current)
    current_wf = current.document_specific.get("_workflow", {})
    curr_sig = current_wf.get("file_signature") or current.source_file
    state.setdefault("lineage", {}).setdefault(current.source_file, [])
    lineage_list = state["lineage"][current.source_file]
    if curr_sig not in lineage_list:
        lineage_list.append(curr_sig)
    if prior:
        ensure_workflow_metadata(prior)
        prior_sig = prior.document_specific.get("_workflow", {}).get("file_signature") or prior.source_file
        current_wf.setdefault("lineage", {})["replaces"] = prior_sig
        state.setdefault("files", {}).setdefault(prior_sig, {"source_file": prior.source_file, "file_signature": prior_sig})
        state["files"][prior_sig].setdefault("lineage", {})["superseded_by"] = curr_sig
        state["files"][prior_sig]["updated_at"] = utc_now_iso()
    current.document_specific["_workflow"] = current_wf
    persist_evidence_state(current, state=state, path=path)
    return load_state(path)


def record_question_event(evidence: AuditEvidence, question: Question, actor: str, action: str, comment: str | None = None) -> None:
    ensure_workflow_metadata(evidence)
    wf = evidence.document_specific.get("_workflow", {})
    history = list(wf.get("question_history", []))
    history.append({
        "timestamp": utc_now_iso(),
        "actor": actor,
        "action": action,
        "question_type": question.question_type,
        "source_flag": question.source_flag,
        "question_text": question.question_text,
        "comment": comment or question.resolution or "",
    })
    wf["question_history"] = history
    evidence.document_specific["_workflow"] = wf


def _flag_description_for_question(ev: AuditEvidence, q: Question) -> str:
    for flag in ev.flags or []:
        if flag.type == q.source_flag:
            return flag.description or ""
    return ""


def _active_question_count(ev: AuditEvidence) -> int:
    rd = ev.readiness
    if not rd:
        return 0
    return sum(1 for q in (rd.questions or []) if not q.resolved)


def build_prioritized_action_queue(evidence_items: list[AuditEvidence]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for ev in evidence_items:
        rd = ev.readiness
        if not rd:
            continue
        for q in rd.questions or []:
            if q.resolved:
                continue
            queue.append({
                "source_file": ev.source_file,
                "question_id": q.question_id,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "audience": q.audience,
                "blocking": q.blocking,
                "readiness_status": rd.readiness_status,
                "source_flag": q.source_flag,
                "flag_description": _flag_description_for_question(ev, q),
                "priority_label": "Blocking" if q.blocking else "Review",
            })
    queue.sort(key=lambda item: (
        not item["blocking"],
        item["audience"] != "client",
        item["audience"] != "reviewer",
        item["source_file"].lower(),
        item["question_text"],
    ))
    return queue


def next_best_question(evidence_items: list[AuditEvidence]) -> dict[str, Any] | None:
    queue = build_prioritized_action_queue(evidence_items)
    return queue[0] if queue else None


def build_client_followup_package(evidence_items: list[AuditEvidence]) -> list[dict[str, Any]]:
    package: list[dict[str, Any]] = []
    for ev in evidence_items:
        rd = ev.readiness
        if not rd:
            continue
        client_questions = [q for q in rd.questions or [] if q.audience == "client" and not q.resolved]
        if not client_questions:
            continue
        package.append({
            "source_file": ev.source_file,
            "request_count": len(client_questions),
            "blocking_count": sum(1 for q in client_questions if q.blocking),
            "requests": [
                {
                    "question_text": q.question_text,
                    "flag_description": _flag_description_for_question(ev, q),
                    "question_id": q.question_id,
                    "source_flag": q.source_flag,
                }
                for q in client_questions
            ],
        })
    package.sort(key=lambda item: (-item["blocking_count"], -item["request_count"], item["source_file"].lower()))
    return package
