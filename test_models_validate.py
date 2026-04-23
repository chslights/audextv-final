import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_ingestion.models import AuditEvidence, ExtractionMeta, Flag
from audit_ingestion.readiness import compute_readiness, resolve_question
from audit_ingestion.workflow import (
    build_client_followup_package,
    build_prioritized_action_queue,
    merge_state_into_evidence,
    persist_evidence_state,
)


def make_evidence(*flags, source_file="test.pdf"):
    ev = AuditEvidence(
        source_file=source_file,
        flags=[Flag(type=f, description=f, severity="warning") for f in flags],
        extraction_meta=ExtractionMeta(primary_extractor="pdfplumber", total_chars=500),
    )
    ev.readiness = compute_readiness(ev)
    return ev


def test_resolve_question_captures_workflow_metadata(tmp_path):
    ev = make_evidence("missing_period")
    q = ev.readiness.questions[0]
    resolve_question(ev, q.question_id, "FY24", actor="alice", resolution_type="override", comment="reviewed")
    assert q.resolved is True
    assert q.status == "overridden"
    assert q.resolved_by == "alice"
    assert q.comments == "reviewed"
    history = ev.document_specific["_workflow"]["question_history"]
    assert history[-1]["actor"] == "alice"
    assert history[-1]["action"] == "overridden"


def test_persist_and_restore_question_state(tmp_path):
    path = tmp_path / "workflow.json"
    ev = make_evidence("missing_period")
    q = ev.readiness.questions[0]
    resolve_question(ev, q.question_id, "2024", actor="reviewer")
    persist_evidence_state(ev, path=path)

    ev2 = make_evidence("missing_period")
    merge_state_into_evidence(ev2, state=None if False else __import__('audit_ingestion.workflow', fromlist=['load_state']).load_state(path))
    q2 = ev2.readiness.questions[0]
    assert q2.resolved is True
    assert q2.resolution == "2024"


def test_prioritized_queue_and_client_package():
    ev1 = make_evidence("missing_period", source_file="a.pdf")
    ev2 = make_evidence("tb_year_unconfirmed", source_file="b.pdf")
    ev3 = make_evidence("related_party", source_file="c.pdf")
    queue = build_prioritized_action_queue([ev1, ev2, ev3])
    assert queue[0]["audience"] == "client"
    package = build_client_followup_package([ev1, ev2, ev3])
    assert len(package) == 1
    assert package[0]["source_file"] == "a.pdf"
