from __future__ import annotations

import inspect

from audit_ingestion.models import AuditEvidence, DocumentFamily
from audit_ingestion.router import _should_auto_vision_for_missing_fields, ingest_one


def _ev(family: DocumentFamily) -> AuditEvidence:
    return AuditEvidence(source_file='x.pdf', family=family)


def test_auto_missing_fields_allowlist_true_for_financial_families():
    allowed = [
        DocumentFamily.CONTRACT,
        DocumentFamily.INVOICE,
        DocumentFamily.PAYMENT,
        DocumentFamily.BANK,
        DocumentFamily.PAYROLL,
        DocumentFamily.ACCOUNTING,
        DocumentFamily.GRANT,
        DocumentFamily.TAX_REG,
        DocumentFamily.SCHEDULE,
    ]
    for fam in allowed:
        assert _should_auto_vision_for_missing_fields(_ev(fam)) is True, fam


def test_auto_missing_fields_allowlist_false_for_other_family():
    assert _should_auto_vision_for_missing_fields(_ev(DocumentFamily.OTHER)) is False


def test_auto_missing_fields_allowlist_false_for_correspondence_family():
    assert _should_auto_vision_for_missing_fields(_ev(DocumentFamily.CORRESPONDENCE)) is False


def test_router_uses_shared_apply_vision_result_for_auto_and_manual_paths():
    src = inspect.getsource(ingest_one)
    assert 'def _apply_vision_result(' in src
    assert '_apply_vision_result(_auto_result, "auto_missing_fields")' in src
    # manual path uses the selected vision_mode variable
    assert '_apply_vision_result(_vision_result, vision_mode)' in src


def test_auto_missing_fields_requires_nonempty_vision_output_before_apply():
    src = inspect.getsource(ingest_one)
    assert '_auto_result.get("chars_returned", 0) > 0' in src


def test_ingest_app_surfaces_no_usable_vision_state():
    from pathlib import Path
    src = (Path(__file__).parent.parent / 'ingest_app.py').read_text()
    assert 'Vision attempted — no usable rescue text returned' in src


def test_ingest_app_uses_targeted_metrics_for_auto_missing_fields():
    from pathlib import Path
    src = (Path(__file__).parent.parent / 'ingest_app.py').read_text()
    assert 'if _vmode == "auto_missing_fields"' in src
    assert 'Targeted rescue mode' in src
    assert 'Vision pages' in src
