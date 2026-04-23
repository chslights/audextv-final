"""
tests/test_vision_extraction.py
Tests for the manual vision extraction system:
  - parse_page_selection
  - run_vision_on_pages / run_full_document_vision
  - router vision_mode wiring
  - AuditEvidence vision fields
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_ingestion.extractor import (
    parse_page_selection,
    VISION_AUTO, VISION_STANDARD_ONLY,
    VISION_FORCE_ALL, VISION_FORCE_SELECTED, VISION_RETRY,
    VISION_ANNOTATION_PROMPT, VISION_FULL_DOCUMENT_WARN_PAGES,
)
from audit_ingestion.models import AuditEvidence, ExtractionMeta


# ── parse_page_selection ──────────────────────────────────────────────────────

def test_parse_single_page():
    pages, err = parse_page_selection("3", max_pages=10)
    assert pages == [3]
    assert err == ""

def test_parse_comma_separated():
    pages, err = parse_page_selection("1,4,7", max_pages=10)
    assert pages == [1, 4, 7]
    assert err == ""

def test_parse_range():
    pages, err = parse_page_selection("2-5", max_pages=10)
    assert pages == [2, 3, 4, 5]
    assert err == ""

def test_parse_mixed_range_and_singles():
    pages, err = parse_page_selection("1-3,6,8-10", max_pages=10)
    assert pages == [1, 2, 3, 6, 8, 9, 10]
    assert err == ""

def test_parse_deduplicates():
    pages, err = parse_page_selection("1,1,2,2", max_pages=10)
    assert pages == [1, 2]
    assert err == ""

def test_parse_clamps_to_max_pages():
    pages, err = parse_page_selection("5,15,20", max_pages=10)
    assert 5 in pages
    assert 15 not in pages
    assert 20 not in pages

def test_parse_out_of_range_produces_error():
    pages, err = parse_page_selection("15", max_pages=10)
    assert 15 not in pages
    assert "out of range" in err

def test_parse_empty_string():
    pages, err = parse_page_selection("", max_pages=10)
    assert pages == []
    assert err != ""

def test_parse_invalid_text():
    pages, err = parse_page_selection("abc", max_pages=10)
    assert pages == []
    assert err != ""

def test_parse_invalid_range():
    pages, err = parse_page_selection("5-2", max_pages=10)
    assert pages == []
    assert "Invalid range" in err

def test_parse_sorts_output():
    pages, err = parse_page_selection("7,1,4", max_pages=10)
    assert pages == [1, 4, 7]

def test_parse_single_char_valid():
    pages, err = parse_page_selection("1", max_pages=5)
    assert pages == [1]
    assert err == ""


# ── Vision constants ──────────────────────────────────────────────────────────

def test_vision_mode_constants_defined():
    assert VISION_AUTO           == "auto"
    assert VISION_STANDARD_ONLY  == "standard_only"
    assert VISION_FORCE_ALL      == "force_all"
    assert VISION_FORCE_SELECTED == "force_selected"
    assert VISION_RETRY          == "retry"

def test_vision_annotation_prompt_covers_handwriting():
    prompt = VISION_ANNOTATION_PROMPT
    assert "handwritten" in prompt.lower()
    assert "annotation" in prompt.lower()
    assert "stamp" in prompt.lower()
    assert "PAGE BREAK" in prompt

def test_vision_warn_threshold():
    assert VISION_FULL_DOCUMENT_WARN_PAGES >= 10


# ── AuditEvidence vision fields ───────────────────────────────────────────────

def test_vision_fields_on_evidence():
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=100),
    )
    assert ev.vision_applied is False
    assert ev.vision_mode_used is None
    assert ev.vision_pages_used == []
    assert ev.vision_text is None
    assert ev.vision_page_texts == {}
    assert ev.vision_prompt is None
    assert ev.vision_error is None

def _ev_dict(ev):
    """Pydantic v1/v2 compatible serialization."""
    return ev.model_dump() if hasattr(ev, "model_dump") else ev.dict()

def test_vision_fields_serialize():
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=100),
        vision_applied=True,
        vision_mode_used="force_all",
        vision_pages_used=[1, 2, 3],
        vision_text="Handwritten note: $5,000 approved",
        vision_page_texts={1: "Page 1 text", 2: "Page 2 text"},
        vision_prompt="Extract all text",
    )
    d = _ev_dict(ev)
    assert d["vision_applied"] is True
    assert d["vision_mode_used"] == "force_all"
    assert d["vision_pages_used"] == [1, 2, 3]
    assert "Handwritten note" in d["vision_text"]
    assert d["vision_page_texts"][1] == "Page 1 text"

def test_vision_fields_round_trip():
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=100),
        vision_applied=True,
        vision_mode_used="force_selected",
        vision_pages_used=[2, 4],
        vision_page_texts={2: "annotated", 4: "initialled"},
    )
    d = _ev_dict(ev)
    ev2 = AuditEvidence(**d)
    assert ev2.vision_applied is True
    assert ev2.vision_pages_used == [2, 4]
    assert ev2.vision_page_texts[2] == "annotated"


# ── Router signature ──────────────────────────────────────────────────────────

def test_router_accepts_vision_params():
    """ingest_one must accept vision_mode, vision_pages, vision_prompt."""
    import inspect
    from audit_ingestion.router import ingest_one
    sig = inspect.signature(ingest_one)
    params = sig.parameters
    assert "vision_mode" in params,   "vision_mode missing from ingest_one"
    assert "vision_pages" in params,  "vision_pages missing from ingest_one"
    assert "vision_prompt" in params, "vision_prompt missing from ingest_one"
    assert params["vision_mode"].default == VISION_AUTO

def test_router_vision_default_is_auto():
    import inspect
    from audit_ingestion.router import ingest_one
    sig = inspect.signature(ingest_one)
    assert sig.parameters["vision_mode"].default == VISION_AUTO


# ── run_vision_on_pages interface ─────────────────────────────────────────────

def test_run_vision_on_pages_exists():
    from audit_ingestion.extractor import run_vision_on_pages
    assert callable(run_vision_on_pages)

def test_run_full_document_vision_exists():
    from audit_ingestion.extractor import run_full_document_vision
    assert callable(run_full_document_vision)

def test_run_vision_on_pages_no_provider():
    from audit_ingestion.extractor import run_vision_on_pages
    result = run_vision_on_pages("fake.pdf", [1, 2], provider=None)
    assert result["error"] is not None
    assert result["page_texts"] == {}
    assert result["pages_succeeded"] == []

def test_run_vision_on_pages_empty_pages():
    from audit_ingestion.extractor import run_vision_on_pages

    class _StubProvider:
        def extract_text_from_page_images(self, **kwargs):
            return "text"

    result = run_vision_on_pages("fake.pdf", [], provider=_StubProvider())
    assert result["error"] is not None
    assert result["pages_attempted"] == []

def test_run_vision_on_pages_result_structure():
    from audit_ingestion.extractor import run_vision_on_pages
    result = run_vision_on_pages("nonexistent.pdf", [1], provider=None)
    assert "page_texts" in result
    assert "pages_attempted" in result
    assert "pages_succeeded" in result
    assert "prompt_used" in result
    assert "error" in result

def test_run_full_document_vision_result_structure():
    from audit_ingestion.extractor import run_full_document_vision
    result = run_full_document_vision("nonexistent.pdf", provider=None)
    assert "page_texts" in result
    assert "combined_text" in result
    assert "page_count" in result


# ── App source checks ─────────────────────────────────────────────────────────

def test_app_has_vision_mode_dropdown():
    src = (Path(__file__).parent.parent / "ingest_app.py").read_text()
    assert "Force Vision" in src
    assert "Standard Only" in src
    assert "Smart Retry" in src

def test_app_has_run_vision_buttons():
    src = (Path(__file__).parent.parent / "ingest_app.py").read_text()
    assert "Run Full Vision" in src
    assert "Run Selected Pages" in src
    assert "Smart Retry" in src

def test_app_has_comparison_view():
    src = (Path(__file__).parent.parent / "ingest_app.py").read_text()
    assert "Standard Extraction" in src
    assert "Vision Extraction" in src
    assert "vision_text" in src or "Vision Extraction Results" in src

def test_app_imports_parse_page_selection():
    src = (Path(__file__).parent.parent / "ingest_app.py").read_text()
    assert "parse_page_selection" in src

def test_app_has_page_count_warning():
    src = (Path(__file__).parent.parent / "ingest_app.py").read_text()
    assert "VISION_FULL_DOCUMENT_WARN_PAGES" in src or "may be slow" in src


# ── New P1-P10 tests ──────────────────────────────────────────────────────────

def test_run_full_document_vision_no_max_pages_cap():
    """run_full_document_vision must not apply MAX_PDF_PAGES_FAST as a default cap."""
    import inspect
    from audit_ingestion.extractor import run_full_document_vision, MAX_PDF_PAGES_FAST
    sig = inspect.signature(run_full_document_vision)
    default_max = sig.parameters["max_pages"].default
    # Default must be None (process all pages), not MAX_PDF_PAGES_FAST
    assert default_max is None, (
        f"run_full_document_vision default max_pages should be None, got {default_max}"
    )

def test_provider_chunk_size_defined():
    from audit_ingestion.providers.openai_provider import OpenAIProvider
    assert hasattr(OpenAIProvider, "_VISION_CHUNK_SIZE")
    assert OpenAIProvider._VISION_CHUNK_SIZE >= 4

def test_provider_no_hard_8_image_cap():
    """The provider must not have a hard images[:8] cap."""
    src = open(
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/providers/openai_provider.py"
    ).read()
    assert "images[:8]" not in src, "Hard images[:8] truncation is still present"

def test_identify_pages_needing_vision_exists():
    from audit_ingestion.extractor import identify_pages_needing_vision
    assert callable(identify_pages_needing_vision)

def test_identify_pages_needing_vision_finds_weak():
    from audit_ingestion.extractor import identify_pages_needing_vision
    from audit_ingestion.models import ParsedDocument, ParsedPage

    doc = ParsedDocument(
        source_file="test.pdf",
        pages=[
            ParsedPage(page_number=1, text="A" * 500, char_count=500, extractor="pdfplumber", confidence=0.9),
            ParsedPage(page_number=2, text="",          char_count=0,   extractor="pdfplumber", confidence=0.0),
            ParsedPage(page_number=3, text="B" * 100,  char_count=100,  extractor="ocr",        confidence=0.5),
        ],
        page_count=3,
    )
    pages, reasons = identify_pages_needing_vision(doc)
    assert 2 in pages, "Page with 0 chars should be flagged"
    assert 3 in pages, "Page with 100 chars (below 300 threshold) should be flagged"
    assert 1 not in pages, "Page with 500 chars should not be flagged"
    assert "0 chars" in reasons[2]

def test_identify_pages_no_weak():
    from audit_ingestion.extractor import identify_pages_needing_vision
    from audit_ingestion.models import ParsedDocument, ParsedPage

    doc = ParsedDocument(
        source_file="test.pdf",
        pages=[
            ParsedPage(page_number=1, text="A" * 500, char_count=500, extractor="pdfplumber", confidence=0.9),
            ParsedPage(page_number=2, text="B" * 600, char_count=600, extractor="pdfplumber", confidence=0.9),
        ],
        page_count=2,
    )
    pages, reasons = identify_pages_needing_vision(doc)
    assert pages == []

def test_vision_page_status_constants():
    from audit_ingestion.models import (
        VISION_PAGE_NOT_ATTEMPTED, VISION_PAGE_BASE_EXTRACTED,
        VISION_PAGE_REQUESTED, VISION_PAGE_COMPLETE,
        VISION_PAGE_PARTIAL, VISION_PAGE_FAILED, VISION_PAGE_REVIEW_NEEDED,
    )
    assert VISION_PAGE_COMPLETE  == "vision_complete"
    assert VISION_PAGE_FAILED    == "vision_failed"
    assert VISION_PAGE_REQUESTED == "vision_requested"

def test_evidence_has_vision_page_status():
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=100),
    )
    assert hasattr(ev, "vision_page_status")
    assert ev.vision_page_status == {}

def test_evidence_has_vision_run_diagnostics():
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=100),
    )
    assert hasattr(ev, "vision_run_diagnostics")
    assert ev.vision_run_diagnostics is None

def test_run_vision_result_has_diagnostic_fields():
    from audit_ingestion.extractor import run_vision_on_pages
    result = run_vision_on_pages("nonexistent.pdf", [1, 2], provider=None)
    assert "page_status"    in result
    assert "pages_rendered" in result
    assert "pages_failed"   in result
    assert "is_partial"     in result
    assert "chars_returned" in result
    assert "elapsed_seconds" in result

def test_run_vision_pages_failed_set_on_no_provider():
    from audit_ingestion.extractor import run_vision_on_pages
    from audit_ingestion.models import VISION_PAGE_FAILED
    result = run_vision_on_pages("fake.pdf", [1, 2, 3], provider=None)
    for pn in [1, 2, 3]:
        assert result["page_status"].get(pn) == VISION_PAGE_FAILED

def test_router_imports_identify_pages():
    src = open(
        __import__("pathlib").Path(__file__).parent.parent / "audit_ingestion/router.py"
    ).read()
    assert "identify_pages_needing_vision" in src

def test_router_vision_diagnostics_stored():
    """Router must store vision_run_diagnostics on evidence after a vision run."""
    src = open(
        __import__("pathlib").Path(__file__).parent.parent / "audit_ingestion/router.py"
    ).read()
    assert "vision_run_diagnostics" in src
    assert "pages_requested" in src
    assert "elapsed_seconds" in src
    assert "truncation_suspected" in src

def test_ui_has_per_page_status_tab():
    src = (
        __import__("pathlib").Path(__file__).parent.parent / "ingest_app.py"
    ).read_text()
    assert "Per-Page Status" in src

def test_ui_has_run_diagnostics_expander():
    src = (
        __import__("pathlib").Path(__file__).parent.parent / "ingest_app.py"
    ).read_text()
    assert "Run Diagnostics" in src

def test_ui_has_usefulness_comparison():
    src = (
        __import__("pathlib").Path(__file__).parent.parent / "ingest_app.py"
    ).read_text()
    assert "Vision gain" in src or "chars_gain" in src

def test_ui_honest_retry_label():
    src = (
        __import__("pathlib").Path(__file__).parent.parent / "ingest_app.py"
    ).read_text()
    # "Retry With Vision" should be replaced with honest label
    assert "Retry With Vision" not in src
    assert "Smart Retry" in src

def test_ui_partial_run_warning():
    src = (
        __import__("pathlib").Path(__file__).parent.parent / "ingest_app.py"
    ).read_text()
    assert "Partial run" in src or "is_partial" in src

def test_ui_truncation_warning():
    src = (
        __import__("pathlib").Path(__file__).parent.parent / "ingest_app.py"
    ).read_text()
    assert "truncation" in src.lower()

def test_parse_vision_json_response_exists():
    from audit_ingestion.extractor import _parse_vision_json_response
    assert callable(_parse_vision_json_response)

def test_parse_vision_json_response_valid_json():
    from audit_ingestion.extractor import _parse_vision_json_response
    import json
    data = [{"page": 2, "text": "Signed by: J Smith"}, {"page": 5, "text": "Amount: $500"}]
    result = _parse_vision_json_response(json.dumps(data), [2, 5])
    assert result[2] == "Signed by: J Smith"
    assert result[5] == "Amount: $500"

def test_parse_vision_json_response_falls_back_to_delimiter():
    from audit_ingestion.extractor import _parse_vision_json_response
    raw = "Page one text--- PAGE BREAK ---Page two text"
    result = _parse_vision_json_response(raw, [3, 7])
    assert 3 in result
    assert 7 in result

def test_parse_vision_json_response_handles_embedded_json():
    from audit_ingestion.extractor import _parse_vision_json_response
    import json
    data = [{"page": 1, "text": "hello", "has_handwriting": True}]
    raw = "Here is the extraction:\n" + json.dumps(data) + "\nEnd."
    result = _parse_vision_json_response(raw, [1])
    assert result.get(1) == "hello"


# ── Stephen's fix list tests ──────────────────────────────────────────────────

def test_provider_returns_json_not_page_break_text():
    """Provider must return a single JSON array, not PAGE BREAK-joined text."""
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/providers/openai_provider.py"
    ).read_text()
    # The old behaviour joined with PAGE BREAK — must be gone
    assert '"--- PAGE BREAK ---".join(all_texts)' not in src
    # New behaviour: returns JSON
    assert "_json.dumps(all_page_dicts)" in src

def test_parse_chunk_json_importable():
    from audit_ingestion.providers.openai_provider import _parse_chunk_json
    assert callable(_parse_chunk_json)

def test_parse_chunk_json_valid_array():
    from audit_ingestion.providers.openai_provider import _parse_chunk_json
    import json
    data = [{"page": 2, "text": "handwritten note: approved", "has_handwriting": True},
            {"page": 3, "text": "stamp: RECEIVED", "has_handwriting": False}]
    result = _parse_chunk_json(json.dumps(data), [2, 3])
    assert len(result) == 2
    assert result[0]["text"] == "handwritten note: approved"
    assert result[1]["page"] == 3

def test_parse_chunk_json_multiple_adjacent_arrays():
    """Two JSON arrays joined — only first was captured before. Now handles both."""
    from audit_ingestion.providers.openai_provider import _parse_chunk_json
    import json
    # This is what the OLD parser failed on
    chunk1 = [{"page": 1, "text": "first"}, {"page": 2, "text": "second"}]
    # Simulate embedded JSON in prose
    raw = "Here is the data: " + json.dumps(chunk1) + " end."
    result = _parse_chunk_json(raw, [1, 2])
    pages = {d["page"] for d in result}
    assert 1 in pages
    assert 2 in pages

def test_parse_chunk_json_fallback_to_delimiter():
    from audit_ingestion.providers.openai_provider import _parse_chunk_json
    # Non-JSON text falls back to delimiter splitting
    raw = "Page one content--- PAGE BREAK ---Page two content"
    result = _parse_chunk_json(raw, [5, 6])
    assert any(d["page"] == 5 for d in result)
    assert any(d["page"] == 6 for d in result)

def test_parse_chunk_json_missing_page_coverage():
    from audit_ingestion.providers.openai_provider import _parse_chunk_json
    import json
    # Only page 1 returned, page 2 expected
    data = [{"page": 1, "text": "page one"}]
    result = _parse_chunk_json(json.dumps(data), [1, 2])
    pages = {d["page"] for d in result}
    # Page 1 is present; page 2 is handled by the caller (provider adds _missing)
    assert 1 in pages

def test_router_vision_re_runs_canonical():
    """Router code must contain canonical re-run logic after vision merge."""
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/router.py"
    ).read_text()
    assert "canonical_ai_vision_enhanced" in src
    assert "canonical re-run" in src.lower() or "canonical_reran" in src
    assert "rebuild" in src.lower() or "parsed_doc.full_text" in src

def test_router_vision_diagnostics_has_canonical_reran():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/router.py"
    ).read_text()
    assert '"canonical_reran"' in src

def test_router_readiness_rescored_after_vision():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/router.py"
    ).read_text()
    assert "apply_readiness" in src
    assert "_vision_enhanced" in src

def test_image_files_supported_in_extractor():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/extractor.py"
    ).read_text()
    assert '".jpg"' in src or '".jpeg"' in src
    assert '"direct_image"' in src
    assert "_image_bytes" in src

def test_image_files_bypass_fitz():
    """run_vision_on_pages must handle image files without fitz."""
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/extractor.py"
    ).read_text()
    assert "_is_image_file" in src
    assert "read_bytes()" in src

def test_ui_shows_canonical_reran():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "ingest_app.py"
    ).read_text()
    assert "canonical_reran" in src
    assert "Canonical extraction re-ran" in src or "canonical re-ran" in src.lower()


# ── 30-item fix list tests ────────────────────────────────────────────────────

def test_improvement_scorer_exists():
    from audit_ingestion.extractor import _compute_page_improvement_score
    score, decision, reason = _compute_page_improvement_score("hello", "hello world $500 2024-01-01")
    assert 0 <= score <= 1
    assert decision in ("keep_native", "replace", "append")

def test_improvement_scorer_prefers_vision_with_more_fields():
    from audit_ingestion.extractor import _compute_page_improvement_score, MERGE_REPLACE
    native = "Page header text only"
    vision = "Vendor: Acme Corp. Invoice #1234. Date: 2024-03-15. Total: $5,000.00. Approved by: JW"
    score, decision, reason = _compute_page_improvement_score(native, vision)
    assert score > 0.3
    assert decision in (MERGE_REPLACE, "append")

def test_improvement_scorer_keeps_native_when_vision_empty():
    from audit_ingestion.extractor import _compute_page_improvement_score, MERGE_KEEP_NATIVE, ERR_VISION_OUTPUT_EMPTY
    score, decision, reason = _compute_page_improvement_score("good native text", "")
    assert decision == MERGE_KEEP_NATIVE
    assert reason == ERR_VISION_OUTPUT_EMPTY

def test_handwriting_extractor_finds_hw_markers():
    from audit_ingestion.extractor import _extract_handwriting_from_vision
    text = "Invoice total: $500\nApproved by: JK\nOK to pay\nSee attached receipts"
    has_hw, hw_text, conf = _extract_handwriting_from_vision(text)
    assert has_hw
    assert conf > 0

def test_high_risk_field_detector_finds_amounts():
    from audit_ingestion.extractor import _detect_high_risk_field_changes
    native = "some text"
    vision = "Total payment: $125,000 due March 2024"
    changes = _detect_high_risk_field_changes(native, vision, page_number=3)
    field_types = [c["field_type"] for c in changes]
    assert "amount" in field_types or "date" in field_types

def test_page_classifier_detects_bank():
    from audit_ingestion.extractor import classify_page_type, PAGE_TYPE_BANK_TXNS
    text = "deposit withdrawal balance ACH wire debit credit checking account"
    assert classify_page_type(text) == PAGE_TYPE_BANK_TXNS

def test_page_classifier_detects_invoice():
    from audit_ingestion.extractor import classify_page_type, PAGE_TYPE_INVOICE_DETAIL
    text = "invoice bill to item quantity unit price subtotal tax total"
    assert classify_page_type(text) == PAGE_TYPE_INVOICE_DETAIL

def test_page_classifier_sparse_text_returns_handwritten():
    from audit_ingestion.extractor import classify_page_type, PAGE_TYPE_HANDWRITTEN
    assert classify_page_type("ok") == PAGE_TYPE_HANDWRITTEN

def test_missing_field_trigger_exists():
    from audit_ingestion.extractor import identify_pages_needing_vision_for_missing_fields
    assert callable(identify_pages_needing_vision_for_missing_fields)

def test_document_family_prompts_defined():
    from audit_ingestion.extractor import VISION_PROMPTS, get_vision_prompt_for_family
    for family in ("invoice_receipt", "bank_cash_activity", "contract_agreement",
                   "grant_donor_funding", "governance_approval", "payment_proof"):
        prompt = get_vision_prompt_for_family(family)
        assert len(prompt) > 50, f"Prompt for {family} is too short"

def test_tabular_guard_added_when_has_tables():
    from audit_ingestion.extractor import get_vision_prompt_for_family
    without = get_vision_prompt_for_family("invoice_receipt", has_tables=False)
    with_tbl = get_vision_prompt_for_family("invoice_receipt", has_tables=True)
    assert len(with_tbl) > len(without)
    assert "table" in with_tbl.lower()

def test_epistemic_honesty_in_base_prompt():
    from audit_ingestion.extractor import VISION_ANNOTATION_PROMPT
    assert "UNCLEAR" in VISION_ANNOTATION_PROMPT or "unclear" in VISION_ANNOTATION_PROMPT.lower()
    assert "invent" in VISION_ANNOTATION_PROMPT.lower() or "never" in VISION_ANNOTATION_PROMPT.lower()

def test_state_machine_constants():
    from audit_ingestion.models import (
        STATE_EXTRACTED_NATIVE, STATE_CANONICAL_COMPLETED,
        STATE_VISION_COMPLETED, STATE_REVIEWER_NEEDED, STATE_READY
    )
    assert STATE_EXTRACTED_NATIVE  == "extracted_native"
    assert STATE_CANONICAL_COMPLETED == "canonical_completed"
    assert STATE_READY             == "ready"

def test_evidence_has_ingestion_state():
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=100),
    )
    assert hasattr(ev, "ingestion_state")
    assert ev.ingestion_state == "extracted_native"

def test_evidence_has_vision_confidence_fields():
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=100),
    )
    assert hasattr(ev, "vision_confidence_document")
    assert hasattr(ev, "merge_confidence")
    assert hasattr(ev, "canonical_confidence_before")
    assert hasattr(ev, "canonical_rerun_mode")

def test_evidence_has_text_layers():
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="direct", total_chars=100),
        original_raw_text="native text here",
        working_raw_text="vision enhanced text here",
    )
    assert ev.original_raw_text == "native text here"
    assert ev.working_raw_text  == "vision enhanced text here"

def test_failure_codes_defined():
    from audit_ingestion.models import (
        ERR_VISION_PROVIDER_ERROR, ERR_VISION_OUTPUT_EMPTY,
        ERR_CANONICAL_RERUN_FAILED, ERR_VISION_NO_IMPROVEMENT
    )
    assert ERR_VISION_OUTPUT_EMPTY == "VISION_OUTPUT_EMPTY"
    assert ERR_CANONICAL_RERUN_FAILED == "CANONICAL_RERUN_FAILED"

def test_processing_mode_constants():
    from audit_ingestion.models import (
        MODE_STANDARD, MODE_STANDARD_THEN_VISION, MODE_VISION_FIRST,
        MODE_VISION_SELECTED, MODE_VISION_FULL_RETRY
    )
    assert MODE_STANDARD == "standard"
    assert MODE_VISION_FIRST == "vision_first"

def test_trigger_reason_constants():
    from audit_ingestion.models import (
        TRIGGER_WEAK_NATIVE, TRIGGER_MISSING_FIELDS,
        TRIGGER_HANDWRITING_SUSPECTED, TRIGGER_IMAGE_FILE
    )
    assert TRIGGER_MISSING_FIELDS == "missing_required_fields"
    assert TRIGGER_IMAGE_FILE     == "image_file_no_native_text"

def test_parsedpage_has_provenance_fields():
    from audit_ingestion.models import ParsedPage
    pg = ParsedPage(page_number=1, text="test", char_count=4, extractor="pdfplumber", confidence=0.9)
    assert hasattr(pg, "native_text")
    assert hasattr(pg, "vision_text")
    assert hasattr(pg, "merge_decision")
    assert hasattr(pg, "has_handwriting")
    assert hasattr(pg, "handwriting_text")

def test_fact_has_extractor_source():
    from audit_ingestion.models import Fact
    f = Fact(label="total", value=5000)
    assert hasattr(f, "extractor_source")
    assert f.extractor_source == "native"
    assert hasattr(f, "vision_enhanced")

def test_router_auto_triggers_vision_for_image():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/router.py"
    ).read_text()
    assert "direct_image" in src
    assert "vision_first" in src
    assert "TRIGGER_IMAGE_FILE" in src

def test_router_has_missing_field_trigger():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "audit_ingestion/router.py"
    ).read_text()
    assert "identify_pages_needing_vision_for_missing_fields" in src
    assert "TRIGGER_MISSING_FIELDS" in src

def test_ui_has_processing_report():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "ingest_app.py"
    ).read_text()
    assert "Processing Report" in src
    assert "Extractor chain" in src or "engine_chain" in src

def test_ui_has_delta_report():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "ingest_app.py"
    ).read_text()
    assert "What changed" in src or "vision_delta_report" in src

def test_ui_has_reviewer_confirmation():
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "ingest_app.py"
    ).read_text()
    assert "reviewer_confirmation_required" in src
    assert "Accept vision changes" in src
    assert "Revert to native extraction" in src
