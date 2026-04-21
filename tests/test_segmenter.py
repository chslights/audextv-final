"""
tests/test_segmenter.py
Comprehensive tests for v05 segmentation: classification, clustering,
primary selection, fallback, and build_primary_document.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_ingestion.models import (
    ParsedDocument, ParsedPage, DocumentComponent, AttachmentSummary, SegmentationResult,
)
from audit_ingestion.segmenter import (
    _summarize_pages, _build_components, build_primary_document,
    get_primary_pages, segment, CONF_HIGH, CONF_MEDIUM, SEGMENTATION_SCHEMA_VERSION,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_page(num: int, text: str, extractor: str = "pdfplumber") -> ParsedPage:
    return ParsedPage(
        page_number=num, text=text, char_count=len(text),
        extractor=extractor, confidence=0.9, image_used=False,
    )

def make_doc(pages: list[ParsedPage], source: str = "test.pdf") -> ParsedDocument:
    full = "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in pages)
    return ParsedDocument(
        source_file=source, full_text=full, page_count=len(pages),
        pages=pages, tables=[], extraction_chain=["pdfplumber"],
        primary_extractor="pdfplumber", confidence=0.9,
        weak_pages=[], ocr_pages=[], vision_pages=[], warnings=[], errors=[],
    )

def make_segmentation_result(
    bundle: bool, primary_pages: list[int],
    attachments: list[dict] = None, confidence: float = 0.90,
    band: str = "high", note: str = None,
) -> SegmentationResult:
    band = "high" if confidence >= CONF_HIGH else ("medium" if confidence >= CONF_MEDIUM else "low")
    att_summaries = []
    for i, a in enumerate(attachments or []):
        att_summaries.append(AttachmentSummary(
            component_id=f"attachment_{i+1}",
            component_group=a.get("group", "attachment"),
            pages=a["pages"],
            name=a.get("name", "Attachment"),
            summary=a.get("summary", "Supporting attachment."),
            key_identifiers=a.get("ids", []),
        ))
    return SegmentationResult(
        source_file="test.pdf",
        bundle_detected=bundle,
        bundle_confidence=confidence,
        confidence_band=band,
        primary_component=DocumentComponent(
            component_id="primary",
            component_group="primary_document",
            role="main",
            pages=primary_pages,
            description="Test Document",
            confidence=confidence,
        ),
        attachment_components=att_summaries,
        conservative_note=note,
    )


# ── Schema version ────────────────────────────────────────────────────────────

def test_segmentation_schema_version():
    assert SEGMENTATION_SCHEMA_VERSION == "v05.1"


# ── Model properties ──────────────────────────────────────────────────────────

def test_segmentation_result_has_attachments():
    r = make_segmentation_result(True, [1, 2], [{"pages": [3, 4], "name": "Spec"}])
    assert r.has_attachments is True
    assert r.primary_page_count == 2

def test_segmentation_result_no_attachments():
    r = make_segmentation_result(False, [1, 2, 3])
    assert r.has_attachments is False
    assert r.primary_page_count == 3


# ── Page summarization ────────────────────────────────────────────────────────

def test_summarize_pages_basic():
    pages = [
        make_page(1, "Schedule A TLSA ChoiceLease fixed charge lessee"),
        make_page(2, "Customer Proposal SPO Number vehicle spec"),
    ]
    summary = _summarize_pages(pages)
    assert "Page 1" in summary
    assert "Page 2" in summary
    assert "Schedule A" in summary
    assert "Customer Proposal" in summary

def test_summarize_pages_truncates_long_text():
    long_text = "A" * 2000
    pages = [make_page(1, long_text)]
    summary = _summarize_pages(pages)
    assert len(summary) < 900  # preview is capped at 600 chars

def test_summarize_pages_empty():
    summary = _summarize_pages([])
    assert summary == ""


# ── Component building ────────────────────────────────────────────────────────

def test_build_components_no_bundle():
    """When bundle_detected=False from AI, all pages go to primary. No conservative note — AI said no bundle."""
    pages = [make_page(i, f"content {i}") for i in range(1, 5)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": False,
        "bundle_confidence": 0.30,
        "primary_document_description": "Single document",
        "pages": [],
        "attachment_summaries": [],
    }
    result = _build_components(doc, classification)
    assert result.bundle_detected is False
    assert result.primary_component.pages == [1, 2, 3, 4]
    assert result.has_attachments is False
    assert result.confidence_band == "low"
    # No conservative note when AI explicitly says no bundle — only added when
    # bundle_detected=True but confidence is too low to split safely
    assert result.conservative_note is None

def test_build_components_low_confidence_bundle_adds_note():
    """When bundle is detected but confidence is too low to split, add conservative note."""
    pages = [make_page(i, f"content {i}") for i in range(1, 5)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": True,   # AI thinks it's a bundle
        "bundle_confidence": 0.40,  # but confidence too low to act on it
        "primary_document_description": "Unknown",
        "pages": [{"page_number": i, "role": "main", "component_group": "main",
                    "confidence": 0.40, "reason": "unclear"} for i in range(1, 5)],
        "attachment_summaries": [],
    }
    result = _build_components(doc, classification)
    assert result.bundle_detected is False  # overridden by low confidence
    assert result.conservative_note is not None  # note added

def test_build_components_high_confidence_bundle():
    """High-confidence bundle splits correctly into primary + attachments."""
    pages = [make_page(i, f"content {i}") for i in range(1, 6)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": True,
        "bundle_confidence": 0.92,
        "primary_document_description": "Schedule A Lease",
        "pages": [
            {"page_number": 1, "role": "main",       "component_group": "lease", "confidence": 0.95, "reason": "lease schedule"},
            {"page_number": 2, "role": "main",       "component_group": "lease", "confidence": 0.93, "reason": "lease terms"},
            {"page_number": 3, "role": "main",       "component_group": "lease", "confidence": 0.91, "reason": "signature"},
            {"page_number": 4, "role": "attachment", "component_group": "proposal", "confidence": 0.90, "reason": "vehicle proposal"},
            {"page_number": 5, "role": "attachment", "component_group": "proposal", "confidence": 0.88, "reason": "spec sheet"},
        ],
        "attachment_summaries": [
            {"component_group": "proposal", "name": "Vehicle Purchasing Proposal",
             "pages": [4, 5], "summary": "Chassis and body spec.", "key_identifiers": ["SPO 006-845"]},
        ],
    }
    result = _build_components(doc, classification)
    assert result.bundle_detected is True
    assert result.confidence_band == "high"
    assert result.primary_component.pages == [1, 2, 3]
    assert len(result.attachment_components) == 1
    assert result.attachment_components[0].name == "Vehicle Purchasing Proposal"
    assert result.attachment_components[0].pages == [4, 5]
    assert "SPO 006-845" in result.attachment_components[0].key_identifiers
    assert result.conservative_note is None

def test_build_components_medium_confidence_adds_note():
    """Medium confidence adds a conservative note but still segments."""
    pages = [make_page(i, f"content {i}") for i in range(1, 4)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": True,
        "bundle_confidence": 0.65,
        "primary_document_description": "Agreement",
        "pages": [
            {"page_number": 1, "role": "main",       "component_group": "main", "confidence": 0.70, "reason": "agreement"},
            {"page_number": 2, "role": "attachment", "component_group": "spec", "confidence": 0.60, "reason": "specs"},
            {"page_number": 3, "role": "main",       "component_group": "main", "confidence": 0.65, "reason": "terms"},
        ],
        "attachment_summaries": [
            {"component_group": "spec", "name": "Spec Sheet", "pages": [2], "summary": "Specs.", "key_identifiers": []},
        ],
    }
    result = _build_components(doc, classification)
    assert result.bundle_detected is True
    assert result.confidence_band == "medium"
    assert result.conservative_note is not None
    assert "conservatively" in result.conservative_note.lower()

def test_build_components_low_confidence_no_split():
    """Low confidence skips splitting — returns all pages as primary."""
    pages = [make_page(i, f"content {i}") for i in range(1, 5)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": True,
        "bundle_confidence": 0.40,
        "primary_document_description": "Unknown",
        "pages": [
            {"page_number": p, "role": "main", "component_group": "main",
             "confidence": 0.40, "reason": "unclear"} for p in range(1, 5)
        ],
        "attachment_summaries": [],
    }
    result = _build_components(doc, classification)
    assert result.bundle_detected is False
    assert result.primary_component.pages == [1, 2, 3, 4]
    assert result.has_attachments is False

def test_build_components_skip_pages_excluded():
    """Pages with role='skip' are excluded from primary and attachments."""
    pages = [make_page(i, f"content {i}") for i in range(1, 5)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": True,
        "bundle_confidence": 0.88,
        "primary_document_description": "Main Doc",
        "pages": [
            {"page_number": 1, "role": "main",       "component_group": "main", "confidence": 0.92, "reason": "main"},
            {"page_number": 2, "role": "main",       "component_group": "main", "confidence": 0.90, "reason": "main"},
            {"page_number": 3, "role": "skip",       "component_group": "blank", "confidence": 0.99, "reason": "blank"},
            {"page_number": 4, "role": "attachment", "component_group": "spec", "confidence": 0.87, "reason": "spec"},
        ],
        "attachment_summaries": [
            {"component_group": "spec", "name": "Spec", "pages": [4], "summary": ".", "key_identifiers": []},
        ],
    }
    result = _build_components(doc, classification)
    assert 3 not in result.primary_component.pages
    assert all(a.pages != [3] for a in result.attachment_components)

def test_build_components_unclassified_pages_go_to_primary():
    """Pages not in AI classification response default to primary."""
    pages = [make_page(i, f"content {i}") for i in range(1, 4)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": True,
        "bundle_confidence": 0.85,
        "primary_document_description": "Doc",
        "pages": [
            # Only page 1 classified — pages 2 and 3 missing from response
            {"page_number": 1, "role": "main", "component_group": "main", "confidence": 0.90, "reason": "main"},
        ],
        "attachment_summaries": [],
    }
    result = _build_components(doc, classification)
    # Pages 2 and 3 should be in primary since unclassified defaults to main
    assert 2 in result.primary_component.pages
    assert 3 in result.primary_component.pages

def test_build_components_all_attachment_falls_back_to_full():
    """If all pages classified as attachment, fall back to full document as primary."""
    pages = [make_page(i, f"spec content {i}") for i in range(1, 4)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": True,
        "bundle_confidence": 0.85,
        "primary_document_description": "Doc",
        "pages": [
            {"page_number": i, "role": "attachment", "component_group": "spec",
             "confidence": 0.85, "reason": "spec"} for i in range(1, 4)
        ],
        "attachment_summaries": [
            {"component_group": "spec", "name": "Spec", "pages": [1, 2, 3], "summary": ".", "key_identifiers": []},
        ],
    }
    result = _build_components(doc, classification)
    # All pages become primary — can't have zero primary pages
    assert len(result.primary_component.pages) == 3
    assert result.has_attachments is False


# ── Short document bypass ─────────────────────────────────────────────────────

def test_segment_skips_short_documents():
    """Documents with 2 or fewer pages skip AI classification."""
    pages = [make_page(1, "invoice content"), make_page(2, "page 2")]
    doc = make_doc(pages)

    class FakeProvider:
        model = "gpt-test"
        def extract_structured(self, **kwargs):
            raise AssertionError("Should not call AI for 2-page doc")

    result = segment(doc, FakeProvider())
    assert result.bundle_detected is False
    assert result.primary_component.pages == [1, 2]
    assert result.confidence_band == "high"


# ── AI failure fallback ───────────────────────────────────────────────────────

def test_segment_ai_failure_returns_full_document():
    """When AI classification fails, returns full document as primary."""
    # Use content that passes the segmentation gate (has bundle keyword signals)
    pages = [
        make_page(1, "Schedule A TLSA ChoiceLease lease terms fixed charge lessee"),
        make_page(2, "lease terms vehicle services liability insurance"),
        make_page(3, "party responsible insurance deductible signature"),
        make_page(4, "By Name Title Date execution block ryder"),
        make_page(5, "Customer Proposal SPO Number vehicle purchasing prepared for"),
    ]
    doc = make_doc(pages)

    class FailingProvider:
        model = "gpt-test"
        def extract_structured(self, **kwargs):
            raise RuntimeError("API timeout")

    result = segment(doc, FailingProvider())
    assert result.bundle_detected is False
    assert result.primary_component.pages == [1, 2, 3, 4, 5]
    assert result.confidence_band == "low"


# ── build_primary_document ────────────────────────────────────────────────────

def test_build_primary_document_filters_pages():
    """build_primary_document returns only primary component pages."""
    pages = [make_page(i, f"page {i} content here") for i in range(1, 6)]
    doc = make_doc(pages)
    seg = make_segmentation_result(True, [1, 2, 3], [{"pages": [4, 5], "name": "Spec"}])

    primary_doc = build_primary_document(doc, seg)
    assert primary_doc.page_count == 3
    page_nums = [p.page_number for p in primary_doc.pages]
    assert page_nums == [1, 2, 3]
    assert "segmented_primary" in primary_doc.extraction_chain

def test_build_primary_document_text_only_primary_pages():
    """Full text of primary document contains only primary page content."""
    pages = [
        make_page(1, "LEASE TERMS fixed charge $2273"),
        make_page(2, "LIABILITY insurance ryder"),
        make_page(3, "SPO Number 006-845 chassis weight"),   # attachment
        make_page(4, "Morgan body spec aluminum van"),        # attachment
    ]
    doc = make_doc(pages)
    seg = make_segmentation_result(True, [1, 2], [{"pages": [3, 4], "name": "Spec"}])

    primary_doc = build_primary_document(doc, seg)
    assert "LEASE TERMS" in primary_doc.full_text
    assert "LIABILITY" in primary_doc.full_text
    assert "SPO Number" not in primary_doc.full_text
    assert "Morgan body" not in primary_doc.full_text

def test_build_primary_document_no_bundle():
    """When no bundle, build_primary_document returns full document unchanged."""
    pages = [make_page(i, f"content {i}") for i in range(1, 4)]
    doc = make_doc(pages)
    seg = make_segmentation_result(False, [1, 2, 3])

    primary_doc = build_primary_document(doc, seg)
    assert primary_doc.page_count == 3
    for i in [1, 2, 3]:
        assert f"content {i}" in primary_doc.full_text


# ── get_primary_pages ─────────────────────────────────────────────────────────

def test_get_primary_pages_returns_correct_objects():
    pages = [make_page(i, f"text {i}") for i in range(1, 5)]
    doc = make_doc(pages)
    seg = make_segmentation_result(True, [1, 3])

    primary = get_primary_pages(doc, seg)
    assert len(primary) == 2
    assert {p.page_number for p in primary} == {1, 3}


# ── SegmentationResult model validation ──────────────────────────────────────

def test_segmentation_result_serializes():
    """SegmentationResult must serialize to dict cleanly."""
    r = make_segmentation_result(
        True, [1, 2, 3],
        [{"pages": [4, 5], "name": "Spec Sheet", "summary": "Specs.", "ids": ["SPO 123"]}],
        confidence=0.91,
    )
    d = r.model_dump()
    assert d["bundle_detected"] is True
    assert d["primary_component"]["pages"] == [1, 2, 3]
    assert len(d["attachment_components"]) == 1
    assert d["attachment_components"][0]["name"] == "Spec Sheet"
    assert d["attachment_components"][0]["key_identifiers"] == ["SPO 123"]


# ── Confidence band logic ─────────────────────────────────────────────────────

def test_confidence_band_thresholds():
    assert CONF_HIGH == 0.80
    assert CONF_MEDIUM == 0.55

def test_build_components_confidence_bands():
    """Test that confidence bands are assigned correctly."""
    pages = [make_page(i, f"content {i}") for i in range(1, 4)]
    doc = make_doc(pages)

    for conf, expected_band in [(0.90, "high"), (0.65, "medium"), (0.40, "low")]:
        cls = {
            "bundle_detected": True if conf >= CONF_MEDIUM else True,
            "bundle_confidence": conf,
            "primary_document_description": "Doc",
            "pages": [{"page_number": i, "role": "main", "component_group": "main",
                        "confidence": conf, "reason": "test"} for i in range(1, 4)],
            "attachment_summaries": [],
        }
        result = _build_components(doc, cls)
        assert result.confidence_band == expected_band, f"Expected {expected_band} for conf={conf}, got {result.confidence_band}"
