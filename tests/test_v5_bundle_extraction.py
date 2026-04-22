"""
tests/test_v5_bundle_extraction.py
Integration tests for v05 bundle-aware extraction pipeline.
Tests the full segmentation → primary scoping → canonical flow.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_ingestion.models import (
    ParsedDocument, ParsedPage, AuditEvidence, SegmentationResult,
    DocumentComponent, AttachmentSummary, Flag,
)
from audit_ingestion.segmenter import (
    segment, build_primary_document, get_primary_pages,
    _build_components, _summarize_pages,
)
from audit_ingestion.router import _annotate_with_segmentation, _score


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_page(num, text, extractor="pdfplumber"):
    return ParsedPage(
        page_number=num, text=text, char_count=len(text),
        extractor=extractor, confidence=0.9, image_used=False,
    )

def make_doc(pages, source="test.pdf"):
    full = "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in pages)
    return ParsedDocument(
        source_file=source, full_text=full, page_count=len(pages),
        pages=pages, tables=[], extraction_chain=["pdfplumber"],
        primary_extractor="pdfplumber", confidence=0.9,
        weak_pages=[], ocr_pages=[], vision_pages=[], warnings=[], errors=[],
    )

def ryder_lease_doc():
    """Simulated Ryder lease bundle — 4 lease pages + 5 spec/proposal pages."""
    return make_doc([
        make_page(1, "Ryder ChoiceLease Full Service TRUCK LEASE & SERVICE AGREEMENT TLSA SCHEDULE A "
                     "Customer Name: Test customer Lessee Number: 1 "
                     "Original Value: $88,112 Term In Months: 72 "
                     "Monthly Depreciation: $552.28 Fixed Charge Per Month: $2,273.00 "
                     "Estimated Annual Mileage: 45,000 Mileage Rate per Mile: $0.0722 "
                     "Schedule A No. 2329109 Schedule A Date: December 10th, 2019"),
        make_page(2, "TLSA lease terms investment original value monthly depreciation "
                     "CPI base index adjustment percent fixed charge mileage surcharge "
                     "Per Vehicle Annual Allowances state motor vehicle license registration "
                     "Vehicle Related Services Ryder substitute vehicles exterior washing"),
        make_page(3, "Party Responsible for Liability Insurance Ryder Combined Single Limits "
                     "$1,000,000 Customer Deductible per occurrence $5,000 "
                     "Physical Damage Ryder customer deductible $5,000 "
                     "Promotional Incentive $2,273.00 holdover charges 20 percent "
                     "OPIS Base Price fuel Ryder schedule A vehicle lease agreement"),
        make_page(4, "RYDER TRUCK RENTAL INC dba RYDER TRANSPORTATION SERVICES "
                     "By: Name: Page Breaux Title: Director of Sales DOS Date: "
                     "Customer/You By: Name: Title: Date: signature execution"),
        make_page(5, "Vehicle Purchasing Customer Proposal Contains Confidential Information "
                     "Prepared For: Test customer Power Unit Specifications "
                     "Vendor: INTL - INTERNATIONAL TRUCK SPO Number: 006 - 845 "
                     "MODEL MV607 GVWR 25,999 AXLE CONFIGURATION 4X2 "
                     "CHASSIS AND CAB WEIGHTS STANDARD ORDER DESCRIPTIONS"),
        make_page(6, "Power Unit Specifications Vendor: INTL - INTERNATIONAL TRUCK "
                     "SPO Number: 006 - 845 COOLING ALUM RAD ENGINE CUMM B6.7 "
                     "ELECTRICAL 12 VOLT SYSTEM DRIVELINE SPICER SPL-100 "
                     "TRANSMISSION ALLI 2500RDS FUEL TANKS 40 GAL LH 50 GAL RH"),
        make_page(7, "Power Unit Specifications TIRES BRIDGESTONE LOW PROFILE "
                     "TRANSMISSION ALLI 2500RDS WHEELS HUB-PILOTED "
                     "MISCELLANEOUS ADDITIONAL SOI ENGINE CREDIT "
                     "Ryder Approval: Customer Approval: Date: Page 3 of 3"),
        make_page(8, "Body Specifications Vendor: MRGN - MORGAN CORPORATION "
                     "SPO Number: 050 - 201 MODEL GVSD97-24 LGTH INSIDE 23ft "
                     "WEIGHT 3542 LBS CROSSMEMBERS 3 INCH I-BEAM "
                     "FLOOR 1-1/8 LAMINATED HARDWOOD SIDE POSTS GALVANIZED STEEL"),
        make_page(9, "Body Specifications Vendor: MRGN - MORGAN CORPORATION "
                     "SPO Number: 050 - 201 GRABHANDLE CURBSIDE ROADSIDE "
                     "MAXON GPTB-3 3000 CAP FREIGHT NOT INCLUDED "
                     "Ryder Approval: Customer Approval: Date: Page 2 of 2"),
    ], source="Redacted_Ryder_Lease_Agreement.pdf")


# ── Bundle detection ──────────────────────────────────────────────────────────

class MockProvider:
    """Mock provider that returns realistic classification for the Ryder bundle."""
    model = "gpt-5.4"

    def extract_structured(self, system, user, json_schema, max_tokens=2000):
        # Simulate AI correctly classifying the Ryder bundle
        return {
            "bundle_detected": True,
            "bundle_confidence": 0.92,
            "primary_document_description": "Ryder Schedule A Vehicle Lease",
            "pages": [
                {"page_number": 1, "role": "main",       "component_group": "lease", "confidence": 0.97, "reason": "TLSA Schedule A with lease terms"},
                {"page_number": 2, "role": "main",       "component_group": "lease", "confidence": 0.95, "reason": "Lease terms and vehicle services"},
                {"page_number": 3, "role": "main",       "component_group": "lease", "confidence": 0.94, "reason": "Insurance and promotional terms"},
                {"page_number": 4, "role": "main",       "component_group": "lease", "confidence": 0.92, "reason": "Signature execution block"},
                {"page_number": 5, "role": "attachment", "component_group": "vehicle_proposal", "confidence": 0.93, "reason": "Customer Proposal - vehicle specs"},
                {"page_number": 6, "role": "attachment", "component_group": "vehicle_proposal", "confidence": 0.91, "reason": "Power unit specifications"},
                {"page_number": 7, "role": "attachment", "component_group": "vehicle_proposal", "confidence": 0.89, "reason": "Transmission and wheels specs"},
                {"page_number": 8, "role": "attachment", "component_group": "body_spec", "confidence": 0.90, "reason": "Morgan body specifications"},
                {"page_number": 9, "role": "attachment", "component_group": "body_spec", "confidence": 0.88, "reason": "Body spec continued"},
            ],
            "attachment_summaries": [
                {
                    "component_group": "vehicle_proposal",
                    "name": "Vehicle Purchasing Proposal",
                    "pages": [5, 6, 7],
                    "summary": "Customer proposal for 2020 International MV607 chassis with power unit specifications.",
                    "key_identifiers": ["SPO 006-845", "Vendor: INTL International Truck", "Model: MV607"],
                },
                {
                    "component_group": "body_spec",
                    "name": "Body Specifications",
                    "pages": [8, 9],
                    "summary": "Morgan Corporation body specifications for aluminum van with liftgate.",
                    "key_identifiers": ["SPO 050-201", "Vendor: Morgan Corporation", "Model: GVSD97-24"],
                },
            ],
        }


def test_ryder_bundle_detection():
    """Ryder lease should detect as bundle with 4 primary + 5 attachment pages."""
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())

    assert result.bundle_detected is True
    assert result.confidence_band == "high"
    assert result.primary_component.pages == [1, 2, 3, 4]
    assert len(result.attachment_components) == 2


def test_ryder_attachment_names():
    """Attachment names should be human-readable, not technical."""
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())

    names = [a.name for a in result.attachment_components]
    assert "Vehicle Purchasing Proposal" in names
    assert "Body Specifications" in names


def test_ryder_attachment_identifiers():
    """Key identifiers should be captured in attachment summaries."""
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())

    all_ids = []
    for a in result.attachment_components:
        all_ids.extend(a.key_identifiers)

    assert any("SPO 006-845" in id_ for id_ in all_ids)
    assert any("SPO 050-201" in id_ for id_ in all_ids)


def test_primary_document_excludes_spec_pages():
    """Primary document text must not contain spec/proposal content."""
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())
    primary_doc = build_primary_document(doc, result)

    # Lease content present
    assert "TLSA" in primary_doc.full_text
    assert "Fixed Charge Per Month" in primary_doc.full_text
    assert "2,273" in primary_doc.full_text
    assert "Lessee" in primary_doc.full_text.lower() or "lessee" in primary_doc.full_text.lower()

    # Spec content excluded
    assert "SPO Number: 006 - 845" not in primary_doc.full_text
    assert "MORGAN CORPORATION" not in primary_doc.full_text
    assert "BRIDGESTONE" not in primary_doc.full_text
    assert "Customer Proposal" not in primary_doc.full_text


def test_primary_document_page_count():
    """Primary document should have exactly 4 pages for the Ryder bundle."""
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())
    primary_doc = build_primary_document(doc, result)

    assert primary_doc.page_count == 4
    assert len(primary_doc.pages) == 4


def test_signature_page_stays_in_primary():
    """Page 4 (signature) must remain in primary, not become an attachment."""
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())

    assert 4 in result.primary_component.pages
    for att in result.attachment_components:
        assert 4 not in att.pages


# ── Annotation ────────────────────────────────────────────────────────────────

def test_annotate_with_bundle_adds_flag():
    """_annotate_with_segmentation should add bundle_detected flag."""
    from audit_ingestion.models import AuditEvidence, ExtractionMeta, LinkKeys, AuditOverview
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="pdfplumber"),
    )
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())
    _annotate_with_segmentation(ev, result)

    flag_types = [f.type for f in ev.flags]
    assert "bundle_detected" in flag_types

    bundle_flag = next(f for f in ev.flags if f.type == "bundle_detected")
    assert "Ryder Schedule A Vehicle Lease" in bundle_flag.description
    assert "Vehicle Purchasing Proposal" in bundle_flag.description


def test_annotate_stores_segmentation_in_doc_specific():
    """Segmentation data should be stored in document_specific._segmentation."""
    from audit_ingestion.models import AuditEvidence, ExtractionMeta
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="pdfplumber"),
    )
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())
    _annotate_with_segmentation(ev, result)

    assert "_segmentation" in ev.document_specific
    seg_data = ev.document_specific["_segmentation"]
    assert seg_data["bundle_detected"] is True
    assert seg_data["primary_pages"] == [1, 2, 3, 4]
    assert len(seg_data["attachments"]) == 2


def test_annotate_no_bundle_no_flag():
    """When no bundle, no bundle_detected flag should be added."""
    from audit_ingestion.models import AuditEvidence, ExtractionMeta
    ev = AuditEvidence(
        source_file="test.pdf",
        extraction_meta=ExtractionMeta(primary_extractor="pdfplumber"),
    )
    pages = [make_page(i, f"content {i}") for i in range(1, 3)]
    doc = make_doc(pages)

    class NoBundleProvider:
        model = "gpt-5.4"
        def extract_structured(self, **kwargs):
            return {
                "bundle_detected": False, "bundle_confidence": 0.20,
                "primary_document_description": "Simple doc",
                "pages": [], "attachment_summaries": [],
            }

    result = segment(doc, NoBundleProvider())
    _annotate_with_segmentation(ev, result)

    flag_types = [f.type for f in ev.flags]
    assert "bundle_detected" not in flag_types


# ── Conservative fallback ─────────────────────────────────────────────────────

def test_conservative_fallback_language():
    """Conservative note should use user-friendly language."""
    pages = [make_page(i, f"content {i}") for i in range(1, 5)]
    doc = make_doc(pages)
    classification = {
        "bundle_detected": True, "bundle_confidence": 0.35,
        "primary_document_description": "Unknown",
        "pages": [{"page_number": i, "role": "main", "component_group": "main",
                    "confidence": 0.35, "reason": "unclear"} for i in range(1, 5)],
        "attachment_summaries": [],
    }
    result = _build_components(doc, classification)
    assert result.conservative_note is not None
    # Should be plain English, no technical jargon
    note = result.conservative_note.lower()
    assert "multiple sections" in note or "conservatively" in note
    assert "component_group" not in note
    assert "confidence_band" not in note


# ── Single document passthrough ───────────────────────────────────────────────

def test_single_document_passthrough():
    """Simple single-document PDFs should pass through unchanged."""
    pages = [
        make_page(1, "Chase Performance Business Checking Beginning Balance $143,588.77 "
                     "Deposits and Additions 31 $47,041.80 Ending Balance $165,650.71 "
                     "FLASHTECH LLC Account Number 000000761309889"),
        make_page(2, "Checks Paid 1957 05/16 $980.00 ATM Debit Card Withdrawals "
                     "Electronic Withdrawals Barclaycard Chase Quickpay"),
    ]
    doc = make_doc(pages, "May16_statement.pdf")

    # 2-page doc skips AI classification entirely
    class ShouldNotBeCalled:
        model = "gpt-5.4"
        def extract_structured(self, **kwargs):
            raise AssertionError("Should not classify 2-page doc")

    result = segment(doc, ShouldNotBeCalled())
    assert result.bundle_detected is False
    assert result.primary_component.pages == [1, 2]


# ── Attachment summaries do not populate primary facts ───────────────────────

def test_attachment_content_excluded_from_primary_text():
    """
    Core test: spec/proposal content must not appear in primary document text.
    This directly validates the contamination prevention goal.
    """
    doc = ryder_lease_doc()
    result = segment(doc, MockProvider())
    primary_doc = build_primary_document(doc, result)

    # Things that MUST be in primary
    must_have = ["TLSA", "Schedule A", "2,273", "Term In Months"]
    for item in must_have:
        assert item in primary_doc.full_text, f"Missing from primary: {item}"

    # Things that must NOT be in primary (these are in spec/proposal pages)
    must_not_have = [
        "SPO Number: 006 - 845",    # proposal page 5
        "MORGAN CORPORATION",        # body spec page 8
        "BRIDGESTONE",               # tire specs page 7
        "GVSD97-24",                 # body model page 8
        "3542 LBS",                  # body weight page 8
    ]
    for item in must_not_have:
        assert item not in primary_doc.full_text, f"Contamination found in primary: {item}"


# ── v05 version markers ───────────────────────────────────────────────────────

def test_build_version():
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "ingest_app.py"
    src = app_path.read_text()
    assert 'BUILD_VERSION = "v06.5"' in src

def test_canonical_schema_version():
    from audit_ingestion.canonical import SCHEMA_VERSION
    assert SCHEMA_VERSION == "v05.1"

def test_segmentation_schema_version():
    from audit_ingestion.segmenter import SEGMENTATION_SCHEMA_VERSION
    assert SEGMENTATION_SCHEMA_VERSION == "v05.1"

def test_package_version():
    import audit_ingestion
    assert audit_ingestion.__version__ == "6.5.0"
