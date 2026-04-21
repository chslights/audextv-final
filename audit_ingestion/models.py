"""
audit_ingestion_v04.2/audit_ingestion/models.py
Canonical audit evidence schema — Pydantic models.

Two layers:
1. ParsedDocument — page-aware raw extraction output
2. AuditEvidence  — canonical audit evidence output

Every document goes through both layers.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any, Literal
from enum import Enum


# ── Document Family ───────────────────────────────────────────────────────────

class DocumentFamily(str, Enum):
    CONTRACT       = "contract_agreement"
    INVOICE        = "invoice_receipt"
    PAYMENT        = "payment_proof"
    BANK           = "bank_cash_activity"
    PAYROLL        = "payroll_support"
    ACCOUNTING     = "accounting_report"
    GOVERNANCE     = "governance_approval"
    GRANT          = "grant_donor_funding"
    TAX_REG        = "tax_regulatory"
    CORRESPONDENCE = "correspondence"
    SCHEDULE       = "schedule_listing"
    OTHER          = "other"


# ── Layer 1: Page-Aware Parsed Document ───────────────────────────────────────

class ParsedPage(BaseModel):
    """Text extracted from one PDF page."""
    page_number:  int
    text:         str = ""          # working text (native, OCR, or vision — whichever is best)
    char_count:   int = 0
    extractor:    str = "none"      # pdfplumber, pypdf2, extractous, ocr, vision
    confidence:   float = 0.0
    image_used:   bool = False
    warnings:     list[str] = Field(default_factory=list)
    # ── Per-page provenance (Items 2, 3, 9) ──────────────────────────────────
    native_text:          Optional[str]  = None   # original machine-extracted text
    vision_text:          Optional[str]  = None   # vision-extracted text if run
    final_text_used:      Optional[str]  = None   # what canonical extraction saw
    final_extractor_used: str            = "native"  # native | ocr | vision | hybrid
    merge_decision:       str            = "keep_native"  # keep_native | replace | append
    merge_reason:         Optional[str]  = None
    improvement_score:    float          = 0.0    # 0-1, how much vision improved this page
    has_handwriting:      bool           = False
    handwriting_text:     Optional[str]  = None
    handwriting_confidence: float        = 0.0

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.text)


class ParsedTable(BaseModel):
    """Table extracted from a document page."""
    page_number:  int
    table_index:  int = 0
    headers:      list[str] = Field(default_factory=list)
    rows:         list[dict[str, Any]] = Field(default_factory=list)
    row_count:    int = 0
    extractor:    str = "unknown"


class ParsedDocument(BaseModel):
    """
    Page-aware extraction output — the foundation for canonical extraction.
    Every page has its own text record and provenance.
    """
    source_file:      str
    file_hash:        Optional[str] = None   # MD5 of file content — used for cache keying
    mime_type:        Optional[str] = None
    full_text:        str = ""           # Assembled from page texts
    page_count:       int = 0
    pages:            list[ParsedPage]   = Field(default_factory=list)
    tables:           list[ParsedTable]  = Field(default_factory=list)
    extraction_chain: list[str]          = Field(default_factory=list)
    primary_extractor: str = "none"
    confidence:       float = 0.0
    weak_pages:       list[int]          = Field(default_factory=list)  # Pages below threshold
    ocr_pages:        list[int]          = Field(default_factory=list)  # Pages rescued by OCR
    vision_pages:     list[int]          = Field(default_factory=list)  # Pages rescued by vision
    warnings:         list[str]          = Field(default_factory=list)
    errors:           list[str]          = Field(default_factory=list)

    @property
    def chars_per_page(self) -> float:
        if not self.page_count:
            return 0.0
        return len(self.full_text) / self.page_count

    @property
    def is_sufficient(self) -> bool:
        return len(self.full_text) >= 300 and self.chars_per_page >= 150


# ── Layer 2: Canonical Audit Evidence ────────────────────────────────────────

class Provenance(BaseModel):
    """Source evidence for any extracted item — mandatory for material facts."""
    page:       Optional[int]   = None
    quote:      Optional[str]   = None   # ≤20 word verbatim excerpt
    confidence: float           = 0.0


class Party(BaseModel):
    role:       str                      # lessor, vendor, grantor, payer, client, etc.
    name:       str
    normalized: str                      # UPPERCASE no-punctuation for matching
    provenance: Optional[Provenance] = None


class Amount(BaseModel):
    type:       str                      # monthly_fixed_charge, total_award, etc.
    value:      float
    currency:   str = "USD"
    provenance: Optional[Provenance] = None


class DateItem(BaseModel):
    type:       str                      # effective_date, invoice_date, period_start, etc.
    value:      str                      # YYYY-MM-DD
    provenance: Optional[Provenance] = None


class Identifier(BaseModel):
    type:       str                      # invoice_number, schedule_number, grant_number, etc.
    value:      str
    provenance: Optional[Provenance] = None


class AssetItem(BaseModel):
    type:        str                     # vehicle, equipment, property, program, etc.
    description: str
    value:       Optional[float] = None
    provenance:  Optional[Provenance] = None


class Fact(BaseModel):
    """Atomic extracted fact — drives matching. Must have provenance."""
    label:      str                      # snake_case: term_months, mileage_rate, etc.
    value:      Any
    provenance: Optional[Provenance] = None
    extractor_source: str = "native"     # native | ocr | vision | hybrid (Item 16)
    vision_enhanced:  bool = False       # True if vision changed this field


class Claim(BaseModel):
    """Auditor-readable interpretation built from facts. Must cite source facts."""
    statement:         str
    audit_area:        str               # leases, expenses, revenue, etc.
    basis_fact_labels: list[str] = Field(default_factory=list)
    provenance:        Optional[Provenance] = None


# ── Vision trigger reasons (Item 5) ──────────────────────────────────────────
TRIGGER_WEAK_NATIVE          = "weak_native_text"
TRIGGER_MISSING_FIELDS       = "missing_required_fields"
TRIGGER_HANDWRITING_SUSPECTED= "handwriting_suspected"
TRIGGER_LAYOUT_COMPLEXITY    = "layout_complexity"
TRIGGER_LOW_NUMERIC          = "low_numeric_capture"
TRIGGER_REVIEWER_REQUESTED   = "reviewer_requested"
TRIGGER_PAGE_TYPE            = "page_type_requires_vision"
TRIGGER_IMAGE_FILE           = "image_file_no_native_text"

# ── Processing modes (Item 6) ─────────────────────────────────────────────────
MODE_STANDARD                = "standard"
MODE_STANDARD_THEN_VISION    = "standard_then_vision_if_needed"
MODE_VISION_FIRST            = "vision_first"
MODE_VISION_SELECTED         = "vision_selected_pages"
MODE_VISION_MISSING_FIELDS   = "vision_after_missing_fields"
MODE_VISION_FULL_RETRY       = "vision_full_retry"

# ── Ingestion state machine (Items 25, 30) ────────────────────────────────────
STATE_EXTRACTED_NATIVE      = "extracted_native"
STATE_EXTRACTED_OCR         = "extracted_ocr"
STATE_VISION_REQUESTED      = "vision_requested"
STATE_VISION_COMPLETED      = "vision_completed"
STATE_MERGE_APPLIED         = "merge_applied"
STATE_CANONICAL_COMPLETED   = "canonical_completed"
STATE_CANONICAL_BACKFILLED  = "canonical_backfilled"
STATE_REVIEWER_NEEDED       = "reviewer_confirmation_needed"
STATE_READY                 = "ready"
STATE_EXCEPTION             = "exception"

# ── Merge decisions (Item 1) ──────────────────────────────────────────────────
MERGE_KEEP_NATIVE    = "keep_native"
MERGE_REPLACE        = "replace"
MERGE_APPEND         = "append"
MERGE_HYBRID         = "hybrid"

# ── Failure reason codes (Item 20) ───────────────────────────────────────────
ERR_VISION_PROVIDER_ERROR        = "VISION_PROVIDER_ERROR"
ERR_VISION_SCHEMA_INVALID        = "VISION_SCHEMA_INVALID"
ERR_VISION_PAGE_RENDER_FAILED    = "VISION_PAGE_RENDER_FAILED"
ERR_VISION_OUTPUT_EMPTY          = "VISION_OUTPUT_EMPTY"
ERR_VISION_NO_IMPROVEMENT        = "VISION_NO_IMPROVEMENT"
ERR_VISION_MERGE_SKIPPED         = "VISION_MERGE_SKIPPED"
ERR_CANONICAL_RERUN_FAILED       = "CANONICAL_RERUN_FAILED"
ERR_MISSING_FIELDS_POST_VISION   = "MISSING_REQUIRED_FIELDS_POST_VISION"


class Flag(BaseModel):
    """Audit exception, risk, or attention item."""
    type:        str
    description: str
    severity:    Literal["info", "warning", "critical"] = "info"


# ── Vision page status values ─────────────────────────────────────────────────
VISION_PAGE_NOT_ATTEMPTED    = "not_attempted"
VISION_PAGE_BASE_EXTRACTED   = "base_extracted"
VISION_PAGE_REQUESTED        = "vision_requested"
VISION_PAGE_COMPLETE         = "vision_complete"
VISION_PAGE_PARTIAL          = "vision_partial"
VISION_PAGE_FAILED           = "vision_failed"
VISION_PAGE_REVIEW_NEEDED    = "operator_review_needed"


# ── Evidence Readiness Models ─────────────────────────────────────────────────

class Question(BaseModel):
    """
    A structured question generated from an unresolved flag or gap.
    Drives the evidence completion workflow.
    """
    question_id:   str                                          # e.g. "missing_period_q1"
    question_type: str                                          # e.g. "period_confirmation"
    question_text: str                                          # human-readable prompt
    audience:      Literal["reviewer", "client"] = "reviewer"  # who needs to answer
    blocking:      bool = True                                  # blocks Ready status?
    source_flag:   Optional[str] = None                         # flag that triggered this
    resolved:      bool = False
    resolution:    Optional[str] = None                         # user's answer
    status:        Literal["open", "resolved", "overridden", "dismissed", "superseded"] = "open"
    resolution_type: Optional[Literal["answer", "reviewer_confirmed", "override", "dismissed", "superseded"]] = None
    resolved_by:   Optional[str] = None
    resolved_at:   Optional[str] = None
    comments:      Optional[str] = None


class ReadinessResult(BaseModel):
    """
    Evidence readiness assessment — separate from processing status.
    Computed after extraction; can be updated as questions are resolved.
    """
    readiness_status: Literal[
        "ready",
        "needs_reviewer_confirmation",
        "needs_client_answer",
        "exception_open",
        "unusable",
    ] = "ready"
    blocking_state:   Literal["blocking", "non_blocking"] = "non_blocking"
    blocking_issues:  list[str] = Field(default_factory=list)   # flag types that block
    questions:        list[Question] = Field(default_factory=list)
    population_ready: Optional[bool] = None   # financial files only
    population_status: Optional[str] = None   # description of why not population-ready
    evidence_use_mode: Literal["evidence_and_population", "evidence_only", "unusable"] = "evidence_and_population"


class AuditPeriod(BaseModel):
    effective_date: Optional[str] = None
    start:          Optional[str] = None
    end:            Optional[str] = None
    term_months:    Optional[int] = None


class AuditOverview(BaseModel):
    summary:       str
    audit_areas:   list[str] = Field(default_factory=list)
    assertions:    list[str] = Field(default_factory=list)
    period:        Optional[AuditPeriod] = None
    match_targets: list[str] = Field(default_factory=list)


class LinkKeys(BaseModel):
    """Normalized keys for cross-document matching."""
    party_names:       list[str]   = Field(default_factory=list)
    document_numbers:  list[str]   = Field(default_factory=list)
    agreement_numbers: list[str]   = Field(default_factory=list)
    invoice_numbers:   list[str]   = Field(default_factory=list)
    asset_descriptions:list[str]   = Field(default_factory=list)
    recurring_amounts: list[float] = Field(default_factory=list)
    key_dates:         list[str]   = Field(default_factory=list)
    other_ids:         list[str]   = Field(default_factory=list)


class ExtractionMeta(BaseModel):
    primary_extractor:    str = "none"
    pages_processed:      int = 0
    weak_pages_count:     int = 0
    ocr_pages_count:      int = 0
    vision_pages_count:   int = 0
    total_chars:          int = 0
    overall_confidence:   float = 0.0
    needs_human_review:   bool = True
    canonical_validated:  bool = False
    canonical_retried:    bool = False
    warnings:             list[str] = Field(default_factory=list)
    errors:               list[str] = Field(default_factory=list)


class AuditEvidence(BaseModel):
    """
    Canonical audit evidence object.
    One per document. Always the same shape.
    Works for any document type.
    """
    source_file:      str
    family:           DocumentFamily = DocumentFamily.OTHER
    subtype:          Optional[str] = None
    title:            Optional[str] = None
    audit_overview:   Optional[AuditOverview] = None
    parties:          list[Party]      = Field(default_factory=list)
    amounts:          list[Amount]     = Field(default_factory=list)
    dates:            list[DateItem]   = Field(default_factory=list)
    identifiers:      list[Identifier] = Field(default_factory=list)
    assets:           list[AssetItem]  = Field(default_factory=list)
    facts:            list[Fact]       = Field(default_factory=list)
    claims:           list[Claim]      = Field(default_factory=list)
    flags:            list[Flag]       = Field(default_factory=list)
    link_keys:        LinkKeys         = Field(default_factory=LinkKeys)
    document_specific:dict[str, Any]   = Field(default_factory=dict)
    raw_text:         Optional[str]    = None
    tables:           list[dict]       = Field(default_factory=list)
    extraction_meta:  ExtractionMeta   = Field(
        default_factory=lambda: ExtractionMeta(primary_extractor="none")
    )
    readiness:        Optional["ReadinessResult"] = None
    # ── Text layers — four explicit versions (Item 2) ────────────────────────
    original_raw_text:     Optional[str]         = None   # native extraction, never mutated
    working_raw_text:      Optional[str]         = None   # what canonical actually ran on
    vision_overlay_text:   Optional[str]         = None   # combined vision page texts
    # ── Vision extraction fields ─────────────────────────────────────────────
    vision_applied:        bool                  = False
    vision_mode_used:      Optional[str]         = None
    vision_trigger_reason: Optional[str]         = None   # why vision was triggered
    processing_mode:       str                   = MODE_STANDARD
    vision_pages_used:     list[int]             = Field(default_factory=list)
    vision_text:           Optional[str]         = None
    vision_page_texts:     dict[int, str]        = Field(default_factory=dict)
    vision_page_status:    dict[int, str]        = Field(default_factory=dict)
    vision_prompt:         Optional[str]         = None
    vision_error:          Optional[str]         = None
    vision_run_diagnostics: Optional[dict]       = None
    # ── Delta report — before/after vision (Item 3) ──────────────────────────
    vision_delta_report:   Optional[dict]        = None
    # ── Vision confidence (Item 14) ──────────────────────────────────────────
    vision_confidence_document: Optional[str]    = None  # high/medium/low
    vision_confidence_pages:    dict[int, str]   = Field(default_factory=dict)
    merge_confidence:           Optional[str]    = None
    canonical_confidence_before: Optional[float] = None
    canonical_confidence_after:  Optional[float] = None
    # ── Canonical rerun mode (Item 15) ───────────────────────────────────────
    canonical_rerun_mode:  Optional[str]         = None  # full_rerun | field_backfill | none
    ingestion_state:       str                   = STATE_EXTRACTED_NATIVE  # current state
    # ── Field provenance — which extractor produced each fact (Item 16) ──────
    page_provenance:       dict[int, dict]       = Field(default_factory=dict)
    # ── High-risk field change review (Item 22) ──────────────────────────────
    vision_changed_high_risk_fields: list[dict]  = Field(default_factory=list)
    reviewer_confirmation_required:  bool        = False


class IngestionResult(BaseModel):
    evidence:     Optional[AuditEvidence] = None
    status:       Literal["success", "partial", "failed"] = "partial"
    errors:       list[str] = Field(default_factory=list)
    engine_chain: list[str] = Field(default_factory=list)


# ── v05 Segmentation Models ───────────────────────────────────────────────────

class DocumentComponent(BaseModel):
    """A logical grouping of pages within a bundled PDF."""
    component_id:    str
    component_group: str
    role:            str
    pages:           list[int]
    description:     str
    confidence:      float = 0.0


class AttachmentSummary(BaseModel):
    """Lightweight summary of a supporting attachment component."""
    component_id:    str
    component_group: str
    pages:           list[int]
    name:            str
    summary:         str
    key_identifiers: list[str] = Field(default_factory=list)
    attachment_role: str = "supporting_document"


class SegmentationResult(BaseModel):
    """Output of the segmenter."""
    source_file:           str
    bundle_detected:       bool
    bundle_confidence:     float
    confidence_band:       str
    primary_component:     DocumentComponent
    attachment_components: list[AttachmentSummary] = Field(default_factory=list)
    conservative_note:     Optional[str] = None

    @property
    def has_attachments(self) -> bool:
        return len(self.attachment_components) > 0

    @property
    def primary_page_count(self) -> int:
        return len(self.primary_component.pages)
