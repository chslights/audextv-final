"""
audit_ingestion_v05/audit_ingestion/segmenter.py

Automatic bundle-aware document segmentation.

Design principle: if the user has to think about segmentation, we surfaced too much.

Flow:
  1. Page summarization — extract a short descriptor for each page
  2. AI classification — one API call classifies all pages at once
  3. Component grouping — consecutive pages with same role are grouped
  4. Primary selection — pick the main document component
  5. Return SegmentationResult — primary pages + attachment summaries

Confidence bands:
  HIGH   (0.80-1.00) → segment automatically, no user flag
  MEDIUM (0.55-0.79) → conservative split, note added
  LOW    (0.00-0.54) → do not split, single component, quiet note

The user never sees page families, confidence scores, or component IDs.
They see: "Primary document identified" + "Supporting attachments separated".
"""
from __future__ import annotations
import json
import logging
import hashlib
from typing import Optional
from .models import ParsedDocument, ParsedPage, SegmentationResult, DocumentComponent, AttachmentSummary

logger = logging.getLogger(__name__)

# Confidence thresholds
CONF_HIGH   = 0.80
CONF_MEDIUM = 0.55

# Cache keyed by (file_hash, page_count, schema_version)
_segmentation_cache: dict[str, SegmentationResult] = {}

SEGMENTATION_SCHEMA_VERSION = "v05.1"


# ── System prompt ─────────────────────────────────────────────────────────────

SEGMENTATION_SYSTEM = """You are an expert document analyst classifying pages inside a PDF.

Your job: decide which pages are the MAIN document and which are SUPPORTING ATTACHMENTS.

MAIN document pages contain:
- Legal agreement, contract, or lease terms (Schedule A, TLSA, lease schedule, award terms)
- Core financial terms (fixed charge, term in months, award amount, original value)
- Parties in legal context (lessee, lessor, grantee, grantor, between the parties)
- Signature or execution blocks
- Bank statement summary pages (beginning balance, ending balance, account summary)
- Board minutes (attendees, motions, votes, chair, secretary)
- Invoice or receipt with billing information
- Grant award notification or agreement

SUPPORTING ATTACHMENT pages contain:
- Vehicle, equipment, or body specification tables
- Customer or vendor proposals (Prepared For, SPO Number, Power Unit Specifications)
- Dense technical spec tables (model numbers, weights, dimensions, part codes)
- Boilerplate legal notices with no financial content
- Pages that say "This Page Intentionally Left Blank"

RULES:
1. Signature pages belong to the preceding main document — do NOT classify them as attachments
2. When in doubt, classify as MAIN — false splitting is worse than missing a split
3. Return confidence: how certain you are this classification is correct
4. If the entire file is clearly one document type, return bundle_detected: false
5. For vehicle lease bundles: the FIRST page almost always contains the core financial terms table (original value, term in months, fixed charge, mileage rate). Even if it starts with a header or logo, classify it as MAIN.
6. A page that contains ANY of these is MAIN: fixed charge, term in months, original value, schedule number, lessee, lessor, award amount, beginning balance, ending balance.
7. A page is SUPPORTING only if it contains NONE of the above AND is clearly a spec sheet, proposal, or boilerplate.

Return ONLY valid JSON. No markdown, no explanation."""

SEGMENTATION_USER_TEMPLATE = """Classify each page of this PDF as MAIN or ATTACHMENT.

Filename: {filename}
Total pages: {page_count}

Page summaries (up to 600 chars of each page):
{page_summaries}

Return this exact JSON:
{{
  "bundle_detected": true or false,
  "bundle_confidence": 0.0 to 1.0,
  "primary_document_description": "one phrase describing the main document",
  "pages": [
    {{
      "page_number": 1,
      "role": "main" or "attachment" or "skip",
      "component_group": "a short label grouping related pages e.g. schedule_a_lease or vehicle_proposal",
      "confidence": 0.0 to 1.0,
      "reason": "one short phrase"
    }}
  ],
  "attachment_summaries": [
    {{
      "component_group": "same label as above",
      "name": "human-readable name e.g. Vehicle Purchasing Proposal",
      "pages": [5, 6, 7],
      "summary": "one sentence description",
      "key_identifiers": ["SPO 006-845", "Vendor: INTL International Truck"]
    }}
  ]
}}"""


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _seg_cache_key(parsed_doc: ParsedDocument, model: str) -> str:
    base = parsed_doc.file_hash or hashlib.md5(
        (parsed_doc.full_text or "")[:2000].encode("utf-8", errors="ignore")
    ).hexdigest()
    h = hashlib.md5()
    h.update(f"{base}:{model}:{SEGMENTATION_SCHEMA_VERSION}".encode())
    return h.hexdigest()


# ── Page summarization ────────────────────────────────────────────────────────

def _summarize_pages(pages: list[ParsedPage]) -> str:
    """Build a compact multi-page summary for the classification prompt."""
    lines = []
    for pg in pages:
        text = (pg.text or "").strip()
        # Use up to 600 chars — enough to get past headers into financial content
        # Important for scanned PDFs where OCR text starts with logo/header before terms
        preview = " ".join(text[:800].split())[:600]
        lines.append(f"Page {pg.page_number} [{pg.extractor}]: {preview}")
    return "\n".join(lines)


# ── AI classification ─────────────────────────────────────────────────────────

def _classify_pages(parsed_doc: ParsedDocument, provider) -> Optional[dict]:
    """
    Single AI call to classify all pages.
    Returns parsed JSON dict or None on failure.
    """
    if not parsed_doc.pages:
        return None

    page_summaries = _summarize_pages(parsed_doc.pages)
    user_prompt = SEGMENTATION_USER_TEMPLATE.format(
        filename=parsed_doc.source_file,
        page_count=parsed_doc.page_count,
        page_summaries=page_summaries,
    )

    # Use a lightweight JSON schema — page classification doesn't need strict enforcement
    classification_schema = {
        "name": "page_classification",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "bundle_detected":              {"type": "boolean"},
                "bundle_confidence":            {"type": "number"},
                "primary_document_description": {"type": "string"},
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page_number":    {"type": "integer"},
                            "role":           {"type": "string"},
                            "component_group":{"type": "string"},
                            "confidence":     {"type": "number"},
                            "reason":         {"type": "string"}
                        }
                    }
                },
                "attachment_summaries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "component_group":  {"type": "string"},
                            "name":            {"type": "string"},
                            "pages":           {"type": "array", "items": {"type": "integer"}},
                            "summary":         {"type": "string"},
                            "key_identifiers": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        }
    }

    try:
        result = provider.extract_structured(
            system=SEGMENTATION_SYSTEM,
            user=user_prompt,
            json_schema=classification_schema,
            max_tokens=2000,
        )
        return result
    except Exception as e:
        logger.warning(f"Page classification AI call failed: {e}")
        return None


# ── Component building ────────────────────────────────────────────────────────

def _build_components(
    parsed_doc: ParsedDocument,
    classification: dict,
) -> SegmentationResult:
    """
    Build a SegmentationResult from AI classification output.
    Handles edge cases: missing pages, all-main, all-attachment.
    """
    page_map = {pg.page_number: pg for pg in parsed_doc.pages}
    classified_pages = {p["page_number"]: p for p in classification.get("pages", [])}

    bundle_detected   = classification.get("bundle_detected", False)
    bundle_confidence = float(classification.get("bundle_confidence", 0.0))
    primary_desc      = classification.get("primary_document_description", parsed_doc.source_file)

    # Confidence band
    if bundle_confidence >= CONF_HIGH:
        confidence_band = "high"
    elif bundle_confidence >= CONF_MEDIUM:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    # If low confidence or no bundle, return everything as primary
    if not bundle_detected or confidence_band == "low":
        primary_pages = list(parsed_doc.pages)
        note = None
        if bundle_detected and confidence_band == "low":
            note = "Document may contain multiple sections; extracted conservatively."
        return SegmentationResult(
            bundle_detected=False,
            bundle_confidence=bundle_confidence,
            confidence_band=confidence_band,
            primary_component=DocumentComponent(
                component_id="primary",
                component_group="full_document",
                role="main",
                pages=[pg.page_number for pg in primary_pages],
                description=primary_desc,
                confidence=bundle_confidence,
            ),
            attachment_components=[],
            conservative_note=note,
            source_file=parsed_doc.source_file,
        )

    # Sort pages into main vs attachment
    main_pages = []
    attachment_groups: dict[str, list[int]] = {}

    for pg in parsed_doc.pages:
        cp = classified_pages.get(pg.page_number)
        if cp is None:
            # Unclassified pages go to main
            main_pages.append(pg.page_number)
            continue
        role  = cp.get("role", "main")
        group = cp.get("component_group", "unknown")
        if role in ("attachment", "skip"):
            if role == "skip":
                continue
            attachment_groups.setdefault(group, []).append(pg.page_number)
        else:
            main_pages.append(pg.page_number)

    # Ensure at least some primary pages exist
    if not main_pages:
        main_pages = [pg.page_number for pg in parsed_doc.pages]
        attachment_groups = {}

    # Build primary component
    primary = DocumentComponent(
        component_id="primary",
        component_group="primary_document",
        role="main",
        pages=sorted(main_pages),
        description=primary_desc,
        confidence=bundle_confidence,
    )

    # Build attachment summaries from AI output
    ai_summaries = {s["component_group"]: s for s in classification.get("attachment_summaries", [])}
    attachments: list[AttachmentSummary] = []

    for group, page_nums in attachment_groups.items():
        ai_sum = ai_summaries.get(group, {})
        attachments.append(AttachmentSummary(
            component_id=f"attachment_{len(attachments) + 1}",
            component_group=group,
            pages=sorted(page_nums),
            name=ai_sum.get("name", group.replace("_", " ").title()),
            summary=ai_sum.get("summary", "Supporting attachment."),
            key_identifiers=ai_sum.get("key_identifiers", []),
            attachment_role="supporting_document",
        ))

    # Note for medium confidence
    note = None
    if confidence_band == "medium":
        note = "Document contains multiple sections; primary document extracted conservatively."

    return SegmentationResult(
        bundle_detected=True,
        bundle_confidence=bundle_confidence,
        confidence_band=confidence_band,
        primary_component=primary,
        attachment_components=attachments,
        conservative_note=note,
        source_file=parsed_doc.source_file,
    )


# ── Primary page extraction ───────────────────────────────────────────────────

def get_primary_pages(
    parsed_doc: ParsedDocument,
    result: SegmentationResult,
) -> list[ParsedPage]:
    """Return only the ParsedPage objects that belong to the primary component."""
    primary_page_nums = set(result.primary_component.pages)
    return [pg for pg in parsed_doc.pages if pg.page_number in primary_page_nums]


def build_primary_document(
    parsed_doc: ParsedDocument,
    result: SegmentationResult,
) -> ParsedDocument:
    """
    Return a new ParsedDocument containing only the primary component pages.
    Used to scope canonical AI extraction to the main document only.
    """
    primary_pages = get_primary_pages(parsed_doc, result)
    primary_set = set(result.primary_component.pages)
    primary_text = "\n\n".join(
        f"[Page {pg.page_number}]\n{pg.text}"
        for pg in primary_pages if pg.text.strip()
    )
    return ParsedDocument(
        source_file=parsed_doc.source_file,
        file_hash=parsed_doc.file_hash,
        mime_type=parsed_doc.mime_type,
        full_text=primary_text,
        page_count=len(primary_pages),
        pages=primary_pages,
        tables=[
            t for t in parsed_doc.tables
            if (t.get("page_number") if isinstance(t, dict) else t.page_number)
            in set(result.primary_component.pages)
        ],
        extraction_chain=parsed_doc.extraction_chain + ["segmented_primary"],
        primary_extractor=parsed_doc.primary_extractor,
        confidence=parsed_doc.confidence,
        weak_pages=[p for p in parsed_doc.weak_pages if p in set(result.primary_component.pages)],
        ocr_pages=[p for p in parsed_doc.ocr_pages if p in primary_set],
        vision_pages=[p for p in parsed_doc.vision_pages if p in primary_set],
        warnings=parsed_doc.warnings,
        errors=parsed_doc.errors,
    )




# ── Segmentation gate ─────────────────────────────────────────────────────────

# Strong bundle-signal keywords — any one of these triggers segmentation
_BUNDLE_KEYWORDS = [
    "schedule a",
    "schedule b",
    "schedule c",
    "exhibit",
    "appendix",
    "addendum",
    "customer proposal",
    "vehicle purchasing",
    "power unit specifications",
    "body specifications",
    "prepared for:",
    "spo number",
    "tlsa",
]

# Per-page title/header patterns that indicate a new logical section starting
_SECTION_HEADER_PATTERNS = [
    "customer proposal",
    "vehicle purchasing",
    "power unit specifications",
    "body specifications",
    "prepared for",
    "spo number",
    "exhibit ",
    "appendix ",
    "addendum ",
    "schedule a",
    "schedule b",
    "schedule c",
]


def _has_header_change(pages: list) -> bool:
    """
    Detect major title/header changes across pages — a strong bundle signal.
    Looks at the first 100 chars of each page for section-header keywords.
    Returns True if more than one distinct section header type is found.
    """
    seen_headers = set()
    for pg in pages:
        preview = (pg.text or "")[:200].lower()
        for pat in _SECTION_HEADER_PATTERNS:
            if pat in preview:
                seen_headers.add(pat)
    return len(seen_headers) >= 2


def _has_party_change(pages: list) -> bool:
    """
    Detect strong party-name changes across pages.
    If page 1 names a party not present in later pages, or vice versa,
    this suggests different logical documents with different principals.
    Uses a simple heuristic: look for "Prepared For:" blocks that name
    a different entity than the primary lessor/lessee pattern.
    """
    first_page_text = (pages[0].text or "").lower() if pages else ""
    has_lessor     = "lessee" in first_page_text or "lessor" in first_page_text
    has_proposal   = any("prepared for" in (pg.text or "").lower() for pg in pages[2:])
    return has_lessor and has_proposal


def _should_segment(parsed_doc: ParsedDocument) -> bool:
    """
    Heuristic gate: decide whether to run the segmentation AI call.

    Conditions to run segmentation (ALL must be true):
    1. File is large enough to be a bundle (5+ pages)
    2. At least one bundle signal is present:
       - Strong keyword match (Schedule A, TLSA, Exhibit, Customer Proposal, etc.)
       - Major title/header change detected across pages
       - Strong party-name change (lessor/lessee + later 'Prepared For' block)

    No document family is hard-excluded — bank statements, board minutes, grant
    letters can all theoretically be bundles. The gate is purely content-based.
    """
    if not parsed_doc.pages or len(parsed_doc.pages) < 5:
        return False

    text_lower = (parsed_doc.full_text or "").lower()

    # Signal 1: strong keyword match
    if any(kw in text_lower for kw in _BUNDLE_KEYWORDS):
        return True

    # Signal 2: major header/title change across pages
    if _has_header_change(parsed_doc.pages):
        return True

    # Signal 3: party-name discontinuity
    if _has_party_change(parsed_doc.pages):
        return True

    return False

# ── Main entry point ──────────────────────────────────────────────────────────

def segment(
    parsed_doc: ParsedDocument,
    provider,
) -> SegmentationResult:
    """
    Classify pages and identify logical document components.

    Returns a SegmentationResult with:
      - primary_component: the main document pages
      - attachment_components: supporting attachment summaries
      - bundle_detected: whether multiple logical sections were found
      - confidence_band: high / medium / low

    If the AI call fails or confidence is low, falls back to treating
    the entire document as primary (v4 behavior).
    """
    # Gate: skip segmentation entirely for documents that are unlikely to be bundles.
    # This avoids an extra AI call on invoices, bank statements, board minutes, etc.
    # Only runs the segmentation AI call when there are 5+ pages AND bundle keywords present.
    if not parsed_doc.pages or not _should_segment(parsed_doc):
        return SegmentationResult(
            bundle_detected=False,
            bundle_confidence=1.0,
            confidence_band="high",
            primary_component=DocumentComponent(
                component_id="primary",
                component_group="full_document",
                role="main",
                pages=[pg.page_number for pg in parsed_doc.pages],
                description=parsed_doc.source_file,
                confidence=1.0,
            ),
            attachment_components=[],
            conservative_note=None,
            source_file=parsed_doc.source_file,
        )

    # Check cache
    model_name = getattr(provider, "model", "unknown")
    cache_key  = _seg_cache_key(parsed_doc, model_name)
    if cache_key in _segmentation_cache:
        logger.info(f"Segmentation cache hit: {parsed_doc.source_file}")
        return _segmentation_cache[cache_key]

    # AI classification
    classification = _classify_pages(parsed_doc, provider)

    if classification is None:
        # AI failed — treat as single document
        logger.warning(f"Segmentation AI failed for {parsed_doc.source_file}, using full document")
        result = SegmentationResult(
            bundle_detected=False,
            bundle_confidence=0.0,
            confidence_band="low",
            primary_component=DocumentComponent(
                component_id="primary",
                component_group="full_document",
                role="main",
                pages=[pg.page_number for pg in parsed_doc.pages],
                description=parsed_doc.source_file,
                confidence=0.0,
            ),
            attachment_components=[],
            conservative_note="Document may contain multiple sections; extracted conservatively.",
            source_file=parsed_doc.source_file,
        )
        return result

    result = _build_components(parsed_doc, classification)

    # Cache and return
    _segmentation_cache[cache_key] = result
    logger.info(
        f"Segmentation: {parsed_doc.source_file} | "
        f"bundle={result.bundle_detected} | "
        f"confidence={result.bundle_confidence:.2f} ({result.confidence_band}) | "
        f"primary_pages={result.primary_component.pages} | "
        f"attachments={len(result.attachment_components)}"
    )
    return result
