"""
audit_ingestion_v05/audit_ingestion/router.py
Pipeline orchestrator with v05 segmentation step.

Stage 1: Extract           → ParsedDocument (page-aware, v4)
Stage 2: Segment (NEW)     → SegmentationResult (primary + attachments)
Stage 3: Canonical AI      → AuditEvidence (primary pages only)
Stage 4: Normalize         → normalized evidence
Stage 5: Score + annotate  → IngestionResult with segmentation info
"""
from __future__ import annotations
import logging
import threading
import time
from pathlib import Path
from typing import Optional
from .models import (
    AuditEvidence, IngestionResult, ExtractionMeta, Flag,
    MERGE_KEEP_NATIVE, MERGE_REPLACE, MERGE_APPEND,
    TRIGGER_WEAK_NATIVE, TRIGGER_MISSING_FIELDS, TRIGGER_IMAGE_FILE,
    TRIGGER_REVIEWER_REQUESTED, MODE_STANDARD,
    ERR_CANONICAL_RERUN_FAILED, ERR_MISSING_FIELDS_POST_VISION,
    STATE_EXTRACTED_NATIVE, STATE_EXTRACTED_OCR, STATE_VISION_REQUESTED,
    STATE_VISION_COMPLETED, STATE_MERGE_APPLIED, STATE_CANONICAL_COMPLETED,
    STATE_REVIEWER_NEEDED, STATE_READY, STATE_EXCEPTION,
    SegmentationResult, DocumentComponent, AttachmentSummary, DocumentFamily,
)
from .extractor import (
    extract, render_page_image_cached,
    run_vision_on_pages, run_full_document_vision,
    identify_pages_needing_vision,
    identify_pages_needing_vision_for_missing_fields,
    classify_page_type,
    _compute_page_improvement_score,
    _extract_handwriting_from_vision,
    _detect_high_risk_field_changes,
    get_vision_prompt_for_family,
    VISION_AUTO, VISION_STANDARD_ONLY,
    VISION_FORCE_ALL, VISION_FORCE_SELECTED, VISION_RETRY,
)
from .normalizers import normalize_evidence
from .financial_classifier import classify_financial_file, is_financial_file, TYPE_NOT_FINANCIAL
from .canonical import extract_canonical
from .readiness import apply_readiness

logger = logging.getLogger(__name__)

_AI_SEMAPHORE  = threading.Semaphore(2)
_OCR_SEMAPHORE = threading.Semaphore(2)

ROUTER_BUILD = "v05.0"


def ingest_one(
    path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    mode: str = "fast",
    allow_rescue: bool = False,
    financial_type_override: Optional[str] = None,
    financial_sheet_override: Optional[str] = None,
    bypass_cache: bool = False,
    vision_mode: str = VISION_AUTO,
    vision_pages: Optional[list[int]] = None,
    vision_prompt: Optional[str] = None,
) -> IngestionResult:
    """
    Ingest one document through the v05 pipeline.
    Segmentation runs automatically — the user never configures it.
    financial_type_override: if set, skips Layer 1/2 classification and
    uses this type directly (user_confirmed finality). Used after reclassification.
    """
    input_p = Path(path)
    engine_chain: list[str] = []
    errors: list[str] = []
    stage_timings: dict[str, float] = {}

    if not input_p.exists():
        return IngestionResult(
            status="failed",
            errors=["File not found"],
            evidence=AuditEvidence(
                source_file=input_p.name,
                flags=[Flag(type="file_not_found", description="File not found", severity="critical")]
            ),
        )

    # Provider init
    provider = None
    if api_key:
        try:
            from .providers import get_provider
            provider = get_provider("openai", api_key=api_key, model=model)
        except Exception as e:
            errors.append(f"Provider init failed: {e}")
            logger.error(f"Provider init: {e}")
    else:
        errors.append("No API key — extraction only, no canonical analysis")

    # Stage 1: Extract
    t0 = time.perf_counter()
    try:
        parsed_doc = extract(path, provider=provider, mode=mode,
                             ocr_semaphore=_OCR_SEMAPHORE)
        stage_timings["extraction"] = round(time.perf_counter() - t0, 3)
        engine_chain.extend(parsed_doc.extraction_chain)
        if parsed_doc.errors:
            errors.extend(parsed_doc.errors)
    except Exception as e:
        stage_timings["extraction"] = round(time.perf_counter() - t0, 3)
        errors.append(f"Extraction failed: {e}")
        return IngestionResult(
            status="failed", errors=errors,
            evidence=AuditEvidence(
                source_file=input_p.name,
                flags=[Flag(type="extraction_error", description=str(e), severity="critical")]
            ),
        )

    meta = ExtractionMeta(
        primary_extractor=parsed_doc.primary_extractor,
        pages_processed=parsed_doc.page_count,
        weak_pages_count=len(parsed_doc.weak_pages),
        ocr_pages_count=len(parsed_doc.ocr_pages),
        vision_pages_count=len(parsed_doc.vision_pages),
        total_chars=len(parsed_doc.full_text),
        overall_confidence=parsed_doc.confidence,
        needs_human_review=not parsed_doc.is_sufficient,
        warnings=parsed_doc.warnings,
        errors=errors,
    )

    # Minimal evidence object — always available for early returns
    # Gets replaced by canonical AI extraction if pipeline proceeds
    evidence = AuditEvidence(
        source_file=input_p.name,
        raw_text=parsed_doc.full_text,
        tables=[t if isinstance(t, dict) else t.model_dump() for t in parsed_doc.tables],
        extraction_meta=meta,
    )

    # Image file: auto-upgrade to vision_first mode (Items 13, 18)
    if parsed_doc.primary_extractor == "direct_image" and vision_mode == VISION_AUTO:
        vision_mode = VISION_FORCE_ALL
        evidence.vision_trigger_reason = TRIGGER_IMAGE_FILE
        evidence.processing_mode = "vision_first"
        logger.info(f"Image file detected — upgrading to vision_first mode: {input_p.name}")

    # Stage 1b: Financial classification (CSV/Excel files)
    financial_data: Optional[dict] = None
    if is_financial_file(path):
        t_fin = time.perf_counter()
        try:
            financial_data = classify_financial_file(
                path,
                provider=provider,
                type_override=financial_type_override,
                sheet_override=financial_sheet_override,
            )
            stage_timings["financial_classification"] = round(time.perf_counter() - t_fin, 3)
            engine_chain.append("financial_classified")
            logger.info(
                f"Financial: {input_p.name} → "
                f"{financial_data.get('doc_type','?')} "
                f"(finality={financial_data.get('finality_state','?')})"
            )
        except Exception as e:
            stage_timings["financial_classification"] = round(time.perf_counter() - t_fin, 3)
            logger.warning(f"Financial classification failed: {e}")
            financial_data = None

    # Early gate: stop here if financial classification requires user confirmation.
    # Do not run segmentation or canonical AI — both waste credits and are
    # meaningless until the user confirms the document type.
    if financial_data and financial_data.get("finality_state") == "review_required":
        logger.info(
            f"review_required — stopping pipeline for {input_p.name} "
            f"(doc_type={financial_data.get('doc_type','?')}). "
            f"Awaiting user confirmation before canonical extraction."
        )
        _annotate_with_financial_data(evidence, financial_data)

        # Set family/subtype from financial classification so the summary
        # table shows the correct type instead of "other"
        _fin_doc_type = financial_data.get("doc_type", "")
        _fin_type_to_family = {
            "general_ledger":             ("accounting_report", "general_ledger"),
            "journal_entry_listing":      ("accounting_report", "journal_entry_listing"),
            "trial_balance_unknown_year": ("accounting_report", "trial_balance"),
            "trial_balance_current":      ("accounting_report", "trial_balance_current"),
            "trial_balance_prior_year":   ("accounting_report", "trial_balance_prior_year"),
            "budget":                     ("accounting_report", "budget"),
            "bank_statement_csv":         ("bank_cash_activity", "bank_statement_csv"),
            "chart_of_accounts":          ("accounting_report", "chart_of_accounts"),
        }
        if _fin_doc_type in _fin_type_to_family:
            _fam_str, _sub_str = _fin_type_to_family[_fin_doc_type]
            try:
                evidence.family = DocumentFamily(_fam_str)
            except ValueError:
                pass
            evidence.subtype = _sub_str

        apply_readiness(evidence)
        score = _score(evidence)
        evidence.extraction_meta.overall_confidence = score
        stage_timings["segmentation"]  = 0.0
        stage_timings["canonical_ai"]  = 0.0
        stage_timings["total"] = round(sum(
            v for k, v in stage_timings.items()
            if isinstance(v, float) and k != "total"
        ), 3)
        evidence.document_specific["_stage_timings"] = stage_timings
        return IngestionResult(
            source_file=path,
            # Classification may still need reviewer confirmation, but the file
            # was successfully processed and financial facts were extracted.
            # Showing this as partial confuses users into thinking data is
            # missing or extraction failed.
            status="success",
            evidence=evidence,
            errors=errors,
            engine_chain=engine_chain,
        )

    # Stage 2: Segment (NEW in v05)
    # Also skipped for financial structured files — CSVs/Excel have no page bundles
    segmentation: Optional[SegmentationResult] = None
    extraction_doc = parsed_doc  # what gets passed to canonical AI

    if provider is not None and parsed_doc.full_text and len(parsed_doc.pages) > 2             and not financial_data:  # financial files never need bundle segmentation
        t1 = time.perf_counter()
        try:
            from .segmenter import segment, build_primary_document
            segmentation = segment(parsed_doc, provider)
            stage_timings["segmentation"] = round(time.perf_counter() - t1, 3)
            engine_chain.append("segmented")

            if segmentation.bundle_detected:
                # Scope canonical extraction to primary component pages only
                extraction_doc = build_primary_document(parsed_doc, segmentation)
                logger.info(
                    f"Bundle detected: {parsed_doc.source_file} | "
                    f"primary={segmentation.primary_page_count} pages | "
                    f"attachments={len(segmentation.attachment_components)}"
                )
            else:
                logger.info(f"No bundle: {parsed_doc.source_file} — single document")

        except Exception as e:
            stage_timings["segmentation"] = round(time.perf_counter() - t1, 3)
            logger.warning(f"Segmentation failed: {e} — proceeding with full document")
            segmentation = None
    else:
        stage_timings["segmentation"] = 0.0

    # Attach financial data to extraction doc so canonical AI can use it
    if financial_data and hasattr(extraction_doc, '__dict__'):
        # Fix: use setattr directly — __fields__ check was deprecated Pydantic v1 pattern
        try:
            setattr(extraction_doc, '_financial_data', financial_data)
        except Exception:
            pass  # ParsedDocument is a dataclass — setattr always works

    if provider is not None and extraction_doc.full_text:
        t2 = time.perf_counter()
        with _AI_SEMAPHORE:
            try:
                evidence = extract_canonical(extraction_doc, provider, mode=mode, bypass_cache=bypass_cache)
                engine_chain.append("canonical_ai")
            except Exception as e:
                errors.append(f"Canonical extraction failed: {e}")
                logger.error(f"Canonical: {e}")
                evidence = AuditEvidence(
                    source_file=input_p.name,
                    raw_text=extraction_doc.full_text,
                    tables=[t if isinstance(t, dict) else t.model_dump()
                            for t in extraction_doc.tables],
                    extraction_meta=meta,
                    flags=[Flag(type="canonical_failed", description=str(e), severity="critical")]
                )
                engine_chain.append("canonical_failed")
        stage_timings["canonical_ai"] = round(time.perf_counter() - t2, 3)

        # Stage 3b: Rescue (allow_rescue path — unchanged from v04)
        if allow_rescue and parsed_doc.weak_pages and provider is not None:
            worst_pages = sorted(
                [p for p in parsed_doc.pages if p.page_number in parsed_doc.weak_pages],
                key=lambda p: p.char_count
            )[:2]
            if worst_pages:
                t_rescue = time.perf_counter()
                try:
                    from .providers.openai_provider import RESCUE_MODEL
                    rescue_texts = []
                    with _AI_SEMAPHORE:
                        for pg in worst_pages:
                            img = render_page_image_cached(path, pg.page_number - 1, dpi=200)
                            if not img:
                                continue
                            rescued = provider.extract_text_from_page_images(
                                images=[img],
                                prompt=f"Read page {pg.page_number} of this document image. "
                                       f"Extract all audit-relevant facts. Return plain text only.",
                                model=RESCUE_MODEL,
                            )
                            if rescued and rescued.strip():
                                rescue_texts.append(f"[Rescued page {pg.page_number}]\n{rescued.strip()}")
                                engine_chain.append(f"rescue_p{pg.page_number}")
                    if rescue_texts:
                        evidence.flags.append(Flag(
                            type="rescue_applied",
                            description=f"gpt-5.4-pro visual rescue applied to {len(rescue_texts)} page(s).",
                            severity="info",
                        ))
                        evidence.document_specific["rescued_page_text"] = "\n\n".join(rescue_texts)
                    stage_timings["rescue"] = round(time.perf_counter() - t_rescue, 3)
                except Exception as e:
                    logger.warning(f"Rescue failed: {e}")
                    stage_timings["rescue"] = 0.0

    else:
        flag_type = "no_ai" if not api_key else "no_text"
        flag_desc = "No API key — canonical extraction skipped" if not api_key else "No text extracted"
        evidence = AuditEvidence(
            source_file=input_p.name,
            raw_text=parsed_doc.full_text,
            tables=[t if isinstance(t, dict) else t.model_dump() for t in parsed_doc.tables],
            extraction_meta=meta,
            flags=[Flag(type=flag_type, description=flag_desc, severity="warning")]
        )
        engine_chain.append("extraction_only")
        stage_timings["canonical_ai"] = 0.0

    # Stage 4: Normalize
    t3 = time.perf_counter()
    try:
        evidence = normalize_evidence(evidence)
        engine_chain.append("normalized")
    except Exception as e:
        logger.warning(f"Normalization failed: {e}")
    stage_timings["normalization"] = round(time.perf_counter() - t3, 3)

    # For financial files: remove partial_extraction flags — these files are
    # extracted deterministically so "partial extraction" is not meaningful.
    # The canonical AI fires this flag when it sees structured data and assumes
    # it couldn't read everything, but we already have all the data via the
    # financial classifier.
    if financial_data and financial_data.get("doc_type") != TYPE_NOT_FINANCIAL:
        evidence.flags = [
            f for f in evidence.flags
            if f.type not in (
                "partial_extraction", "partial_extraction_visibility",
                "incomplete_extraction", "truncated_content",
            )
        ]

    # Stage 5: Annotate with segmentation info + score
    _annotate_with_segmentation(evidence, segmentation)
    if financial_data:
        _annotate_with_financial_data(evidence, financial_data)



    # Compute evidence readiness and generate questions
    apply_readiness(evidence)

    score = _score(evidence)
    ai_unavailable = any(f.type in ("canonical_failed", "no_ai") for f in evidence.flags)
    has_text = (evidence.extraction_meta.total_chars or 0) >= 200
    if ai_unavailable and has_text and score < 0.30:
        score = 0.30

    evidence.extraction_meta.overall_confidence = score
    # Deduplicate flags — same type can appear multiple times from cache replay
    seen_flag_types: set[str] = set()
    deduped_flags = []
    for _flag in evidence.flags:
        if _flag.type not in seen_flag_types:
            seen_flag_types.add(_flag.type)
            deduped_flags.append(_flag)
    evidence.flags = deduped_flags

    evidence.extraction_meta.needs_human_review = (
        score < 0.70 or
        (evidence.readiness and evidence.readiness.blocking_state == "blocking")
    )
    evidence.document_specific["_stage_timings"] = stage_timings
    stage_timings["total"] = round(sum(stage_timings.values()), 3)

    status = "success" if score >= 0.70 else ("partial" if score >= 0.30 else "failed")

    # ── Auto vision trigger: missing required fields (Item 5) ───────────────
    # If standard extraction passed but key fields are empty, auto-target those pages
    if vision_mode == VISION_AUTO and provider is not None:
        _auto_pages, _auto_reasons = identify_pages_needing_vision_for_missing_fields(
            evidence, parsed_doc
        )
        if _auto_pages:
            logger.info(
                f"Auto vision trigger (missing fields): {len(_auto_pages)} page(s) "
                f"on {input_p.name} — {list(_auto_reasons.values())[:3]}"
            )
            _auto_result = run_vision_on_pages(
                str(input_p), _auto_pages, provider,
                prompt=get_vision_prompt_for_family(
                    evidence.family.value if evidence.family else "default"
                ),
            )
            if _auto_result.get("page_texts"):
                evidence.vision_trigger_reason = TRIGGER_MISSING_FIELDS
                # Store vision text for operator visibility without triggering full rerun
                evidence.vision_page_texts.update(_auto_result["page_texts"])
                evidence.vision_applied    = True
                evidence.vision_mode_used  = "auto_missing_fields"
                evidence.vision_pages_used = _auto_result.get("pages_succeeded", [])
                engine_chain.append("vision_auto_missing_fields")

    # ── Manual vision paths ───────────────────────────────────────────────────
    if vision_mode not in (VISION_AUTO, VISION_STANDARD_ONLY) and provider is not None:
        _vision_result = None
        _vpath = str(input_p)

        # Auto-select document-family-specific prompt if not provided (Item 8)
        if not vision_prompt and evidence:
            _doc_family = evidence.family.value if evidence.family else "default"
            vision_prompt = get_vision_prompt_for_family(_doc_family)

        if vision_mode == VISION_FORCE_ALL:
            logger.info(f"Manual vision: FORCE_ALL on {input_p.name}")
            _vision_result = run_full_document_vision(_vpath, provider, prompt=vision_prompt)

        elif vision_mode == VISION_FORCE_SELECTED and vision_pages:
            logger.info(f"Manual vision: FORCE_SELECTED pages {vision_pages} on {input_p.name}")
            _vision_result = run_vision_on_pages(_vpath, vision_pages, provider, prompt=vision_prompt)

        elif vision_mode == VISION_RETRY:
            _retry_pages, _retry_reasons = identify_pages_needing_vision(parsed_doc)
            if _retry_pages:
                logger.info(f"Manual vision RETRY: {len(_retry_pages)} weak page(s) on {input_p.name}")
                _vision_result = run_vision_on_pages(_vpath, _retry_pages, provider, prompt=vision_prompt)
                _vision_result["retry_reasons"] = _retry_reasons
            else:
                logger.info(f"Manual vision RETRY: no weak pages — full pass on {input_p.name}")
                _vision_result = run_full_document_vision(_vpath, provider, prompt=vision_prompt)
                _vision_result["retry_reasons"] = {"all": "No weak pages — full document pass"}

        if _vision_result and evidence:
            _pts = _vision_result.get("page_texts", {})
            _sep = "\n\n"
            _page_strs = [f"[Page {pn}]\n{t}" for pn, t in sorted(_pts.items())]
            _vision_combined = _sep.join(_page_strs) if _pts else _vision_result.get("combined_text", "")

            # ── Field-aware vision merge (Items 1, 2, 3, 4, 9, 22) ───────────
            _vision_enhanced  = False
            _pages_replaced   = []
            _pages_appended   = []
            _high_risk_changes= []
            _delta_pages      = {}

            # Preserve original raw_text before mutation
            if not evidence.original_raw_text:
                evidence.original_raw_text = evidence.raw_text

            if _pts and parsed_doc:
                # Respect segmentation primary-doc scope (Item 4)
                _seg_data = (evidence.document_specific or {}).get("_segmentation") or {}
                _primary_pages = set(_seg_data.get("primary_pages") or [])

                for _pn, _vtext in _pts.items():
                    _pidx = _pn - 1
                    if _pidx >= len(parsed_doc.pages):
                        continue
                    _page   = parsed_doc.pages[_pidx]
                    _native = _page.native_text or _page.text or ""

                    # Skip attachment pages when segmentation identified a primary scope
                    if _primary_pages and _pn not in _primary_pages:
                        continue

                    _imp_score, _decision, _reason = _compute_page_improvement_score(_native, _vtext)
                    _has_hw, _hw_text, _hw_conf    = _extract_handwriting_from_vision(_vtext)
                    _hr_changes                    = _detect_high_risk_field_changes(_native, _vtext, _pn)

                    from .extractor import _score_text
                    _ns = _score_text(_native)
                    _vs = _score_text(_vtext)
                    _delta_pages[_pn] = {
                        "merge_decision":    _decision,
                        "merge_reason":      _reason,
                        "improvement_score": _imp_score,
                        "native_chars":      _ns["chars"],
                        "vision_chars":      _vs["chars"],
                        "added_numbers":     max(0, _vs["numbers"]   - _ns["numbers"]),
                        "added_dates":       max(0, _vs["dates"]     - _ns["dates"]),
                        "added_amounts":     max(0, _vs["currencies"] - _ns["currencies"]),
                        "has_handwriting":   _has_hw,
                        "handwriting_text":  _hw_text[:200] if _hw_text else "",
                    }

                    if _decision in (MERGE_REPLACE, MERGE_APPEND):
                        from .models import ParsedPage as _PP
                        _final_text = _vtext if _decision == MERGE_REPLACE else (
                            _native + "\n\n[Vision annotation]\n" + _vtext
                        )
                        _final_ext = "vision" if _decision == MERGE_REPLACE else "hybrid"
                        parsed_doc.pages[_pidx] = _PP(
                            page_number=_pn,
                            text=_final_text,
                            char_count=len(_final_text),
                            extractor=_final_ext,
                            confidence=0.88,
                            image_used=True,
                            native_text=_native,
                            vision_text=_vtext,
                            final_text_used=_final_text,
                            final_extractor_used=_final_ext,
                            merge_decision=_decision,
                            merge_reason=_reason,
                            improvement_score=_imp_score,
                            has_handwriting=_has_hw,
                            handwriting_text=_hw_text,
                            handwriting_confidence=_hw_conf,
                        )
                        (_pages_replaced if _decision == MERGE_REPLACE else _pages_appended).append(_pn)
                        _vision_enhanced = True
                        _high_risk_changes.extend(_hr_changes)

                if _vision_enhanced:
                    parsed_doc.full_text = "\n\n".join(
                        f"[Page {pg.page_number}]\n{pg.text}"
                        for pg in parsed_doc.pages if pg.text.strip()
                    )
                    parsed_doc.vision_pages = _pages_replaced + _pages_appended
                    logger.info(f"Vision merge: replaced={_pages_replaced} appended={_pages_appended}")

                    _canonical_before = {
                        "amounts":    len(evidence.amounts),
                        "dates":      len(evidence.dates),
                        "parties":    len(evidence.parties),
                        "facts":      len(evidence.facts),
                        "confidence": evidence.extraction_meta.overall_confidence,
                        "readiness":  evidence.readiness.readiness_status if evidence.readiness else "unknown",
                    }

                    try:
                        _enhanced_evidence = extract_canonical(
                            parsed_doc, provider, mode=mode, bypass_cache=True
                        )
                        _enhanced_evidence.original_raw_text = evidence.original_raw_text
                        _enhanced_evidence.raw_text           = evidence.original_raw_text
                        _enhanced_evidence.working_raw_text   = parsed_doc.full_text
                        evidence = _enhanced_evidence
                        engine_chain.append("canonical_ai_vision_enhanced")
                        logger.info(f"Re-ran canonical on vision-enhanced text: {input_p.name}")
                    except Exception as _ce:
                        logger.warning(f"Canonical re-run after vision failed: {_ce}")
                        evidence.document_specific = evidence.document_specific or {}
                        evidence.document_specific["_vision_canonical_error"] = str(_ce)

            # Store all vision outputs and provenance (Items 2, 3, 10, 16, 22)
            evidence.vision_applied        = True
            evidence.vision_mode_used      = vision_mode
            evidence.vision_pages_used     = _vision_result.get("pages_succeeded", [])
            evidence.vision_page_texts     = _pts
            evidence.vision_page_status    = _vision_result.get("page_status", {})
            evidence.vision_prompt         = _vision_result.get("prompt_used")
            evidence.vision_error          = _vision_result.get("error")
            evidence.vision_text           = _vision_combined
            evidence.vision_overlay_text   = _vision_combined
            evidence.working_raw_text      = parsed_doc.full_text if _vision_enhanced else evidence.raw_text

            # High-risk field changes requiring reviewer confirmation (Item 22)
            if _high_risk_changes:
                evidence.vision_changed_high_risk_fields = _high_risk_changes
                evidence.reviewer_confirmation_required  = True
                evidence.ingestion_state = STATE_REVIEWER_NEEDED

            # Delta report: before/after vision (Item 3)
            _canonical_after = {
                "amounts":    len(evidence.amounts),
                "dates":      len(evidence.dates),
                "parties":    len(evidence.parties),
                "facts":      len(evidence.facts),
                "confidence": evidence.extraction_meta.overall_confidence,
                "readiness":  evidence.readiness.readiness_status if evidence.readiness else "unknown",
            } if _vision_enhanced else {}

            evidence.vision_delta_report = {
                "pages_changed":      _pages_replaced + _pages_appended,
                "pages_replaced":     _pages_replaced,
                "pages_appended":     _pages_appended,
                "page_level_changes": _delta_pages,
                "high_risk_changes":  _high_risk_changes,
                "canonical_before":   _canonical_before if _vision_enhanced else {},
                "canonical_after":    _canonical_after,
                "canonical_reran":    _vision_enhanced,
            }

            evidence.vision_run_diagnostics = {
                "document":              input_p.name,
                "mode":                  vision_mode,
                "trigger_reason":        evidence.vision_trigger_reason,
                "pages_requested":       _vision_result.get("pages_attempted", []),
                "pages_rendered":        _vision_result.get("pages_rendered", []),
                "pages_succeeded":       _vision_result.get("pages_succeeded", []),
                "pages_failed":          _vision_result.get("pages_failed", []),
                "is_partial":            _vision_result.get("is_partial", False),
                "chars_returned":        _vision_result.get("chars_returned", 0),
                "elapsed_seconds":       _vision_result.get("elapsed_seconds", 0.0),
                "truncation_suspected":  _vision_result.get("truncation_suspected", False),
                "retry_reasons":         _vision_result.get("retry_reasons"),
                "canonical_reran":       _vision_enhanced,
                "reviewer_confirmation_required": evidence.reviewer_confirmation_required,
                "error":                 _vision_result.get("error"),
            }

            engine_chain.append(f"vision_manual:{vision_mode}")

            # Re-score and re-assess readiness after canonical re-run
            if _vision_enhanced:
                apply_readiness(evidence)
                _new_score = _score(evidence)
                evidence.extraction_meta.overall_confidence = _new_score
                status = "success" if _new_score >= 0.70 else ("partial" if _new_score >= 0.30 else "failed")

                # Vision confidence banding (Item 14)
                _v_chars = _vision_result.get("chars_returned", 0)
                _v_pages_ok = len(_vision_result.get("pages_succeeded", []))
                _v_total = max(1, len(_vision_result.get("pages_attempted", [1])))
                _v_ratio = _v_pages_ok / _v_total
                evidence.vision_confidence_document = (
                    "high" if _v_ratio >= 0.9 and _v_chars > 500 else
                    "medium" if _v_ratio >= 0.6 else "low"
                )
                evidence.canonical_confidence_before = _canonical_before.get("confidence", 0)
                evidence.canonical_confidence_after  = _new_score
                evidence.canonical_rerun_mode        = "full_rerun"
                evidence.merge_confidence = (
                    "high" if len(_pages_replaced) > 0 and not _vision_result.get("is_partial") else
                    "medium" if len(_pages_appended) > 0 else "low"
                )

    return IngestionResult(
        evidence=evidence,
        status=status,
        errors=errors,
        engine_chain=engine_chain,
    )


def _annotate_with_segmentation(
    evidence: AuditEvidence,
    segmentation: Optional[SegmentationResult],
) -> None:
    """Add segmentation info to the evidence record. Modifies in place."""
    if segmentation is None:
        return

    if segmentation.bundle_detected and segmentation.has_attachments:
        # User-facing flag — clean language, no technical internals
        attachment_names = ", ".join(a.name for a in segmentation.attachment_components)
        evidence.flags.append(Flag(
            type="bundle_detected",
            description=(
                f"Primary document identified: {segmentation.primary_component.description}. "
                f"Supporting attachments separated: {attachment_names}. "
                f"Core facts extracted from primary document only."
            ),
            severity="info",
        ))

        # Store attachment summaries in document_specific for UI display
        evidence.document_specific["_segmentation"] = {
            "bundle_detected":        True,
            "confidence_band":        segmentation.confidence_band,
            "primary_description":    segmentation.primary_component.description,
            "primary_pages":          segmentation.primary_component.pages,
            "attachments": [
                {
                    "name":            a.name,
                    "pages":           a.pages,
                    "summary":         a.summary,
                    "key_identifiers": a.key_identifiers,
                }
                for a in segmentation.attachment_components
            ],
        }

    if segmentation.conservative_note:
        evidence.flags.append(Flag(
            type="conservative_extraction",
            description=segmentation.conservative_note,
            severity="info",
        ))


def _annotate_with_financial_data(
    evidence: AuditEvidence,
    financial_data: dict,
) -> None:
    """Store financial classification results in document_specific. Modifies in place."""
    doc_type    = financial_data.get("doc_type", TYPE_NOT_FINANCIAL)
    finality    = financial_data.get("finality_state", "")
    confidence  = financial_data.get("doc_type_confidence", 0.0)

    # Store full financial data in document_specific
    evidence.document_specific["_financial"] = financial_data

    # Surface balance issues as flags
    bal = financial_data.get("balance_check", {})
    flag_level = bal.get("flag_level", "")
    if flag_level == "material_balance_difference":
        diff = bal.get("difference", 0)
        pct  = bal.get("pct_of_dr", 0)
        evidence.flags.append(Flag(
            type="material_balance_difference",
            description=(
                f"Trial balance is materially out of balance: "
                f"DR ${bal.get('dr_total',0):,.2f} vs CR ${bal.get('cr_total',0):,.2f} "
                f"(difference ${diff:,.2f} = {pct:.1f}% of DR total). "
                f"Explanation required before audit proceeds."
            ),
            severity="warning",
        ))
    elif flag_level == "balance_difference_detected":
        evidence.flags.append(Flag(
            type="balance_difference_detected",
            description=(
                f"Trial balance has a small difference: ${bal.get('difference',0):,.2f}. "
                f"Likely rounding or timing — confirm before relying on totals."
            ),
            severity="info",
        ))

    # Flag missing period for financial files
    if doc_type != TYPE_NOT_FINANCIAL and not financial_data.get("period_start"):
        evidence.flags.append(Flag(
            type="missing_period",
            description=(
                f"Could not determine fiscal period for {doc_type}. "
                f"Check filename or file header for year/period information."
            ),
            severity="info",
        ))

    # Flag TB that needs year confirmation
    if doc_type == "trial_balance_unknown_year":
        evidence.flags.append(Flag(
            type="tb_year_unconfirmed",
            description=(
                "Trial balance year not resolved. "
                "Confirm whether this is the current year or prior year TB."
            ),
            severity="warning",
        ))


def _score(ev: AuditEvidence) -> float:
    """
    Score the quality of a canonical evidence record.
    Financial files (CSV/Excel) use a separate scoring path because they
    structurally lack parties/provenance/dates — those fields are N/A, not missing.
    """
    ds = ev.document_specific or {}
    fin = ds.get("_financial", {})
    is_financial = bool(fin and fin.get("doc_type") and
                        fin["doc_type"] != "not_financial_structured_data")

    if is_financial:
        return _score_financial(ev, fin)
    return _score_document(ev)


def _score_financial(ev: AuditEvidence, fin: dict) -> float:
    """
    Score a pre-classified financial file.
    Rewards: correct classification, period detection, totals, balance check,
             canonical AI claims, no critical errors.
    Financial files do not lose points for missing parties/provenance —
    those fields are not applicable to structured data.
    """
    s = 0.0

    # 1. Classification quality (0-0.25)
    conf = float(fin.get("doc_type_confidence", 0))
    finality = fin.get("finality_state", "")
    if finality in ("trusted", "user_confirmed"):
        s += 0.25 * conf
    elif finality == "review_recommended":
        s += 0.20 * conf
    else:  # review_required
        s += 0.10 * conf

    # 2. Period detected (0-0.15)
    period_conf = float(fin.get("period_confidence", 0))
    if fin.get("period_start"):
        if period_conf >= 0.90:
            s += 0.15
        elif period_conf >= 0.65:
            s += 0.10
        else:
            s += 0.05

    # 3. Totals extracted (0-0.20)
    totals = fin.get("totals", {})
    non_error_totals = {k: v for k, v in totals.items() if not k.endswith("_error")}
    if len(non_error_totals) >= 4:
        s += 0.20
    elif len(non_error_totals) >= 2:
        s += 0.12
    elif len(non_error_totals) >= 1:
        s += 0.06

    # 4. Balance check result (0-0.15)
    bal = fin.get("balance_check", {})
    bal_flag = bal.get("flag_level", "")
    if bal_flag == "tb_balanced":
        s += 0.15   # clean balance is a positive signal
    elif bal_flag == "balance_difference_detected":
        s += 0.10   # small diff — still extracted correctly
    elif bal_flag == "material_balance_difference":
        s += 0.10   # still extracted correctly, just flagged
    elif not bal_flag and fin.get("doc_type") not in (
        "trial_balance_unknown_year", "trial_balance_current", "trial_balance_prior_year"
    ):
        s += 0.10   # non-TB types don't need a balance check

    # 5. Canonical AI quality (0-0.15)
    if ev.audit_overview and ev.audit_overview.summary:
        s += 0.05
    if ev.amounts:
        s += 0.05
    if ev.claims:
        s += min(0.05, len(ev.claims) * 0.02)

    # 6. No critical errors (0-0.10)
    critical_flags = [f for f in (ev.flags or []) if f.severity == "critical"]
    if not critical_flags:
        s += 0.10

    return round(min(s, 1.0), 3)


def _score_document(ev: AuditEvidence) -> float:
    """Score a standard document (PDF, text). Original scoring logic."""
    s = 0.0
    if ev.audit_overview and ev.audit_overview.summary:
        s += 0.20
    s += 0.07 if ev.amounts  else 0
    s += 0.07 if ev.parties  else 0
    s += 0.06 if ev.dates    else 0
    s += 0.05 if ev.facts    else 0
    if ev.claims:
        s += min(0.15, len(ev.claims) * 0.05)
    all_items = (
        [(a.provenance, a.value) for a in ev.amounts] +
        [(p.provenance, p.name) for p in ev.parties] +
        [(d.provenance, d.value) for d in ev.dates]
    )
    if all_items:
        with_prov = sum(1 for prov, _ in all_items if prov and prov.confidence > 0.5)
        s += 0.20 * (with_prov / len(all_items))
    lk = ev.link_keys
    if any([lk.party_names, lk.document_numbers, lk.invoice_numbers,
            lk.agreement_numbers, lk.recurring_amounts]):
        s += 0.10
    if ev.extraction_meta.total_chars >= 500:
        s += 0.10
    elif ev.extraction_meta.total_chars >= 200:
        s += 0.05
    return round(min(s, 1.0), 3)
