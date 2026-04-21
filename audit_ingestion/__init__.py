from .router import ingest_one
from .models import (
    AuditEvidence, IngestionResult, ParsedDocument,
    DocumentComponent, AttachmentSummary, SegmentationResult,
)
from .segmenter import segment, build_primary_document, get_primary_pages
from .financial_classifier import classify_financial_file, is_financial_file
from .readiness import compute_readiness, apply_readiness
from .providers import get_provider

__version__ = "5.0.0"
__all__ = [
    "ingest_one", "AuditEvidence", "IngestionResult", "ParsedDocument",
    "DocumentComponent", "AttachmentSummary", "SegmentationResult",
    "segment", "build_primary_document", "get_primary_pages",
    "classify_financial_file", "is_financial_file",
    "get_provider",
]
