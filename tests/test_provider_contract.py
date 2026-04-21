"""Tests for provider structured output contract and model constants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from audit_ingestion.providers.openai_provider import (
    CANONICAL_MODEL, VISION_MODEL, RESCUE_MODEL, DEFAULT_MODEL
)


def test_default_model_is_canonical():
    assert DEFAULT_MODEL == CANONICAL_MODEL


def test_canonical_model_is_gpt54():
    assert CANONICAL_MODEL == "gpt-5.4"


def test_vision_model_is_gpt54():
    assert VISION_MODEL == "gpt-5.4"


def test_rescue_model_is_gpt54_pro():
    assert RESCUE_MODEL == "gpt-5.4-pro"


def test_provider_has_extract_structured():
    from audit_ingestion.providers.openai_provider import OpenAIProvider
    assert hasattr(OpenAIProvider, "extract_structured")


def test_provider_has_extract_text_from_page_images():
    from audit_ingestion.providers.openai_provider import OpenAIProvider
    assert hasattr(OpenAIProvider, "extract_text_from_page_images")


def test_provider_has_responses_call():
    from audit_ingestion.providers.openai_provider import OpenAIProvider
    assert hasattr(OpenAIProvider, "_responses_call")


def test_provider_does_not_have_chat_completions_as_primary():
    """Ensure extract_structured uses _responses_call, not chat.completions."""
    import inspect
    from audit_ingestion.providers.openai_provider import OpenAIProvider
    src = inspect.getsource(OpenAIProvider.extract_structured)
    assert "_responses_call" in src
    # Should NOT use chat.completions.create directly
    assert "chat.completions.create" not in src


def test_get_provider_openai_only():
    from audit_ingestion.providers.base import get_provider
    with pytest.raises(ValueError):
        get_provider("anthropic")
    with pytest.raises(ValueError):
        get_provider("stub")


def test_extractor_fast_lane_exists():
    from audit_ingestion.extractor import extract_fast
    assert callable(extract_fast)


def test_extractor_deep_lane_exists():
    from audit_ingestion.extractor import extract_deep
    assert callable(extract_deep)


def test_extractor_mode_routing(tmp_path):
    """extract() routes to fast or deep based on mode."""
    from audit_ingestion.extractor import extract
    test_file = tmp_path / "test.txt"
    test_file.write_text("Sample document content for testing.")
    # Both modes should work on non-PDF
    r_fast = extract(str(test_file), mode="fast")
    r_deep = extract(str(test_file), mode="deep")
    assert r_fast.source_file == "test.txt"
    assert r_deep.source_file == "test.txt"


def test_extractor_limits_defined():
    from audit_ingestion import extractor
    assert extractor.MAX_PDF_PAGES_FAST == 40
    assert extractor.MAX_OCR_PAGES == 6
    assert extractor.MAX_VISION_PAGES == 2
    assert extractor.MIN_CHARS_ACCEPTABLE == 350
    assert extractor.MIN_CHARS_WEAK == 150
    assert extractor.MIN_CHARS_CRITICAL == 60


def test_primary_extractor_computed_from_pages(tmp_path):
    """primary_extractor should reflect actual page extractor, not just chain[0]."""
    import pandas as pd
    test_file = tmp_path / "data.csv"
    df = pd.DataFrame({"Account": ["Cash"], "Amount": [1000]})
    df.to_csv(test_file, index=False)
    from audit_ingestion.extractor import extract
    result = extract(str(test_file))
    assert result.primary_extractor == "direct"
