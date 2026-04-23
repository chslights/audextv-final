"""
audit_ingestion_v04.2/audit_ingestion/providers/base.py
OpenAI-only provider base + factory.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class AIProvider(ABC):

    @abstractmethod
    def extract_structured(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict,
        max_tokens: int = 4000,
    ) -> dict:
        """Extract structured JSON via Responses API + Structured Outputs."""
        ...

    @abstractmethod
    def extract_text_from_page_images(
        self,
        *,
        images: list[bytes],
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """Extract text from page images via vision model."""
        ...

    def extract_text_from_pdf_vision(
        self,
        pdf_bytes: bytes,
        max_pages: int = 2,
    ) -> str:
        """Full PDF vision fallback. Override in concrete providers."""
        return ""


def get_provider(
    provider_name: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> AIProvider:
    """Factory — OpenAI only."""
    if provider_name != "openai":
        raise ValueError(f"Only 'openai' is supported. Got: '{provider_name}'")
    from .openai_provider import OpenAIProvider, DEFAULT_MODEL
    return OpenAIProvider(api_key=api_key, model=model or DEFAULT_MODEL)
