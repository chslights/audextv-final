"""
audit_ingestion_v05/audit_ingestion/providers/openai_provider.py
OpenAI provider — Responses API + Structured Outputs.

Model constants — change here only:
  CANONICAL_MODEL  = "gpt-5.4"      canonical structured extraction
  VISION_MODEL     = "gpt-5.4"      weak-page vision transcription
  RESCUE_MODEL     = "gpt-5.4-pro"  optional manual rescue only
  DEFAULT_MODEL    = CANONICAL_MODEL

PROVIDER BUILD: v05.0
"""
from __future__ import annotations
import base64
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

from .base import AIProvider

logger = logging.getLogger(__name__)

# ── Model constants — change ONLY here ───────────────────────────────────────
CANONICAL_MODEL = "gpt-5.4"
VISION_MODEL    = "gpt-5.4"
RESCUE_MODEL    = "gpt-5.4-pro"
DEFAULT_MODEL   = CANONICAL_MODEL
PROVIDER_BUILD  = "v05.1"

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False



def _parse_chunk_json(raw: str, expected_pages: list[int]) -> list[dict]:
    """
    Parse a single vision chunk response into a list of page dicts.
    Tries: (1) full JSON array, (2) embedded JSON array, (3) positional split fallback.
    Always returns one dict per expected page — missing pages get empty text.
    """
    import json as _j, re as _r
    raw = (raw or "").strip()
    try:
        data = _j.loads(raw)
        if isinstance(data, list):
            return data
    except (ValueError, TypeError):
        pass
    try:
        m = _r.search(r'\[\s*\{.*?\}\s*\]', raw, _r.DOTALL)
        if m:
            data = _j.loads(m.group(0))
            if isinstance(data, list):
                return data
    except (ValueError, TypeError):
        pass
    splits = [t.strip() for t in raw.split("--- PAGE BREAK ---")]
    return [
        {"page": expected_pages[i], "text": splits[i] if i < len(splits) else "",
         "has_handwriting": False, "_fallback": True}
        for i in range(len(expected_pages))
    ]


class OpenAIProvider(AIProvider):

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        if not HAS_OPENAI:
            raise ImportError("openai not installed. Run: pip install openai")
        self.client = openai.OpenAI(api_key=api_key)
        self.model  = model
        logger.info(f"OpenAI provider ready — model: {model} | build: {PROVIDER_BUILD}")

    # ── Core Responses API call ───────────────────────────────────────────────

    def _responses_call(
        self,
        *,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_output_tokens: int = 4000,
        json_schema: Optional[dict] = None,
    ) -> str:
        """
        Call OpenAI Responses API with explicit content blocks.
        When json_schema provided, uses structured outputs:
          text.format.type   = "json_schema"
          text.format.name   = <schema name>  ← REQUIRED at format level
          text.format.schema = <schema body>
          text.format.strict = True
        Returns raw response text.
        """
        m = model or self.model

        # Explicit content blocks — more robust than bare string content
        input_messages = [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user",   "content": [{"type": "input_text", "text": user}]},
        ]

        kwargs: dict = dict(
            model=m,
            input=input_messages,
            max_output_tokens=max_output_tokens,
        )

        if json_schema:
            schema_name   = json_schema.get("name", "audit_evidence")
            schema_strict = json_schema.get("strict", True)
            schema_body   = json_schema.get("schema", json_schema)

            fmt = {
                "type":   "json_schema",
                "name":   schema_name,
                "strict": schema_strict,
                "schema": schema_body,
            }
            kwargs["text"] = {"format": fmt}

            # Preflight validation — explicit raises, never bare assert
            if "name" not in fmt:
                raise ValueError("BUG: text.format.name is required by Responses API")
            if fmt.get("type") != "json_schema":
                raise ValueError(f"BUG: unexpected format type: {fmt.get('type')}")
            if "schema" not in fmt:
                raise ValueError("BUG: text.format.schema is required")

            logger.info(
                f"[responses.create] model={m} | "
                f"format.type={fmt['type']} | "
                f"format.name={fmt.get('name', 'MISSING')} | "
                f"format.strict={fmt.get('strict')} | "
                f"format.schema present={('schema' in fmt)} | "
                f"build={PROVIDER_BUILD}"
            )
        else:
            logger.info(
                f"[responses.create] model={m} | "
                f"plain text mode | build={PROVIDER_BUILD}"
            )

        resp = self.client.responses.create(**kwargs)
        return resp.output_text or ""

    # ── Structured canonical extraction ──────────────────────────────────────

    def extract_structured(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict,
        max_tokens: int = 4000,
    ) -> dict:
        """
        Extract structured JSON via Responses API + Structured Outputs.
        Uses CANONICAL_MODEL. Raises ValueError if empty or unparseable.
        """
        logger.info(
            f"extract_structured | model={CANONICAL_MODEL} | "
            f"schema={json_schema.get('name', 'MISSING')} | build={PROVIDER_BUILD}"
        )

        raw = self._responses_call(
            system=system,
            user=user,
            model=CANONICAL_MODEL,
            max_output_tokens=max_tokens,
            json_schema=json_schema,
        )

        if not raw or not raw.strip():
            raise ValueError("Empty response from structured extraction")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Structured output returned invalid JSON: {e}\nRaw: {raw[:500]}"
            )

    # ── Vision page transcription ─────────────────────────────────────────────

    _VISION_CHUNK_SIZE   = 6    # pages per API call — stays well inside context limits
    _VISION_MAX_TOKENS   = 4096  # per chunk

    def extract_text_from_page_images(
        self,
        *,
        images: list[bytes],
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Send page images to vision model via Responses API.
        Processes in chunks of _VISION_CHUNK_SIZE so no pages are silently dropped.
        Chunks are merged in order with '--- PAGE BREAK ---' separators.
        Returns the full combined text across all chunks.
        """
        if not images:
            return ""

        m = model or VISION_MODEL
        chunk_size = self._VISION_CHUNK_SIZE
        total = len(images)
        logger.info(
            f"extract_text_from_page_images | model={m} | "
            f"total_images={total} | chunk_size={chunk_size} | build={PROVIDER_BUILD}"
        )

        import json as _json

        # Each chunk returns structured JSON: [{"page": N, "text": "..."}, ...]
        # We accumulate all dicts and return ONE merged JSON array — never raw text joins.
        all_page_dicts: list[dict] = []

        for chunk_start in range(0, total, chunk_size):
            chunk = images[chunk_start: chunk_start + chunk_size]
            chunk_end = chunk_start + len(chunk)
            chunk_page_nums = list(range(chunk_start + 1, chunk_end + 1))
            logger.debug(f"Vision chunk pages {chunk_page_nums}")

            content_parts: list[dict] = []
            for img_bytes in chunk:
                b64 = base64.b64encode(img_bytes).decode()
                content_parts.append({
                    "type":      "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                    "detail":    "high",
                })

            chunk_prompt = (
                prompt
                + f" These images are pages {chunk_page_nums} of a {total}-page document."
                + " Return ONLY a JSON array, no prose:"
                + ' [{"page": N, "text": "...", "has_handwriting": true/false}]'
                + f" Page numbers must be exactly: {chunk_page_nums}"
            )
            content_parts.append({"type": "input_text", "text": chunk_prompt})

            try:
                resp = self.client.responses.create(
                    model=m,
                    input=[{"role": "user", "content": content_parts}],
                    max_output_tokens=self._VISION_MAX_TOKENS,
                )
                raw = (resp.output_text or "").strip()
                chunk_dicts = _parse_chunk_json(raw, chunk_page_nums)
                all_page_dicts.extend(chunk_dicts)

                # Validate schema (Item 11): every item must have page + text
                validated = []
                for d in chunk_dicts:
                    if not isinstance(d, dict):
                        continue
                    if "page" not in d or "text" not in d:
                        logger.warning(f"Vision: malformed item missing page/text: {str(d)[:80]}")
                        continue
                    try:
                        d["page"] = int(d["page"])
                    except (ValueError, TypeError):
                        continue
                    validated.append(d)

                # Retry once with stricter schema if items dropped (Item 11)
                if len(validated) < len(chunk_page_nums) and not validated:
                    logger.info("Vision: retrying chunk with stricter schema prompt")
                    strict_prompt = (
                        "Return ONLY valid JSON array. Each element must have exactly these keys: "
                        '"page" (integer), "text" (string), "has_handwriting" (boolean). '
                        f"Pages: {chunk_page_nums}. No prose, no markdown, no code blocks."
                    )
                    content_parts[-1] = {"type": "input_text", "text": strict_prompt}
                    try:
                        resp2 = self.client.responses.create(
                            model=m,
                            input=[{"role": "user", "content": content_parts}],
                            max_output_tokens=self._VISION_MAX_TOKENS,
                        )
                        validated = _parse_chunk_json(resp2.output_text or "", chunk_page_nums)
                    except Exception as _re:
                        logger.warning(f"Vision retry failed: {_re}")

                chunk_dicts = validated
                returned = {d.get("page") for d in chunk_dicts}
                missing  = [p for p in chunk_page_nums if p not in returned]

                # Check for duplicate page numbers (Item 12)
                seen_pages: set = set()
                deduped = []
                for d in chunk_dicts:
                    pn = d.get("page")
                    if pn not in seen_pages:
                        seen_pages.add(pn)
                        deduped.append(d)
                    else:
                        logger.warning(f"Vision: duplicate page {pn} in response — keeping first")
                chunk_dicts = deduped

                if missing:
                    logger.warning(f"Vision chunk: pages {missing} missing after validation")
                    for mp in missing:
                        all_page_dicts.append({"page": mp, "text": "", "_missing": True})

            except Exception as e:
                logger.warning(f"Vision chunk {chunk_page_nums} failed: {e}")
                for p in chunk_page_nums:
                    all_page_dicts.append({"page": p, "text": "", "_failed": True, "_error": str(e)})

        all_page_dicts.sort(key=lambda d: d.get("page", 0))
        return _json.dumps(all_page_dicts)

    # ── PDF vision fallback ───────────────────────────────────────────────────

    def extract_text_from_pdf_vision(
        self,
        pdf_bytes: bytes,
        max_pages: int = 2,
    ) -> str:
        """Convert PDF pages to images and extract via vision. Last-resort fallback."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        img_dir = tempfile.mkdtemp()
        image_bytes_list: list[bytes] = []

        try:
            subprocess.run(
                ["pdftoppm", "-jpeg", "-r", "150", "-l", str(max_pages),
                 tmp_path, f"{img_dir}/page"],
                capture_output=True, timeout=60,
            )
            for img_file in sorted(os.listdir(img_dir)):
                if img_file.endswith(".jpg"):
                    img_path = os.path.join(img_dir, img_file)
                    with open(img_path, "rb") as f:
                        image_bytes_list.append(f.read())
                    os.unlink(img_path)
                    if len(image_bytes_list) >= max_pages:
                        break
        except Exception as e:
            logger.warning(f"pdftoppm failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
                os.rmdir(img_dir)
            except Exception:
                pass

        if not image_bytes_list:
            return ""

        return self.extract_text_from_page_images(
            images=image_bytes_list,
            prompt=(
                "Extract ALL text from these document pages faithfully. "
                "Preserve all numbers, dates, names, amounts, and terms exactly as written. "
                "Separate pages with '--- PAGE BREAK ---'."
            ),
        )
