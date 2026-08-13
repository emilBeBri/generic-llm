"""Gemini adapter.

Single non-streaming POST to `{base}/models/{model}:generateContent` via
`gllm._http`. This was the only conversion off a vendor SDK that was a rewrite
rather than a call-site swap, because `google-genai` did not just transport the
call — it renamed everything. The REST wire format is **camelCase**
(`generationConfig`, `maxOutputTokens`, `usageMetadata`, `promptTokenCount`,
`finishReason`) while the SDK exposed snake_case, and `gllm.usage.from_gemini`
reads the snake_case names. `_snake_keys` converts the response once so that
mapper, `_finish_reason` and every existing test stay untouched.

Structured output:
- response_mime_type = "application/json"
- response_json_schema = <the schema dict>

Plain JSON mode (no schema):
- response_mime_type = "application/json"
"""

from __future__ import annotations

import base64
import os
import re

from .._http import get_json, post_json, wrap
from ..config import resolve_base_url
from ..domain import Request, Response
from ..ports import LLMProvider
from ..reasoning import gemini_thinking_budget
from ..usage import from_gemini
from ._capabilities import is_text_generation_model

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
# Kept OUT of the base URL on purpose. The SDK appended the version itself, so
# every override in the wild — including the jail broker's
# GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:<port>/gemini/ — points at the HOST and
# expects `/v1beta/...` to follow. Folding it into the base makes those 404.
GEMINI_API_VERSION = "v1beta"

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _snake_keys(value):
    """Recursively rewrite camelCase JSON keys to snake_case.

    The point is that `gllm.usage.from_gemini` was written against the SDK's
    snake_case attributes (`prompt_token_count`, `thoughts_token_count`). Doing
    the rename here means usage extraction, truncation detection and their tests
    need no Gemini-specific special case. Values are untouched — only keys.
    """
    if isinstance(value, dict):
        return {
            _CAMEL_BOUNDARY.sub("_", k).lower(): _snake_keys(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_snake_keys(v) for v in value]
    return value


def _candidate_text(resp) -> str:
    """What `resp.text` used to give us: the first candidate's text parts, joined.

    A part may carry `functionCall`/`inlineData` instead of `text`, and a thought
    part is flagged `thought: true` — neither is answer text, so both are skipped.
    """
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    chunks = []
    for part in (getattr(content, "parts", None) or []) if content else []:
        if getattr(part, "thought", False):
            continue
        chunk = getattr(part, "text", None)
        if chunk:
            chunks.append(chunk)
    return "".join(chunks)


def _finish_reason(resp) -> str | None:
    """Gemini reports `candidates[0].finish_reason` as an ENUM, not a string.

    `.name` is the comparable value ("MAX_TOKENS"); `str()` on the enum yields
    "FinishReason.MAX_TOKENS", which matches nothing.
    """
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    return getattr(reason, "name", None) or str(reason)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        # The SDK read GOOGLE_GEMINI_BASE_URL itself; the jail's key broker sets
        # exactly that, so it survives as the legacy name.
        self.base_url = (
            resolve_base_url(
                "gemini", GEMINI_BASE_URL, legacy_env="GOOGLE_GEMINI_BASE_URL"
            )
            or GEMINI_BASE_URL
        ).rstrip("/")
        self.headers = {"x-goog-api-key": key}

    def list_models(self) -> list[str]:
        # Two-stage filter. First the API's own signal: keep only models whose
        # generation methods include `generateContent` (drops embeddings, which
        # expose `embedContent`). But TTS/image/music models advertise
        # `generateContent` too, so also apply the name-based text-gen filter.
        # Names arrive as `models/gemini-3-flash-preview` — strip the prefix.
        #
        # The wire field is `supportedGenerationMethods`. The SDK called it
        # `supported_actions`, which is a DIFFERENT NAME and not a case variant of
        # it — so `_snake_keys` could never have bridged the gap, and reading the
        # SDK's name here silently returned an empty catalog rather than erroring.
        catalog = get_json(
            f"{self.base_url}/{GEMINI_API_VERSION}/models?pageSize=1000",
            self.headers,
        )
        out: list[str] = []
        for m in catalog.get("models", []):
            if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                continue
            mid = (m.get("name") or "").split("/", 1)[-1]
            if mid and is_text_generation_model(mid):
                out.append(mid)
        return sorted(out)

    def generate(self, request: Request) -> Response:
        reasoning_on = request.reasoning is not None
        # Everything the SDK took as `config=GenerateContentConfig(...)` is the
        # REST `generationConfig` object — except systemInstruction, which is a
        # TOP-LEVEL field and 400s if nested in here.
        generation_config: dict = {
            # The output budget arrives already resolved: the CLI sized it
            # for reasoning (reasoning.min_output_tokens) or honoured an
            # explicit --max-tokens. Send it verbatim.
            "maxOutputTokens": request.max_tokens,
        }
        # Gemini accepts a custom temperature alongside thinking.
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature

        if request.schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseJsonSchema"] = request.schema
        elif request.json_mode:
            generation_config["responseMimeType"] = "application/json"

        if reasoning_on:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": gemini_thinking_budget(request.wire_effort)
            }

        # Attachments first, prompt last, so the trailing text reads as the
        # instruction — the same order the SDK's Part list produced.
        parts: list[dict] = [
            {
                "inline_data": {
                    "mime_type": a.mime_type,
                    "data": base64.b64encode(a.data).decode(),
                }
            }
            for a in request.attachments
        ]
        parts.append({"text": request.prompt})

        body: dict = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}

        model = request.wire_model or request.model
        resp = wrap(
            _snake_keys(
                post_json(
                    f"{self.base_url}/{GEMINI_API_VERSION}/models/{model}:generateContent",
                    self.headers,
                    body,
                )
            )
        )

        text = _candidate_text(resp)

        return Response(
            text=text,
            model=request.model,  # registry key, not the wire id
            provider=self.name,
            stop_reason=_finish_reason(resp),
            raw=resp,
            **from_gemini(getattr(resp, "usage_metadata", None)),
        )
