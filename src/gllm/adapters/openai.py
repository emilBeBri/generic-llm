"""OpenAI adapter.

Two API surfaces:
- Responses API: used for o1/o3/o4 reasoning models and the gpt-5 family.
- Chat Completions: used for gpt-4o, gpt-4.1, etc.

Structured output:
- Responses:        text = {"format": {"type": "json_schema", "name": ..., "schema": ..., "strict": True}}
- Chat Completions: response_format = {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}

Plain JSON mode (no schema):
- Responses:        text = {"format": {"type": "json_object"}}
- Chat Completions: response_format = {"type": "json_object"}
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from .._http import get_json, post_json, wrap
from ..config import resolve_base_url
from ..domain import Attachment, Request, Response
from ..ports import LLMProvider
from ..usage import from_openai_chat, from_openai_responses
from ._capabilities import (
    is_text_generation_model,
    openai_file_extension_for_mime,
    openai_file_mime_for_path,
    supports_attachment,
    use_responses_api,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"


def _attachment_filename(attachment: Attachment) -> str:
    name = Path(attachment.source_label).name
    if (
        name
        and name != "<stdin>"
        and openai_file_mime_for_path(Path(name)) is not None
    ):
        return name
    extension = openai_file_extension_for_mime(attachment.mime_type)
    path = Path(name)
    stem = path.stem if path.suffix else name
    if not stem or name == "<stdin>":
        stem = "file"
    return f"{stem}{extension}"


def _responses_input(
    prompt: str,
    attachments: tuple[Attachment, ...],
    *,
    provider: str = "openai",
    model: str = "",
):
    """Build the `input` arg for client.responses.create.

    No attachments -> the bare prompt string (historical shape, unchanged
    wire format for the simple case).
    With attachments -> a single user message with structured content parts.
    """
    if not attachments:
        return prompt
    parts: list[dict] = []
    for a in attachments:
        if not supports_attachment(provider, model, a):
            raise RuntimeError(
                f"{provider} responses adapter cannot encode attachment "
                f"{a.source_label!r} (mime {a.mime_type})."
            )
        b64 = base64.b64encode(a.data).decode()
        if a.mime_type.startswith("image/"):
            parts.append({
                "type": "input_image",
                "image_url": f"data:{a.mime_type};base64,{b64}",
            })
        else:
            parts.append({
                "type": "input_file",
                "filename": _attachment_filename(a),
                "file_data": f"data:{a.mime_type};base64,{b64}",
            })
    parts.append({"type": "input_text", "text": prompt})
    return [{"role": "user", "content": parts}]


def _chat_user_content(
    prompt: str,
    attachments: tuple[Attachment, ...],
    *,
    provider: str = "openai",
    model: str = "",
):
    """User-message content for chat.completions.create."""
    if not attachments:
        return prompt
    parts: list[dict] = [{"type": "text", "text": prompt}]
    for a in attachments:
        if not supports_attachment(provider, model, a):
            raise RuntimeError(
                f"{provider} chat-completions cannot encode attachment "
                f"{a.source_label!r} (mime {a.mime_type})."
            )
        b64 = base64.b64encode(a.data).decode()
        if a.mime_type.startswith("image/"):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{a.mime_type};base64,{b64}"},
            })
        else:
            parts.append({
                "type": "file",
                "file": {
                    "filename": _attachment_filename(a),
                    "file_data": f"data:{a.mime_type};base64,{b64}",
                },
            })
    return parts


def _output_text(resp) -> str:
    """Reconstruct the Responses API's flattened text from the raw `output` list.

    `resp.output_text` was an SDK CONVENIENCE, not a wire field — dropping the
    SDK means rebuilding it. The wire shape is a list of typed items
    (`reasoning`, `message`, `web_search_call`, …); only `message` items carry
    content, and within that only `output_text` parts are answer text.

    A `refusal` part with no accompanying text raises rather than returning "":
    the SDK's `output_text` would have been empty and gllm would have printed a
    blank line as though the call succeeded.
    """
    chunks: list[str] = []
    refusals: list[str] = []
    for item in getattr(resp, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            kind = getattr(part, "type", None)
            if kind == "output_text":
                chunks.append(getattr(part, "text", "") or "")
            elif kind == "refusal":
                refusals.append(getattr(part, "refusal", "") or "")
    text = "".join(chunks)
    if not text and refusals:
        raise RuntimeError(
            "the model refused this request: " + " ".join(r for r in refusals if r)
        )
    return text


def _incomplete_reason(resp) -> str | None:
    """Responses API: a capped generation comes back `status="incomplete"` with
    `incomplete_details.reason = "max_output_tokens"` — there is no
    `finish_reason` on this surface."""
    details = getattr(resp, "incomplete_details", None)
    return getattr(details, "reason", None) if details else None


def _finish_reason(resp) -> str | None:
    """Chat Completions surface: `choices[0].finish_reason` — "length" when
    capped."""
    choices = getattr(resp, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        name: str | None = None,
    ):
        # `base_url`/`name` are the override surface for OpenAI-compatible
        # backends (xAI Grok, Azure Foundry) that subclass this provider.
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        # A subclass-supplied base_url wins (Grok resolves its own; Azure derives
        # one from AZURE_FOUNDRY_ENDPOINT). Only the direct provider consults the
        # environment — and it MUST, now that the SDK is gone: the SDK used to
        # read OPENAI_BASE_URL itself, which is how the jail's key broker
        # redirects us. `legacy_env` preserves that exact variable.
        resolved = base_url or resolve_base_url(
            "openai", OPENAI_BASE_URL, legacy_env="OPENAI_BASE_URL"
        )
        self.base_url = (resolved or OPENAI_BASE_URL).rstrip("/")
        self.headers = {"Authorization": f"Bearer {key}"}
        if name:
            self.name = name

    def list_models(self) -> list[str]:
        # The catalog endpoint returns no capability metadata, so filter
        # non-text-generation families (embeddings/audio/image/moderation) by
        # name. Inherited by Grok and Azure unchanged.
        catalog = get_json(f"{self.base_url}/models", self.headers)
        return sorted(
            m["id"]
            for m in catalog.get("data", [])
            if is_text_generation_model(m["id"])
        )

    def generate(self, request: Request) -> Response:
        if use_responses_api(request.model):
            return self._generate_responses(request)
        return self._generate_chat(request)

    def _generate_responses(self, request: Request) -> Response:
        reasoning_on = request.reasoning is not None
        kwargs: dict = {
            "model": request.wire_model or request.model,
            "input": _responses_input(
                request.prompt,
                request.attachments,
                provider=self.name,
                model=request.model,
            ),
            # The output budget arrives already resolved: the CLI sized it
            # for reasoning (reasoning.min_output_tokens) or honoured an
            # explicit --max-tokens. Send it verbatim.
            "max_output_tokens": request.max_tokens,
            "store": False,
        }
        if request.system:
            kwargs["instructions"] = request.system
        # Reasoning models (o-series, gpt-5) reject a custom temperature.
        if request.temperature is not None and not reasoning_on:
            kwargs["temperature"] = request.temperature
        if reasoning_on:
            kwargs["reasoning"] = {"effort": request.wire_effort}

        if request.schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "Output",
                    "schema": request.schema,
                    "strict": True,
                }
            }
        elif request.json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}

        resp = wrap(post_json(f"{self.base_url}/responses", self.headers, kwargs))

        text = _output_text(resp)

        return Response(
            text=text,
            model=request.model,  # registry key, not the wire id
            provider=self.name,
            stop_reason=_incomplete_reason(resp),
            raw=resp,
            **from_openai_responses(getattr(resp, "usage", None)),
        )

    def _generate_chat(self, request: Request) -> Response:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({
            "role": "user",
            "content": _chat_user_content(
                request.prompt,
                request.attachments,
                provider=self.name,
                model=request.model,
            ),
        })

        kwargs: dict = {
            "model": request.wire_model or request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "Output",
                    "schema": request.schema,
                    "strict": True,
                },
            }
        elif request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = wrap(
            post_json(f"{self.base_url}/chat/completions", self.headers, kwargs)
        )

        text = getattr(resp.choices[0].message, "content", None) or ""

        return Response(
            text=text,
            model=resp.model,
            provider=self.name,
            stop_reason=_finish_reason(resp),
            raw=resp,
            **from_openai_chat(getattr(resp, "usage", None)),
        )
