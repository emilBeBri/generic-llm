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

from openai import OpenAI

from ..domain import Attachment, Request, Response
from ..ports import LLMProvider
from ._capabilities import use_responses_api


def _responses_input(prompt: str, attachments: tuple[Attachment, ...]):
    """Build the `input` arg for client.responses.create.

    No attachments -> the bare prompt string (historical shape, unchanged
    wire format for the simple case).
    With attachments -> a single user message with structured content parts.
    """
    if not attachments:
        return prompt
    parts: list[dict] = []
    for a in attachments:
        b64 = base64.b64encode(a.data).decode()
        if a.mime_type.startswith("image/"):
            parts.append({
                "type": "input_image",
                "image_url": f"data:{a.mime_type};base64,{b64}",
            })
        elif a.mime_type == "application/pdf":
            filename = Path(a.source_label).name or "file.pdf"
            parts.append({
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:application/pdf;base64,{b64}",
            })
        else:
            raise RuntimeError(
                f"openai responses adapter cannot encode attachment "
                f"{a.source_label!r} (mime {a.mime_type})."
            )
    parts.append({"type": "input_text", "text": prompt})
    return [{"role": "user", "content": parts}]


def _chat_user_content(prompt: str, attachments: tuple[Attachment, ...]):
    """User-message content for chat.completions.create. Images only — PDFs
    have no content-block type on this API and should be rejected upstream by
    the capability check."""
    if not attachments:
        return prompt
    parts: list[dict] = [{"type": "text", "text": prompt}]
    for a in attachments:
        if a.mime_type == "application/pdf":
            raise RuntimeError(
                "openai chat-completions cannot accept PDF inputs; use a "
                "Responses-API model (gpt-5, o1/o3/o4, codex)."
            )
        if not a.mime_type.startswith("image/"):
            raise RuntimeError(
                f"openai chat-completions cannot encode attachment "
                f"{a.source_label!r} (mime {a.mime_type})."
            )
        b64 = base64.b64encode(a.data).decode()
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{a.mime_type};base64,{b64}"},
        })
    return parts


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
        client_kwargs: dict = {"api_key": key, "max_retries": 3}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        if name:
            self.name = name

    def generate(self, request: Request) -> Response:
        if use_responses_api(request.model):
            return self._generate_responses(request)
        return self._generate_chat(request)

    def _generate_responses(self, request: Request) -> Response:
        kwargs: dict = {
            "model": request.model,
            "input": _responses_input(request.prompt, request.attachments),
            "max_output_tokens": request.max_tokens,
            "store": False,
        }
        if request.system:
            kwargs["instructions"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

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

        resp = self.client.responses.create(**kwargs)

        # SDK exposes a flattened `output_text` aggregating all text deltas.
        text = getattr(resp, "output_text", "") or ""

        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0

        return Response(
            text=text,
            model=request.model,
            provider=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw=resp,
        )

    def _generate_chat(self, request: Request) -> Response:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({
            "role": "user",
            "content": _chat_user_content(request.prompt, request.attachments),
        })

        kwargs: dict = {
            "model": request.model,
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

        resp = self.client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content or ""

        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

        return Response(
            text=text,
            model=resp.model,
            provider=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw=resp,
        )
