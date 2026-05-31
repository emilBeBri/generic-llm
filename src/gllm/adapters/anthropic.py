"""Anthropic adapter.

Single non-streaming call. Structured output uses the native
`output_config.format = json_schema` path (Claude Opus 4.5/4.6/4.7,
Sonnet 4.5/4.6, Haiku 4.5+).

For `json_mode` (no schema) Anthropic has no formal JSON mode, so we just
inject a short instruction telling the model to reply with valid JSON and
nothing else.
"""

from __future__ import annotations

import base64
import os

import anthropic

from ..domain import Attachment, Request, Response
from ..ports import LLMProvider


def _anthropic_content(prompt: str, attachments: tuple[Attachment, ...]) -> list[dict] | str:
    """Build the `content` value for the user message.

    With no attachments: return the prompt as a plain string (the historical
    shape — keeps the wire format unchanged for the simple case).
    With attachments: return a list of content blocks, attachments first so
    the trailing text reads as the instruction.
    """
    if not attachments:
        return prompt
    blocks: list[dict] = []
    for a in attachments:
        b64 = base64.b64encode(a.data).decode()
        if a.mime_type.startswith("image/"):
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": a.mime_type, "data": b64},
            })
        elif a.mime_type == "application/pdf":
            blocks.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
            })
        else:
            raise RuntimeError(
                f"anthropic adapter cannot encode attachment {a.source_label!r} "
                f"(mime {a.mime_type}); only images and application/pdf are supported."
            )
    blocks.append({"type": "text", "text": prompt})
    return blocks


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=key, max_retries=3)

    def generate(self, request: Request) -> Response:
        content = _anthropic_content(request.prompt, request.attachments)
        kwargs: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if request.system:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": request.schema}
            }
        elif request.json_mode:
            extra = "Respond with valid JSON only. No prose, no code fences."
            kwargs["system"] = (
                f"{request.system}\n\n{extra}" if request.system else extra
            )

        msg = self.client.messages.create(**kwargs)

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

        return Response(
            text=text,
            model=msg.model,
            provider=self.name,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            raw=msg,
        )
