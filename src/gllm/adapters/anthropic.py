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

from .._http import get_json, post_json, post_sse, wrap
from ..config import resolve_base_url
from ..domain import Attachment, Request, Response
from ..ports import LLMProvider
from ..reasoning import anthropic_thinking
from ..usage import from_anthropic
from ._capabilities import thinking_dialect

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# Required on every Messages request. Matches what the SDK sent.
ANTHROPIC_VERSION = "2023-06-01"


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


def _stop_reason(msg) -> str | None:
    """Anthropic `stop_reason` — "max_tokens" when the output cap is hit."""
    return getattr(msg, "stop_reason", None)


def final_message_from_events(events) -> dict:
    """Rebuild the complete Message from an Anthropic SSE event stream.

    This replaces the SDK's `stream.get_final_message()`. The events carry the
    message in pieces and NOTHING repeats them, so each one has a job:

    - `message_start` — the message skeleton, and the only place `input_tokens`
      and the cache counters appear.
    - `content_block_start` / `content_block_delta` — the blocks. Text arrives as
      `text_delta` fragments that must be concatenated in order.
    - `message_delta` — the only place `stop_reason` and the final
      `output_tokens` appear. Miss this event and usage silently reads zero
      output, which is the failure mode to watch for.
    - `error` — can arrive mid-stream after a 200 (e.g. `overloaded_error`), so a
      successful status is not the end of the error handling.

    `thinking_delta` and `signature_delta` are dropped on purpose: gllm is
    one-shot and prints the answer only.
    """
    message: dict = {}
    blocks: dict[int, dict] = {}
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            message = dict(event.get("message") or {})
        elif kind == "content_block_start":
            blocks[event["index"]] = dict(event.get("content_block") or {})
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                block = blocks.setdefault(event["index"], {"type": "text", "text": ""})
                block["text"] = (block.get("text") or "") + (delta.get("text") or "")
        elif kind == "message_delta":
            message.update(
                {k: v for k, v in (event.get("delta") or {}).items() if v is not None}
            )
            usage = message.setdefault("usage", {})
            if isinstance(usage, dict):
                usage.update(event.get("usage") or {})
        elif kind == "error":
            err = event.get("error") or {}
            raise RuntimeError(
                f"anthropic streamed an error mid-response: "
                f"{err.get('type', 'error')}: {err.get('message', event)}"
            )
    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        # The SDK used to read ANTHROPIC_BASE_URL itself; now that it is gone we
        # resolve it, because that is how the jail's key broker redirects us.
        self.base_url = (
            resolve_base_url(
                "anthropic", ANTHROPIC_BASE_URL, legacy_env="ANTHROPIC_BASE_URL"
            )
            or ANTHROPIC_BASE_URL
        ).rstrip("/")
        self.headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def list_models(self) -> list[str]:
        # Every model on the Anthropic API is a text-generation model (no
        # embeddings/audio surface), so no capability filter is needed. The
        # catalog paginates; `limit=1000` takes it in one page rather than
        # threading `after_id` for a list that is currently ~20 rows.
        catalog = get_json(
            f"{self.base_url}/v1/models?limit=1000", self.headers
        )
        return sorted(m["id"] for m in catalog.get("data", []))

    def generate(self, request: Request) -> Response:
        content = _anthropic_content(request.prompt, request.attachments)
        reasoning_on = request.reasoning is not None
        kwargs: dict = {
            "model": request.wire_model or request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if request.system:
            kwargs["system"] = request.system
        # Extended thinking pins temperature to 1; only send it otherwise.
        if request.temperature is not None and not reasoning_on:
            kwargs["temperature"] = request.temperature

        effort: str | None = None
        if reasoning_on:
            # Dialect from the model's registry row: 4.6+ and the 5 family
            # REQUIRE thinking.type=adaptive and reject enabled+budget_tokens.
            r = anthropic_thinking(
                request.wire_effort,
                request.model,
                thinking_dialect(self.name, request.model),
            )
            kwargs["thinking"] = r["thinking"]
            # No max_tokens adjustment here: the thinking budget must be
            # strictly below max_tokens, and the CLI already enforced that from
            # the same `anthropic_thinking` numbers (reasoning.min_output_tokens
            # / hard_min_output_tokens) — refusing an explicit value too low to
            # be legal rather than quietly rewriting it.
            effort = r.get("effort")

        # `output_config` carries BOTH structured-output `format` and reasoning
        # `effort`. It used to need `extra_body` because the SDK had no top-level
        # param for it (passing it directly raised TypeError) — a client-side
        # constraint only. On the wire it is a plain top-level field, which is
        # what extra_body always merged it into. Build it once.
        output_config: dict = {}
        if request.schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.schema}
        elif request.json_mode:
            extra = "Respond with valid JSON only. No prose, no code fences."
            kwargs["system"] = (
                f"{request.system}\n\n{extra}" if request.system else extra
            )
        if effort is not None:
            output_config["effort"] = effort
        if output_config:
            kwargs["output_config"] = output_config

        # Anthropic documents a 10-minute ceiling on non-streaming Messages
        # requests (504 timeout_error, whose remedy text says to stream), and a
        # thinking generation against gllm's 128k default budget can reach it.
        # So the reasoning path streams and reassembles; the plain path does not
        # need to and stays a single POST.
        url = f"{self.base_url}/v1/messages"
        if reasoning_on:
            msg = wrap(
                final_message_from_events(
                    post_sse(url, self.headers, {**kwargs, "stream": True})
                )
            )
        else:
            msg = wrap(post_json(url, self.headers, kwargs))

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

        return Response(
            text=text,
            model=msg.model,
            provider=self.name,
            stop_reason=_stop_reason(msg),
            raw=msg,
            **from_anthropic(msg.usage),
        )
