"""Azure Anthropic (Foundry) adapter.

Claude served through Azure AI Foundry's Anthropic MaaS endpoint, POSTed via
`gllm._http` (it was the SDK's `AnthropicFoundry` client until the transport
rewrite; the wire format is the same Messages API).
Model names carry a `-dev` suffix (e.g. `claude-opus-4-7-dev`), which routing
uses to pick this adapter over the direct Anthropic one.

Foundry exposes `output_config` (verified against Microsoft's docs, 2026-06-12:
learn.microsoft.com/.../foundry-models/concepts/claude-models lists Effort as a
capability, and every example on .../how-to/use-foundry-models-claude passes
`output_config={"effort": ...}`). So this adapter handles output_config the same
way as the direct Anthropic one:

- `output_config.effort` for `--reasoning` (documented-supported on Foundry).
- `output_config.format` json_schema for `--schema` — **attempted natively, but
  NOT yet documented/verified on Foundry.** If Foundry rejects it the API 400s
  loudly (we never fake enforcement). See AZURE-FOUNDRY-SMOKE-TEST.md for the
  verification a work-box agent should run.
- `--json` (no schema) → instruction, same as direct: the Anthropic API has no
  schemaless json-object mode.

We always stream and reassemble the final message: Anthropic documents a
10-minute ceiling on non-streaming Messages requests, and gllm's default output
budget for Claude is large enough to reach it. Only text blocks are returned.
"""

from __future__ import annotations

import os

from .._http import post_sse, wrap
from ..domain import Request, Response
from ..ports import LLMProvider
from ..reasoning import anthropic_thinking
from ..usage import from_anthropic
from ._capabilities import thinking_dialect
from .anthropic import (
    ANTHROPIC_VERSION,
    _anthropic_content,
    _stop_reason,
    final_message_from_events,
    raise_if_refused,
)


def _normalize_foundry_url(endpoint: str) -> str:
    """Resolve a Foundry endpoint to the Anthropic MaaS base_url.

    Agents endpoints (`*.services.ai.azure.com`, `*.cognitiveservices.azure.com`)
    are rewritten to the resource's `*.openai.azure.com` MaaS host, then the
    `/anthropic` suffix is appended if missing.
    """
    final = endpoint
    if "services.ai.azure.com" in final or "cognitiveservices.azure.com" in final:
        try:
            host = final.split("://", 1)[1].split("/")[0]
            resource = host.split(".")[0]
            final = f"https://{resource}.openai.azure.com"
        except (IndexError, ValueError):
            pass
    if not final.endswith("/anthropic"):
        final = final.rstrip("/") + "/anthropic"
    return final


class AzureAnthropicProvider(LLMProvider):
    name = "azure_anthropic"

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        key = api_key or os.environ.get("AZURE_ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("AZURE_ANTHROPIC_API_KEY is not set")
        endpoint = endpoint or os.environ.get("AZURE_FOUNDRY_ENDPOINT")
        if not endpoint:
            raise RuntimeError("AZURE_FOUNDRY_ENDPOINT is not set")

        self.base_url = _normalize_foundry_url(endpoint).rstrip("/")
        self.headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

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
        # Extended thinking pins temperature to 1; only set it otherwise.
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
            # max_tokens already sized by the CLI, same as the direct adapter.
            effort = r.get("effort")

        # output_config carries both structured-output `format` and reasoning
        # `effort`, both plain top-level fields (they needed `extra_body` only to
        # get past the SDK's kwarg validation). Same
        # as the direct adapter — Foundry documents `effort`; `format` is an
        # unverified native attempt (see AZURE-FOUNDRY-SMOKE-TEST.md).
        output_config: dict = {}
        if request.schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.schema}
        elif request.json_mode:
            # The Anthropic API has no schemaless json mode; instruct instead.
            extra = "Respond with valid JSON only. No prose, no code fences."
            kwargs["system"] = (
                f"{request.system}\n\n{extra}" if request.system else extra
            )
        if effort is not None:
            output_config["effort"] = effort
        if output_config:
            kwargs["output_config"] = output_config

        msg = wrap(
            final_message_from_events(
                post_sse(
                    f"{self.base_url}/v1/messages",
                    self.headers,
                    {**kwargs, "stream": True},
                )
            )
        )

        raise_if_refused(msg)
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        )

        return Response(
            text=text,
            model=msg.model,
            provider=self.name,
            stop_reason=_stop_reason(msg),
            raw=msg,
            **from_anthropic(msg.usage),
        )
