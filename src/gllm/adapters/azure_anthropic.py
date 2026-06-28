"""Azure Anthropic (Foundry) adapter.

Claude served through Azure AI Foundry via the `AnthropicFoundry` client.
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

We always stream and take the final message — long thinking generations can
outrun a non-streaming socket timeout; only text blocks are returned.
"""

from __future__ import annotations

import os

from ..domain import Request, Response
from ..ports import LLMProvider
from ..reasoning import anthropic_thinking
from ..usage import from_anthropic
from .anthropic import _anthropic_content


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
        from anthropic import AnthropicFoundry

        key = api_key or os.environ.get("AZURE_ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("AZURE_ANTHROPIC_API_KEY is not set")
        endpoint = endpoint or os.environ.get("AZURE_FOUNDRY_ENDPOINT")
        if not endpoint:
            raise RuntimeError("AZURE_FOUNDRY_ENDPOINT is not set")

        self.client = AnthropicFoundry(
            api_key=key, base_url=_normalize_foundry_url(endpoint)
        )

    def generate(self, request: Request) -> Response:
        content = _anthropic_content(request.prompt, request.attachments)
        reasoning_on = request.reasoning is not None
        kwargs: dict = {
            "model": request.model,
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
            r = anthropic_thinking(request.reasoning, request.model)
            kwargs["thinking"] = r["thinking"]
            kwargs["max_tokens"] = max(kwargs["max_tokens"], r["min_max_tokens"])
            effort = r.get("effort")

        # output_config carries both structured-output `format` and reasoning
        # `effort`, passed via extra_body (the SDK has no top-level param). Same
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
            kwargs["extra_body"] = {"output_config": output_config}

        with self.client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()

        text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        )

        return Response(
            text=text,
            model=msg.model,
            provider=self.name,
            raw=msg,
            **from_anthropic(msg.usage),
        )
