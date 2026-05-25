"""Azure Anthropic (Foundry) adapter.

Claude served through Azure AI Foundry via the `AnthropicFoundry` client.
Model names carry a `-dev` suffix (e.g. `claude-opus-4-7-dev`), which routing
uses to pick this adapter over the direct Anthropic one.

Two things make it diverge from the direct Anthropic adapter:

1. **No `output_config`.** Azure does not support Anthropic's native
   `output_config.format = json_schema`, so `--schema`/`--json` are emulated
   with an instruction injected into the system prompt (same as the direct
   adapter's json_mode fallback).

2. **WORK_ENV forced thinking.** When WORK mode is on (see config.work_env),
   we force maximum extended thinking, scaled by model family. Thinking
   requires a generous max_tokens and an unset temperature, so we bump the
   former and drop the latter. We stream and take the final message because
   long thinking generations can outrun a non-streaming socket timeout; only
   the text blocks are returned (thinking blocks are discarded).
"""

from __future__ import annotations

import json
import os

from ..config import work_env
from ..domain import Request, Response
from ..ports import LLMProvider


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
        kwargs: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }

        # Azure has no native json_schema; emulate via a system instruction.
        system = request.system
        if request.schema is not None:
            schema_txt = json.dumps(request.schema, indent=2)
            extra = (
                "Respond with valid JSON only, matching this JSON Schema. "
                "No prose, no code fences.\n\n" + schema_txt
            )
            system = f"{system}\n\n{extra}" if system else extra
        elif request.json_mode:
            extra = "Respond with valid JSON only. No prose, no code fences."
            system = f"{system}\n\n{extra}" if system else extra
        if system:
            kwargs["system"] = system

        thinking_forced = work_env() and self._force_work_env_thinking(
            kwargs, request.model
        )

        # Extended thinking pins temperature to 1; only set it otherwise.
        if request.temperature is not None and not thinking_forced:
            kwargs["temperature"] = request.temperature

        with self.client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()

        text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        )

        return Response(
            text=text,
            model=msg.model,
            provider=self.name,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            raw=msg,
        )

    @staticmethod
    def _force_work_env_thinking(kwargs: dict, model: str) -> bool:
        """Force maximum extended thinking for WORK mode, scaled by family.

        Azure does not support `output_config`/effort control, so we only set
        the thinking block. 4.6/4.7 use adaptive thinking; `display:summarized`
        is required on 4.7 (its default flipped to 'omitted') or the reasoning
        is suppressed. 4.5 and others use a fixed budget.
        """
        m = model.lower()
        if "4-6" in m or "4-7" in m:
            kwargs["max_tokens"] = 64000
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        elif "4-5" in m:
            kwargs["max_tokens"] = 64000
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 32000}
        else:
            kwargs["max_tokens"] = 32000
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 16000}
        return True
