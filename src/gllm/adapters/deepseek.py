"""DeepSeek adapter.

DeepSeek's API is OpenAI-compatible (chat completions) at api.deepseek.com,
so we POST the chat-completions body straight there via `gllm._http`. Models:
`deepseek-v4-pro`, `deepseek-v4-flash`.

Thinking: the v4 models reason by default and emit a `reasoning_content`
field alongside `content`. gllm is one-shot and prints only the final text,
so the reasoning trace is discarded.

`--reasoning` IS honoured here. gllm long claimed "DeepSeek has no control
surface" and refused `-r` outright — wrong since V4, and verified wrong live on
2026-07-29 (effort=high gave ~72 chars of reasoning_content, max ~200, and
thinking:disabled 0). V4 exposes a toggle
(`extra_body={"thinking": {"type": "enabled"}}`, default enabled) and an effort
control publishing `high|max`; the CLI resolves gllm's rung onto those two
before it reaches us, so `-r xhigh` arrives as `max`.

Structured output: DeepSeek has no native json_schema/strict mode — only
`response_format={"type": "json_object"}`. `--json` flips that on (best-effort
JSON). `--schema` (which promises *enforced* structure) is REFUSED — we will not
fake strict enforcement with a prompt instruction. The CLI gates this earlier
(supports_strict_schema); the raise here is the library-use backstop.
"""

from __future__ import annotations

import os

from .._http import get_json, post_json, wrap
from ..config import resolve_base_url
from ..domain import Request, Response
from ..ports import LLMProvider
from ..usage import from_deepseek
from ._capabilities import is_text_generation_model

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _finish_reason(resp) -> str | None:
    """OpenAI-compatible `choices[0].finish_reason` — "length" when capped."""
    choices = getattr(resp, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.base_url = (
            resolve_base_url("deepseek", DEEPSEEK_BASE_URL) or DEEPSEEK_BASE_URL
        ).rstrip("/")
        self.headers = {"Authorization": f"Bearer {key}"}

    def list_models(self) -> list[str]:
        # OpenAI-compatible catalog endpoint; apply the same text-generation
        # filter for consistency (DeepSeek's catalog is all chat today).
        catalog = get_json(f"{self.base_url}/models", self.headers)
        return sorted(
            m["id"]
            for m in catalog.get("data", [])
            if is_text_generation_model(m["id"])
        )

    def generate(self, request: Request) -> Response:
        if request.attachments:
            raise RuntimeError(
                "deepseek does not accept file attachments (no native image "
                "or document API). Try a vision-capable model like "
                "claude-opus-4-8, gpt-5, or gemini-3.1-pro-preview."
            )
        if request.schema is not None:
            raise RuntimeError(
                "deepseek has no native JSON-schema enforcement (only "
                "response_format=json_object); --schema would be faked via a "
                "prompt instruction with no guarantee. Refusing. Use --json for "
                "best-effort JSON instead."
            )

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        reasoning_on = request.reasoning is not None

        kwargs: dict = {
            "model": request.wire_model or request.model,
            "messages": messages,
            # The output budget arrives already resolved: the CLI sized it
            # for reasoning (reasoning.min_output_tokens) or honoured an
            # explicit --max-tokens. Send it verbatim.
            "max_tokens": request.max_tokens,
        }
        # Thinking mode silently IGNORES temperature/top_p/penalties rather than
        # erroring, so only send one when thinking is off — otherwise the value
        # looks honoured and isn't.
        if request.temperature is not None and not reasoning_on:
            kwargs["temperature"] = request.temperature
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if reasoning_on:
            # Both go top-level in the request body. Under the SDK `thinking`
            # needed `extra_body` to survive the client's kwarg validation, but
            # extra_body was never a wire concept — it merged into this same
            # object. The effort value is already resolved to DeepSeek's own
            # vocabulary (high|max) by the CLI — see reasoning.resolve_effort.
            kwargs["thinking"] = {"type": "enabled"}
            kwargs["reasoning_effort"] = request.wire_effort

        resp = wrap(post_json(f"{self.base_url}/chat/completions", self.headers, kwargs))

        # `content` is null on a reasoning-only response; DeepSeek's
        # reasoning_content is deliberately discarded (see module docstring).
        text = getattr(resp.choices[0].message, "content", None) or ""

        return Response(
            text=text,
            model=resp.model,
            provider=self.name,
            stop_reason=_finish_reason(resp),
            raw=resp,
            **from_deepseek(getattr(resp, "usage", None)),
        )
