"""DeepSeek adapter.

DeepSeek's API is OpenAI-compatible (chat completions) at api.deepseek.com,
so we POST the chat-completions body straight there via `gllm._http`. Models:
`deepseek-flash` (V4.1-Flash, the current flagship — note the versionless id)
and `deepseek-v4-pro`. `deepseek-v4-flash` still resolves but is a retired
alias served by V4.1-Flash.

Thinking: the v4 models reason by default and emit a `reasoning_content`
field alongside `content`. gllm is one-shot and prints only the final text,
so the reasoning trace is discarded.

`--reasoning` IS honoured here. gllm long claimed "DeepSeek has no control
surface" and refused `-r` outright — wrong since V4, and verified wrong live on
2026-07-29 (effort=high gave ~72 chars of reasoning_content, max ~200, and
thinking:disabled 0). V4 exposes a toggle
(`extra_body={"thinking": {"type": "enabled"}}`, default enabled) and an effort
control publishing `low|high|max` since V4.1 (2026-09-10; it was `high|max`
before, so `-r low` used to resolve up to `high` and now genuinely gets `low`).
The CLI resolves gllm's rung onto that vocabulary before it reaches us, so
`-r xhigh` arrives as `max`.

Images: V4.1-Flash reads them natively as OpenAI-shaped `image_url` parts with
a base64 `data:` URL — JPEG, PNG, GIF, WebP, sniffed from content rather than
filename or declared MIME. This is gated on the model's registry caps, not on
the provider: `deepseek-v4-pro` has no image input, and sending it one is a
refusal here rather than a 400 from the API. PDFs are refused on every DeepSeek
model — there is no native document input.

Structured output: DeepSeek has no native json_schema/strict mode — only
`response_format={"type": "json_object"}`. `--json` flips that on (best-effort
JSON). `--schema` (which promises *enforced* structure) is REFUSED — we will not
fake strict enforcement with a prompt instruction. The CLI gates this earlier
(supports_strict_schema); the raise here is the library-use backstop.
"""

from __future__ import annotations

import base64
import os

from .._http import get_json, post_json, wrap
from ..config import resolve_base_url
from ..domain import Attachment, Request, Response
from ..ports import LLMProvider
from ..usage import from_deepseek
from ._capabilities import is_text_generation_model, supports_image

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _image_part(a: Attachment) -> dict:
    b64 = base64.b64encode(a.data).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{a.mime_type};base64,{b64}"},
    }


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
        messages.append({"role": "user", "content": self._user_content(request)})

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
            # vocabulary (low|high|max) by the CLI — see reasoning.resolve_effort.
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

    def _user_content(self, request: Request):
        """Plain string for a text turn; an OpenAI-shaped `[text, image_url...]`
        array once images are attached. The vision check is per MODEL, not per
        provider — V4.1-Flash reads images and V4-Pro does not."""
        for a in request.attachments:
            if a.mime_type == "application/pdf":
                raise RuntimeError(
                    "deepseek has no native PDF input. Use claude-opus-4-8 or "
                    "gemini-3.1-pro-preview for documents."
                )
            if not a.mime_type.startswith("image/"):
                raise RuntimeError(
                    f"deepseek cannot encode attachment {a.source_label!r} "
                    f"(mime {a.mime_type})."
                )

        if not request.attachments:
            return request.prompt
        if not supports_image(self.name, request.model):
            raise RuntimeError(
                f"deepseek model {request.model!r} does not accept images. Use "
                f"deepseek-flash, which reads them natively."
            )
        parts: list[dict] = [{"type": "text", "text": request.prompt}]
        parts.extend(_image_part(a) for a in request.attachments)
        return parts
