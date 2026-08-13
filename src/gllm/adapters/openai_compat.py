"""Generic OpenAI-compatible host adapter.

One adapter parameterised by a `ProviderSpec`, for hosts that serve *other
labs'* open models behind the OpenAI Chat Completions wire format: Groq,
Regolo, and whatever comes next. Adding host number three should be a
`PROVIDERS` entry plus `MODELS` rows and **zero new code** — that is the whole
point of the provider axis.

Standalone, not an `OpenAIProvider` subclass: that one routes by name to the
Responses API, which these hosts do not speak (same reasoning as the Z.AI
adapter — see .llm-memory/CONVENTIONS-zai-glm-adapter.md).

Host quirks are config, not branches:
- `ProviderSpec.extra_body` is merged into every request body. Regolo NEEDS
  `disable_fallbacks`: without it the host silently answers with a DIFFERENT
  model than you asked for, which makes model identity, cost accounting and
  every capability gate a lie.
- `ProviderSpec.image_url_format_field` adds Regolo's `{"format": <mime>}`
  inside `image_url` blocks.
- Thinking follows `ModelCaps.thinking_dialect`: `compat_effort` sends a bare
  `reasoning_effort` (Groq), `compat_thinking_flag` pairs it with a top-level
  `thinking` flag (Regolo).

Neither host enforces a JSON Schema and neither takes documents, so `--schema`
and PDFs are refused. `--json` (best-effort `json_object`) still works.
"""

from __future__ import annotations

import base64
import os

from .._http import get_json, post_json, wrap
from ..config import resolve_base_url
from ..domain import Attachment, Request, Response
from ..models import caps_for
from ..ports import LLMProvider
from ..providers import ProviderSpec
from ..usage import from_openai_chat
from ._capabilities import is_text_generation_model


def _finish_reason(resp) -> str | None:
    """OpenAI-compatible `choices[0].finish_reason` — "length" when capped."""
    choices = getattr(resp, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


class OpenAICompatProvider(LLMProvider):
    """An OpenAI-compatible host, described entirely by its ProviderSpec."""

    def __init__(self, spec: ProviderSpec, api_key: str | None = None):
        self.spec = spec
        self.name = spec.tag
        key = api_key or _first_env(spec.api_key_env)
        if not key:
            expected = " or ".join(spec.api_key_env)
            raise RuntimeError(f"{expected} is not set")
        # One edit covers every openai_compat host (groq, regolo, ...), because
        # they all resolve their endpoint through the shared ProviderSpec.
        self.base_url = (
            resolve_base_url(spec.tag, spec.base_url) or spec.base_url
        ).rstrip("/")
        self.headers = {"Authorization": f"Bearer {key}"}

    def list_models(self) -> list[str]:
        catalog = get_json(f"{self.base_url}/models", self.headers)
        return sorted(
            m["id"]
            for m in catalog.get("data", [])
            if is_text_generation_model(m["id"])
        )

    def generate(self, request: Request) -> Response:
        if request.schema is not None:
            raise RuntimeError(
                f"{self.name} has no native JSON-schema enforcement (only "
                f"response_format=json_object); --schema would be faked via a "
                f"prompt instruction with no guarantee. Refusing. Use --json for "
                f"best-effort JSON instead."
            )

        model = request.wire_model or request.model
        caps = caps_for(request.model, self.name)

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({
            "role": "user",
            "content": self._user_content(request, caps.supports_vision),
        })

        reasoning_on = request.reasoning is not None

        kwargs: dict = {
            "model": model,
            "messages": messages,
            # The output budget arrives already resolved: the CLI sized it
            # for reasoning (reasoning.min_output_tokens) or honoured an
            # explicit --max-tokens. Send it verbatim.
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        extra_body = dict(self.spec.extra_body)
        if reasoning_on:
            if caps.thinking_dialect is None:
                raise RuntimeError(
                    f"{self.name} model {request.model!r} has no reasoning "
                    f"control; drop --reasoning."
                )
            kwargs["reasoning_effort"] = request.wire_effort
            if caps.thinking_dialect == "compat_thinking_flag":
                extra_body["thinking"] = True
        # Merged into the body itself, which is all `extra_body` ever meant —
        # it was the SDK's escape hatch for non-OpenAI keys, not a wire concept.
        # Regolo's `disable_fallbacks` must land here or the host may silently
        # answer with a different model than the one asked for.
        kwargs.update(extra_body)

        resp = wrap(
            post_json(f"{self.base_url}/chat/completions", self.headers, kwargs)
        )

        # Chain-of-thought arrives in `reasoning_content` on these hosts. gllm is
        # one-shot and prints the answer only, so it is deliberately discarded.
        text = getattr(resp.choices[0].message, "content", None) or ""

        return Response(
            text=text,
            # Report the registry key, not the wire id: it is what --usage,
            # pricing and the user all identify the model by.
            model=request.model or model,
            provider=self.name,
            stop_reason=_finish_reason(resp),
            raw=resp,
            **from_openai_chat(getattr(resp, "usage", None)),
        )

    def _user_content(self, request: Request, vision: bool):
        """Plain string for text turns; a multimodal `[text, image_url...]` array
        for vision models with image attachments."""
        for a in request.attachments:
            if a.mime_type == "application/pdf":
                raise RuntimeError(
                    f"{self.name} does not accept PDF attachments. Use "
                    f"claude-opus-5 or gemini-3.6-flash for documents."
                )
            if not a.mime_type.startswith("image/"):
                raise RuntimeError(
                    f"{self.name} cannot encode attachment {a.source_label!r} "
                    f"(mime {a.mime_type})."
                )

        if not request.attachments:
            return request.prompt
        if not vision:
            raise RuntimeError(
                f"{self.name} model {request.model!r} is not a vision model; it "
                f"cannot accept images."
            )
        parts: list[dict] = [{"type": "text", "text": request.prompt}]
        parts.extend(self._image_part(a) for a in request.attachments)
        return parts

    def _image_part(self, a: Attachment) -> dict:
        b64 = base64.b64encode(a.data).decode()
        image_url: dict = {"url": f"data:{a.mime_type};base64,{b64}"}
        if self.spec.image_url_format_field:
            image_url["format"] = a.mime_type
        return {"type": "image_url", "image_url": image_url}


def _first_env(names: tuple[str, ...]) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None
