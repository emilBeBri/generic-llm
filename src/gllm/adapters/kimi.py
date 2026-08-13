"""Moonshot Kimi adapter.

Kimi speaks OpenAI-compatible Chat Completions at api.moonshot.ai (POSTed via
`gllm._http`), but its reasoning controls differ by model family:

- kimi-k3: always reasons; optional reasoning_effort in low|high|max.
- kimi-k2.7-code*: always reasons and rejects reasoning controls.
- kimi-k2.6: binary thinking block; no reasoning_effort.

Every current model accepts images. PDFs and strict JSON Schema are unsupported.
gllm is one-shot, so reasoning_content is deliberately not round-tripped.
"""

from __future__ import annotations

import base64
import os

from .._http import get_json, post_json, wrap
from ..config import resolve_base_url
from ..domain import Attachment, Request, Response
from ..ports import LLMProvider
from ..usage import from_kimi
from ._capabilities import is_text_generation_model

KIMI_BASE_URL = "https://api.moonshot.ai/v1"


def _is_k3(model: str) -> bool:
    return (model or "").lower().startswith("kimi-k3")


def _is_k27_code(model: str) -> bool:
    return (model or "").lower().startswith("kimi-k2.7-code")


def _image_part(attachment: Attachment) -> dict:
    encoded = base64.b64encode(attachment.data).decode()
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{attachment.mime_type};base64,{encoded}",
        },
    }


def _finish_reason(resp) -> str | None:
    """OpenAI-compatible `choices[0].finish_reason` — "length" when capped."""
    choices = getattr(resp, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


class KimiProvider(LLMProvider):
    name = "kimi"

    def __init__(self, api_key: str | None = None):
        key = (
            api_key
            or os.environ.get("MOONSHOT_API_KEY")
            or os.environ.get("KIMI_API_KEY")
        )
        if not key:
            raise RuntimeError("MOONSHOT_API_KEY or KIMI_API_KEY is not set")
        self.base_url = (
            resolve_base_url("kimi", KIMI_BASE_URL) or KIMI_BASE_URL
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
                "Kimi has no native JSON-schema enforcement; --schema would be "
                "faked. Use --json for best-effort JSON instead."
            )
        if request.temperature is not None:
            raise RuntimeError(
                "Kimi fixes temperature server-side; drop --temperature."
            )

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": self._user_content(request)})

        model = request.wire_model or request.model
        kwargs: dict = {
            "model": model,
            "messages": messages,
            # Resolved by the CLI. Note this floor used to be UNCONDITIONAL here
            # (16000 even without -r), so a plain Kimi call now defaults to
            # DEFAULT_MAX_OUTPUT like every other provider. Kimi publishes no
            # fixed output ceiling — k3's limit is `context - prompt_tokens` —
            # so there is no honest `max_output` to put in the registry yet.
            "max_completion_tokens": request.max_tokens,
        }
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self._reasoning_kwargs(request))

        response = wrap(
            post_json(f"{self.base_url}/chat/completions", self.headers, kwargs)
        )
        text = getattr(response.choices[0].message, "content", None) or ""
        return Response(
            text=text,
            model=request.model or response.model,
            provider=self.name,
            stop_reason=_finish_reason(response),
            raw=response,
            **from_kimi(getattr(response, "usage", None)),
        )

    @staticmethod
    def _reasoning_kwargs(request: Request) -> dict:
        if request.reasoning is None:
            return {}
        if _is_k27_code(request.model):
            raise RuntimeError(
                f"Kimi model {request.model!r} always reasons but exposes no "
                "reasoning control; drop --reasoning."
            )
        if _is_k3(request.model):
            return {"reasoning_effort": request.wire_effort}
        # Top-level on the wire; `extra_body` only ever existed to get this
        # past the SDK's kwarg validation.
        return {"thinking": {"type": "enabled"}}

    @staticmethod
    def _user_content(request: Request):
        for attachment in request.attachments:
            if attachment.mime_type == "application/pdf":
                raise RuntimeError(
                    "Kimi does not accept inline PDF attachments. Use an "
                    "Anthropic, OpenAI Responses, or Gemini model."
                )
            if not attachment.mime_type.startswith("image/"):
                raise RuntimeError(
                    f"Kimi cannot encode attachment {attachment.source_label!r} "
                    f"(mime {attachment.mime_type})."
                )

        if not request.attachments:
            return request.prompt
        parts: list[dict] = [{"type": "text", "text": request.prompt}]
        parts.extend(_image_part(attachment) for attachment in request.attachments)
        return parts
