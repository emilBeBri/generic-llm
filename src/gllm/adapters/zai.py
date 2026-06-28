"""Z.AI / GLM adapter.

Zhipu's GLM family speaks the OpenAI Chat Completions wire format at
`api.z.ai`, so we drive it with the `openai` SDK pointed at a different
base_url — the same standalone shape as the DeepSeek adapter (NOT an
OpenAIProvider subclass: that one routes `glm-*` to the Responses API, which
Z.AI does not speak). GLM-specific handling on top:

- Deep thinking via `extra_body={"thinking": {"type": "enabled"}}`. The
  `reasoning_effort` knob is honoured ONLY by glm-5.2+ (gated). Omitting
  `--reasoning` leaves thinking at the provider default (on for the 4.5+ line).
- Vision lives in SEPARATE models (glm-5v*, glm-4.6v*, glm-4.5v, glm-ocr) that
  take multimodal `image_url` content; the text GLMs reject image input.
- Structured output is `json_object` only (no native json_schema). `--schema`
  is refused upstream (supports_strict_schema); the raise here is the backstop.
- No native PDF (Z.AI `file_url` needs a hosted URL, not base64), so PDFs are
  rejected loudly.
"""

from __future__ import annotations

import base64
import os

from openai import OpenAI

from ..domain import Attachment, Request, Response
from ..ports import LLMProvider
from ..reasoning import zai_effort
from ..usage import from_openai_chat
from ._capabilities import (
    glm_supports_reasoning_effort,
    glm_supports_thinking,
    is_glm_vision_model,
    is_text_generation_model,
)

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"


def _image_part(a: Attachment) -> dict:
    b64 = base64.b64encode(a.data).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{a.mime_type};base64,{b64}"},
    }


class ZaiProvider(LLMProvider):
    name = "zai"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ZAI_API_KEY")
        if not key:
            raise RuntimeError("ZAI_API_KEY is not set")
        self.client = OpenAI(api_key=key, base_url=ZAI_BASE_URL, max_retries=3)

    def list_models(self) -> list[str]:
        return sorted(
            m.id
            for m in self.client.models.list()
            if is_text_generation_model(m.id)
        )

    def generate(self, request: Request) -> Response:
        if request.schema is not None:
            raise RuntimeError(
                "GLM has no native JSON-schema enforcement (only "
                "response_format=json_object); --schema would be faked via a "
                "prompt instruction with no guarantee. Refusing. Use --json for "
                "best-effort JSON instead."
            )

        vision = is_glm_vision_model(request.model)
        content = self._user_content(request, vision)

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": content})

        reasoning_on = request.reasoning is not None
        # Thinking tokens count against the output budget; raise the floor so the
        # visible answer isn't starved (mirrors the gemini/openai adapters).
        max_out = max(request.max_tokens, 16000) if reasoning_on else request.max_tokens

        kwargs: dict = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_out,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if reasoning_on:
            # The CLI gates supports_reasoning, but stay defensive for library use.
            if not glm_supports_thinking(request.model):
                raise RuntimeError(
                    f"GLM model {request.model!r} does not support thinking; "
                    f"drop --reasoning."
                )
            # `thinking` is a non-OpenAI param -> extra_body; `reasoning_effort`
            # is a recognised SDK kwarg so it goes top-level.
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            if glm_supports_reasoning_effort(request.model):
                kwargs["reasoning_effort"] = zai_effort(request.reasoning)

        resp = self.client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content or ""

        return Response(
            text=text,
            model=resp.model,
            provider=self.name,
            raw=resp,
            **from_openai_chat(getattr(resp, "usage", None)),
        )

    def _user_content(self, request: Request, vision: bool):
        """Plain string for text turns; a multimodal `[text, image_url...]` array
        for vision models with image attachments. Non-vision models reject
        images; PDFs are unsupported on any GLM."""
        for a in request.attachments:
            if a.mime_type == "application/pdf":
                raise RuntimeError(
                    "GLM does not accept PDF attachments (Z.AI file_url needs a "
                    "hosted URL, not base64). Use claude-opus-4-8 or "
                    "gemini-3.1-pro-preview for PDFs."
                )
            if not a.mime_type.startswith("image/"):
                raise RuntimeError(
                    f"GLM cannot encode attachment {a.source_label!r} "
                    f"(mime {a.mime_type})."
                )

        if not request.attachments:
            return request.prompt
        if not vision:
            raise RuntimeError(
                f"GLM model {request.model!r} is not a vision model; it cannot "
                f"accept images. Use a vision GLM: glm-4.6v, glm-4.5v, "
                f"glm-5v-turbo, or glm-ocr."
            )
        parts: list[dict] = [{"type": "text", "text": request.prompt}]
        parts.extend(_image_part(a) for a in request.attachments)
        return parts
