"""DeepSeek adapter.

DeepSeek's API is OpenAI-compatible (chat completions) at api.deepseek.com,
so we drive it with the `openai` SDK pointed at a different base_url. Models:
`deepseek-v4-pro`, `deepseek-v4-flash`.

Thinking: the v4 models reason by default and emit a `reasoning_content`
field alongside `content`. gllm is one-shot and prints only the final text,
so we discard reasoning and don't touch the thinking config (API default).

Structured output: DeepSeek has no native json_schema/strict mode — only
`response_format={"type": "json_object"}`. `--json` flips that on (best-effort
JSON). `--schema` (which promises *enforced* structure) is REFUSED — we will not
fake strict enforcement with a prompt instruction. The CLI gates this earlier
(supports_strict_schema); the raise here is the library-use backstop.
"""

from __future__ import annotations

import os

from openai import OpenAI

from ..domain import Request, Response
from ..ports import LLMProvider
from ..usage import from_deepseek
from ._capabilities import is_text_generation_model

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL, max_retries=3)

    def list_models(self) -> list[str]:
        # OpenAI-compatible catalog endpoint; apply the same text-generation
        # filter for consistency (DeepSeek's catalog is all chat today).
        return sorted(
            m.id
            for m in self.client.models.list()
            if is_text_generation_model(m.id)
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

        kwargs: dict = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content or ""

        return Response(
            text=text,
            model=resp.model,
            provider=self.name,
            raw=resp,
            **from_deepseek(getattr(resp, "usage", None)),
        )
