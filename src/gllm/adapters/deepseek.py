"""DeepSeek adapter.

DeepSeek's API is OpenAI-compatible (chat completions) at api.deepseek.com,
so we drive it with the `openai` SDK pointed at a different base_url. Models:
`deepseek-v4-pro`, `deepseek-v4-flash`.

Thinking: the v4 models reason by default and emit a `reasoning_content`
field alongside `content`. gllm is one-shot and prints only the final text,
so we discard reasoning and don't touch the thinking config (API default).

Structured output: DeepSeek has no native json_schema/strict mode — only
`response_format={"type": "json_object"}`. So a `--schema` is honoured on a
best-effort basis by switching on json_object and pasting the schema into the
system prompt; `--json` just flips json_object on.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from ..domain import Request, Response
from ..ports import LLMProvider

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL, max_retries=3)

    def generate(self, request: Request) -> Response:
        system = request.system
        if request.schema is not None:
            schema_txt = json.dumps(request.schema, indent=2)
            extra = (
                "Respond with valid JSON only, matching this JSON Schema. "
                "No prose, no code fences.\n\n" + schema_txt
            )
            system = f"{system}\n\n{extra}" if system else extra

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.schema is not None or request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content or ""

        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

        return Response(
            text=text,
            model=resp.model,
            provider=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw=resp,
        )
