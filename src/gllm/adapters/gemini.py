"""Gemini adapter.

Single non-streaming call via `client.models.generate_content`.

Structured output:
- response_mime_type = "application/json"
- response_json_schema = <the schema dict>

Plain JSON mode (no schema):
- response_mime_type = "application/json"
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from ..domain import Request, Response
from ..ports import LLMProvider
from ..reasoning import gemini_thinking_budget
from ._capabilities import is_text_generation_model


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        self.client = genai.Client(api_key=key)

    def list_models(self) -> list[str]:
        # Two-stage filter. First the API's own signal: keep only models whose
        # `supported_actions` includes `generateContent` (drops embeddings,
        # which expose `embedContent`). But TTS/image/music models advertise
        # `generateContent` too, so also apply the name-based text-gen filter.
        # Names arrive as `models/gemini-3-flash-preview` — strip the prefix.
        out: list[str] = []
        for m in self.client.models.list():
            if "generateContent" not in (m.supported_actions or []):
                continue
            mid = m.name.split("/", 1)[-1]
            if is_text_generation_model(mid):
                out.append(mid)
        return sorted(out)

    def generate(self, request: Request) -> Response:
        reasoning_on = request.reasoning is not None
        # Thinking tokens count against the output budget; raise the floor so
        # the visible answer isn't starved.
        max_out = max(request.max_tokens, 16000) if reasoning_on else request.max_tokens
        config_args: dict = {
            "max_output_tokens": max_out,
        }
        if request.system:
            config_args["system_instruction"] = request.system
        # Gemini accepts a custom temperature alongside thinking.
        if request.temperature is not None:
            config_args["temperature"] = request.temperature

        if request.schema is not None:
            config_args["response_mime_type"] = "application/json"
            config_args["response_json_schema"] = request.schema
        elif request.json_mode:
            config_args["response_mime_type"] = "application/json"

        if reasoning_on:
            config_args["thinking_config"] = types.ThinkingConfig(
                thinking_budget=gemini_thinking_budget(request.reasoning, request.model)
            )

        config = types.GenerateContentConfig(**config_args)

        if request.attachments:
            contents = [
                types.Part.from_bytes(data=a.data, mime_type=a.mime_type)
                for a in request.attachments
            ]
            # Gemini SDK accepts a trailing string as a text Part.
            contents.append(request.prompt)
        else:
            contents = request.prompt

        resp = self.client.models.generate_content(
            model=request.model,
            contents=contents,
            config=config,
        )

        text = resp.text or ""

        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

        return Response(
            text=text,
            model=request.model,
            provider=self.name,
            input_tokens=in_tok or 0,
            output_tokens=out_tok or 0,
            raw=resp,
        )
