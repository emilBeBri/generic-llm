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


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        self.client = genai.Client(api_key=key)

    def generate(self, request: Request) -> Response:
        config_args: dict = {
            "max_output_tokens": request.max_tokens,
        }
        if request.system:
            config_args["system_instruction"] = request.system
        if request.temperature is not None:
            config_args["temperature"] = request.temperature

        if request.schema is not None:
            config_args["response_mime_type"] = "application/json"
            config_args["response_json_schema"] = request.schema
        elif request.json_mode:
            config_args["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_args)

        resp = self.client.models.generate_content(
            model=request.model,
            contents=request.prompt,
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
