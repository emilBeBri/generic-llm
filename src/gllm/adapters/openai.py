"""OpenAI adapter.

Two API surfaces:
- Responses API: used for o1/o3/o4 reasoning models and the gpt-5 family.
- Chat Completions: used for gpt-4o, gpt-4.1, etc.

Structured output:
- Responses:        text = {"format": {"type": "json_schema", "name": ..., "schema": ..., "strict": True}}
- Chat Completions: response_format = {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}

Plain JSON mode (no schema):
- Responses:        text = {"format": {"type": "json_object"}}
- Chat Completions: response_format = {"type": "json_object"}
"""

from __future__ import annotations

import os

from openai import OpenAI

from ..domain import Request, Response
from ..ports import LLMProvider

_RESPONSES_PREFIXES = ("o1", "o3", "o4", "gpt-5", "codex")


def _use_responses_api(model: str) -> bool:
    m = model.lower()
    return any(m.startswith(p) for p in _RESPONSES_PREFIXES)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key, max_retries=3)

    def generate(self, request: Request) -> Response:
        if _use_responses_api(request.model):
            return self._generate_responses(request)
        return self._generate_chat(request)

    def _generate_responses(self, request: Request) -> Response:
        kwargs: dict = {
            "model": request.model,
            "input": request.prompt,
            "max_output_tokens": request.max_tokens,
            "store": False,
        }
        if request.system:
            kwargs["instructions"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "Output",
                    "schema": request.schema,
                    "strict": True,
                }
            }
        elif request.json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}

        resp = self.client.responses.create(**kwargs)

        # SDK exposes a flattened `output_text` aggregating all text deltas.
        text = getattr(resp, "output_text", "") or ""

        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0

        return Response(
            text=text,
            model=request.model,
            provider=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw=resp,
        )

    def _generate_chat(self, request: Request) -> Response:
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

        if request.schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "Output",
                    "schema": request.schema,
                    "strict": True,
                },
            }
        elif request.json_mode:
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
