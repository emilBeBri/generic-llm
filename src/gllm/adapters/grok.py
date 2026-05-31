"""xAI Grok adapter.

Grok speaks the OpenAI Responses API byte-for-byte at api.x.ai/v1, so it
inherits OpenAIProvider wholesale — same `text.format` json_schema path, same
output shape. `grok-*` is in the Responses-API prefix list (see
_capabilities.py), so dispatch falls out of the parent with no override.

The subclass exists only to point at xAI's base_url, read XAI_API_KEY, and
tag the provider as 'grok' rather than 'openai'. It is also the override
surface for any future xAI drift from the OpenAI shape.
"""

from __future__ import annotations

import os

from ..domain import Request, Response
from .openai import OpenAIProvider

GROK_BASE_URL = "https://api.x.ai/v1"


class GrokProvider(OpenAIProvider):
    name = "grok"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("XAI_API_KEY")
        if not key:
            raise RuntimeError("XAI_API_KEY is not set")
        super().__init__(api_key=key, base_url=GROK_BASE_URL, name="grok")

    def generate(self, request: Request) -> Response:
        # xAI's Responses API accepts input_image but has no input_file
        # equivalent for PDFs. Reject those up front rather than letting the
        # API return an opaque 400.
        for a in request.attachments:
            if a.mime_type == "application/pdf":
                raise RuntimeError(
                    "grok does not accept PDF attachments. Try claude-opus-4-8, "
                    "a gpt-5/o-series model, or gemini-3-pro-preview."
                )
        return super().generate(request)
