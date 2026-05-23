"""Domain types — provider-agnostic.

The only thing a caller hands an adapter is a Request. The only thing it gets
back is a Response. Both are dumb data classes with no provider awareness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Request:
    prompt: str
    system: str | None = None
    model: str = ""
    max_tokens: int = 4096
    temperature: float | None = None
    # JSON Schema (dict). When set, response.text is guaranteed to be JSON
    # validating against this schema, via each provider's native mechanism.
    schema: dict[str, Any] | None = None
    # Generic JSON mode (no schema). Mostly useful for providers that accept
    # a `response_mime_type=application/json` hint. For Anthropic (no native
    # json-object mode) we add an instruction prefix.
    json_mode: bool = False


@dataclass
class Response:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = field(default=None, repr=False)
