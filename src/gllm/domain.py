"""Domain types — provider-agnostic.

The only thing a caller hands an adapter is a Request. The only thing it gets
back is a Response. Both are dumb data classes with no provider awareness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Attachment:
    """A binary input (image, PDF) attached to a request.

    `source_label` is the path the bytes came from, or `"<stdin>"` when read
    from `-f -`. Used in error messages and as a fallback filename for
    providers (e.g. OpenAI's `input_file` wants a filename).
    """

    data: bytes
    mime_type: str
    source_label: str


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
    # Binary attachments (images, PDFs). Each adapter uses its provider's
    # native attachment API; providers without one raise on non-empty.
    attachments: tuple[Attachment, ...] = ()


@dataclass
class Response:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = field(default=None, repr=False)
