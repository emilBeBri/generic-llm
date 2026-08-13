"""Domain types — provider-agnostic.

The only thing a caller hands an adapter is a Request. The only thing it gets
back is a Response. Both are dumb data classes with no provider awareness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Attachment:
    """A file attached to a request through a provider's native input API.

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
    # The app-facing model identity: the registry key, what the user typed after
    # -m, and the join key for pricing and `--usage`.
    model: str = ""
    # The literal string to put on the wire. Differs from `model` only for
    # namespaced host rows ('groq:openai/gpt-oss-120b' -> 'openai/gpt-oss-120b').
    # Empty means "same as model" — adapters read `wire_model or model`.
    wire_model: str = ""
    max_tokens: int = 4096
    temperature: float | None = None
    # JSON Schema (dict). When set, response.text is guaranteed to be JSON
    # validating against this schema, via each provider's native mechanism.
    schema: dict[str, Any] | None = None
    # Generic JSON mode (no schema). Mostly useful for providers that accept
    # a `response_mime_type=application/json` hint. For Anthropic (no native
    # json-object mode) we add an instruction prefix.
    json_mode: bool = False
    # Native file attachments. Each adapter uses its provider's
    # native attachment API; providers without one raise on non-empty.
    attachments: tuple[Attachment, ...] = ()
    # Abstract reasoning-effort level: one of reasoning.LEVELS (low/medium/high/
    # xhigh) or None (hands-off — no reasoning param is sent). This is the rung
    # the USER asked for, kept for --usage reporting.
    reasoning: str | None = None
    # The provider's own effort value that `reasoning` resolved to, e.g. "max"
    # on DeepSeek for `-r xhigh`. Resolved once in the CLI so every adapter
    # sends the same thing and the translation notice is printed in one place.
    # Empty when `reasoning` is None.
    wire_effort: str = ""


# Every provider's own word for "I stopped because the output budget ran out".
# Compared case-folded, which is what lets Gemini's `MAX_TOKENS` enum name share
# an entry with Anthropic's `max_tokens`. Anything else — "stop", "end_turn",
# "tool_use" — is a complete answer.
_TRUNCATION_REASONS = frozenset({"max_tokens", "max_output_tokens", "length"})


@dataclass
class Response:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Richer usage, populated per-provider by gllm.usage (0 / {} where the
    # provider doesn't report it). cache_write is Anthropic's pricier cache-
    # creation; reasoning_tokens is only what a provider breaks out separately.
    # `usage_raw` is the provider's own usage dict, verbatim — the ground truth
    # for exact cost accounting (the normalised fields are a rough common view).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    usage_raw: dict = field(default_factory=dict)
    # The provider's own stop/finish reason, VERBATIM — `end_turn`, `stop`,
    # `max_tokens`, `MAX_TOKENS`, `length`. Not normalised, for the same reason
    # `usage_raw` isn't: the provider's word is the ground truth and the
    # vocabularies genuinely differ. None when a provider didn't say.
    stop_reason: str | None = None
    raw: Any = field(default=None, repr=False)

    @property
    def truncated(self) -> bool:
        """Did generation stop because it ran out of output budget?

        Worth surfacing because a truncated answer is indistinguishable from a
        complete one on stdout — a `--max-tokens 120` Gemini call printed `1` for
        `23*47` and looked like a confident wrong answer, not a cut-off one.
        """
        return (self.stop_reason or "").strip().lower() in _TRUNCATION_REASONS
