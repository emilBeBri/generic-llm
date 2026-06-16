"""Which OpenAI API surface does a model use? And which providers handle
which kinds of attachments natively?

Single source of truth shared by `openai.py`, `grok.py` (via subclass), and
`azure_openai.py`, so we don't scatter `if "codex" in model or "o1" in model`
checks across adapters.

* Reasoning / agentic models -> Responses API (`/v1/responses`)
  - o1, o3, o4, gpt-5 (incl. gpt-5.5), codex, and xAI grok-*
* Classic chat models -> Chat Completions (`/v1/chat/completions`)
  - gpt-4, gpt-4o, gpt-3.5
* Unknown slugs default to Responses (OpenAI's "Responses for everything new"
  direction; it is the strict superset). Azure `-dev` deployments share the
  same prefixes, so `gpt-5.1-dev` -> Responses, `gpt-4o-dev` -> Chat.

Attachment capability follows the rule **native or fail** — no text-extraction
fallback. PDF support on OpenAI is only on the Responses API (Chat Completions
has no PDF content-block type).
"""

from __future__ import annotations

# Order matters: Responses is checked first so "gpt-5" is not swallowed by the
# "gpt-4"/"gpt-3.5" chat check (it wouldn't be, but keep the intent explicit).
_RESPONSES_API_PREFIXES = ("o1", "o3", "o4", "gpt-5", "codex", "grok")
_CHAT_COMPLETIONS_PREFIXES = ("gpt-4", "gpt-3.5")


def use_responses_api(model: str) -> bool:
    m = (model or "").strip().lower()
    if any(m.startswith(p) for p in _RESPONSES_API_PREFIXES):
        return True
    # Classic chat models -> Chat; everything else (incl. unknown) -> Responses.
    return not any(m.startswith(p) for p in _CHAT_COMPLETIONS_PREFIXES)


_IMAGE_PROVIDERS = {
    "anthropic",
    "azure_anthropic",
    "openai",
    "azure_openai",
    "gemini",
    "grok",
}


def supports_image(provider: str) -> bool:
    return provider in _IMAGE_PROVIDERS


def supports_pdf(provider: str, model: str) -> bool:
    if provider in {"anthropic", "azure_anthropic", "gemini"}:
        return True
    if provider in {"openai", "azure_openai"}:
        # OpenAI native PDF input is `input_file` on the Responses API only.
        return use_responses_api(model)
    # grok, deepseek: no PDF support today.
    return False


def supports_reasoning(provider: str, model: str) -> bool:
    """Can this (provider, model) honour a `--reasoning` level?

    Anthropic/Gemini families think across the board. OpenAI-compatible backends
    only reason on the Responses API (o-series, gpt-5, grok-*) — the Chat
    Completions models (gpt-4o, gpt-4.1) have no reasoning control. DeepSeek
    reasons by default but exposes no effort knob, so we cannot honour a level.
    """
    if provider in {"anthropic", "azure_anthropic", "gemini"}:
        return True
    if provider in {"openai", "azure_openai", "grok"}:
        return use_responses_api(model)
    # deepseek and anything unknown: no control surface.
    return False
