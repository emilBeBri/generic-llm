"""Provider selection from a model name."""

from __future__ import annotations


def provider_for(model: str) -> str:
    m = model.lower()
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "gemini"
    # gpt-*, o1, o3, o4, codex, ...
    return "openai"
