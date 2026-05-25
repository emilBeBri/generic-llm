"""Provider selection from a model name.

Mirrors bebri-chat's `get_model_provider`: the `-dev` suffix is the explicit
Azure Foundry marker and is checked first, so `claude-opus-4-7-dev` routes to
Azure rather than the direct Anthropic adapter.
"""

from __future__ import annotations


def provider_for(model: str) -> str:
    m = model.lower()
    # Azure Foundry deployments carry a `-dev` suffix.
    if m.endswith("-dev"):
        return "azure_anthropic" if "claude" in m else "azure_openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "gemini"
    if "deepseek" in m:
        return "deepseek"
    if "grok" in m:
        return "grok"
    # gpt-*, o1, o3, o4, codex, ...
    return "openai"
