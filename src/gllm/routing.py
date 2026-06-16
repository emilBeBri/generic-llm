"""Provider selection from a model name.

Mirrors bebri-chat's `get_model_provider`: the `-dev` suffix is the explicit
Azure Foundry marker and is checked first, so `claude-opus-4-7-dev` routes to
Azure rather than the direct Anthropic adapter.

`WORK=1` (corporate/Azure mode, see config.work_env) is the env-level switch
for the same redirect: `effective_model` appends `-dev` to direct Anthropic/
OpenAI model names so a clean `claude-opus-4-8` lands on the Azure deployment.
"""

from __future__ import annotations

# Only these two direct providers have an Azure Foundry counterpart; WORK mode
# redirects them. Gemini/Grok/DeepSeek have no Azure variant, so they're left
# alone.
_AZURE_REDIRECTABLE = {"anthropic", "openai"}


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


def effective_model(model: str, work: bool) -> str:
    """The model name actually sent downstream.

    Under WORK mode, a direct Anthropic/OpenAI model is redirected to its Azure
    Foundry deployment by appending the `-dev` marker (so `provider_for` then
    routes it to the Azure adapter). Already-`-dev` names, other providers, and
    `work=False` pass through unchanged. An explicit `-dev` name still selects
    Azure regardless of WORK.
    """
    if not work or model.lower().endswith("-dev"):
        return model
    if provider_for(model) in _AZURE_REDIRECTABLE:
        return f"{model}-dev"
    return model
