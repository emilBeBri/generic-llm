"""Model -> provider resolution.

`provider_for` is a registry lookup and THE single resolver: `MODELS[key].provider`.
It is never a substring guess on the model name, because the name tells you which
lab TRAINED a model, not which API ANSWERS for it — `groq:openai/gpt-oss-120b` is
an OpenAI model served by Groq, `regolo:glm5.2-beta` is a Z.AI model served by
Regolo, and a host id like `groq:deepseek-r1-distill-llama-70b` would send the old
`if "deepseek" in m` ladder straight to the wrong API.

The old ladder survives as `_legacy_guess_provider`, used only for names that have
no registry row. Unknown is NOT an error: vendors ship models faster than this repo
gets updated, and `gllm --models` (a live API probe) is the authority on what
exists. But it IS worth one warning — a silently mis-routed hallucinated model name
is how you end up reporting success for a model nobody asked for.

`WORK=1` (see config.work_env) is the corporate/Azure switch: `effective_model`
swaps a model for its Azure Foundry deployment via `ModelSpec.azure_alias`.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from .models import MODELS, spec_for

# Only these two direct providers have an Azure Foundry counterpart; WORK mode
# redirects them. Gemini/Grok/DeepSeek/GLM have no Azure variant.
WORK_PROVIDER_REDIRECTS = {
    "anthropic": "azure_anthropic",
    "openai": "azure_openai",
}


def provider_for(model: str) -> str:
    spec = spec_for(model)
    if spec is not None:
        return spec.provider
    return _legacy_guess_provider(model)


@lru_cache(maxsize=256)
def _legacy_guess_provider(model: str) -> str:
    """Pre-registry substring ladder, for unregistered names only.

    lru_cached so the warning fires once per name per process, not once per call.
    Do not extend this — a new model gets a MODELS row.
    """
    m = (model or "").lower()
    # Host namespaces first: their ids embed other vendors' names by design
    # ('groq:openai/gpt-oss-120b', 'regolo:glm5.2-beta'), so every check below
    # would misfire on them.
    if ":" in m:
        tag = m.split(":", 1)[0]
        if tag in ("groq", "regolo"):
            guess = tag
            _warn_unregistered(model, guess)
            return guess
    if m.endswith("-dev"):
        guess = "azure_anthropic" if "claude" in m else "azure_openai"
    elif "claude" in m:
        guess = "anthropic"
    elif "gemini" in m:
        guess = "gemini"
    elif "deepseek" in m:
        guess = "deepseek"
    elif "grok" in m:
        guess = "grok"
    elif "glm" in m:
        guess = "zai"
    elif m.startswith("kimi-"):
        guess = "kimi"
    else:
        # gpt-*, o1, o3, o4, codex, and anything unrecognised.
        guess = "openai"
    _warn_unregistered(model, guess)
    return guess


def _warn_unregistered(model: str, guess: str) -> None:
    print(
        f"gllm: {model!r} is not in the model registry; routing to {guess!r} by "
        f"name guess. If the name is real, add it to gllm/models.py; if you are "
        f"not sure it is real, check `gllm --models {guess}`.",
        file=sys.stderr,
    )


def effective_model(model: str, work: bool) -> str:
    """The model name actually sent downstream.

    Under WORK mode a model is swapped for its Azure Foundry deployment, named
    explicitly by `ModelSpec.azure_alias` rather than derived by appending
    `-dev`. Already-Azure names, providers with no Azure counterpart, and
    `work=False` pass through unchanged.

    A registered Anthropic/OpenAI model with no `azure_alias` still gets the
    historical `-dev` append, but loudly: Azure deployment inventory is live
    data the registry cannot know, so the guess may well 404
    (DeploymentNotFound) and you should hear about it before the API says so.
    """
    if not work:
        return model
    spec = spec_for(model)
    if spec is not None:
        if spec.provider.startswith("azure_"):
            return model
        if spec.azure_alias:
            return spec.azure_alias
        if spec.provider not in WORK_PROVIDER_REDIRECTS:
            return model
        return _guess_azure_deployment(model)
    if model.lower().endswith("-dev"):
        return model
    if provider_for(model) in WORK_PROVIDER_REDIRECTS:
        return _guess_azure_deployment(model)
    return model


@lru_cache(maxsize=128)
def _guess_azure_deployment(model: str) -> str:
    candidate = f"{model}-dev"
    if candidate.lower() not in MODELS:
        print(
            f"gllm: WORK=1 but no known Azure deployment for {model!r}; trying "
            f"{candidate!r}. Add an azure_alias to gllm/models.py once you know "
            f"the real deployment name.",
            file=sys.stderr,
        )
    return candidate
