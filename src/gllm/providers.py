"""Provider registry: the explicit provider axis.

Each `ProviderSpec` describes one LLM *host* — not one lab. `tag` is the
provider identity used everywhere a provider string appears: `ModelSpec.provider`,
`Response.provider`, `pricing.compute_cost`'s billing branches, and the
`cli._build_provider` dispatch. Routing is `MODELS[key].provider ->
PROVIDERS[tag]`, never a substring guess on the model name.

The host/lab distinction is the whole point: `openai/gpt-oss-120b` is an OpenAI
model served by Groq, `glm5.2-beta` is a Z.AI model served by Regolo. The lab
that trained a model says nothing about which API answers for it.

Config-layer data only: `adapter_kind` is a STRING key into `cli._build_provider`'s
kind->constructor map, NOT a class reference — this module must never import an
adapter (they pull in heavyweight SDKs, and gllm's per-branch lazy imports are
what stop a missing SDK breaking unrelated providers).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSpec:
    tag: str
    display_name: str
    # Key into cli._build_provider's kind->constructor map. Several tags share a
    # kind: 'groq' and 'regolo' are both 'openai_compat'.
    adapter_kind: str
    # Env var names holding this provider's API key; first one set wins (Gemini
    # accepts GEMINI_API_KEY or GOOGLE_API_KEY).
    api_key_env: tuple[str, ...]
    # openai_compat only: the OpenAI-SDK base_url.
    base_url: str | None = None
    # openai_compat only: merged into every request's extra_body (host quirks).
    extra_body: dict[str, object] = field(default_factory=dict)
    # openai_compat only: emit {"format": <mime>} inside image_url blocks
    # (Regolo's vision dialect).
    image_url_format_field: bool = False
    # Does the live API expose a catalog we can probe for `gllm --models`?
    listable: bool = True
    # Registry-key prefix for host providers ('groq:', 'regolo:'). `--models`
    # prepends it so a printed row is copy-pasteable straight into `-m`.
    key_namespace: str | None = None


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        "anthropic", "Anthropic", "anthropic", ("ANTHROPIC_API_KEY",)
    ),
    "openai": ProviderSpec("openai", "OpenAI", "openai", ("OPENAI_API_KEY",)),
    "gemini": ProviderSpec(
        "gemini", "Google Gemini", "gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    ),
    "deepseek": ProviderSpec(
        "deepseek", "DeepSeek", "deepseek", ("DEEPSEEK_API_KEY",)
    ),
    "grok": ProviderSpec("grok", "xAI Grok", "grok", ("XAI_API_KEY",)),
    "zai": ProviderSpec("zai", "Z.AI", "zai", ("ZAI_API_KEY",)),
    # --- Hosts: OpenAI-compatible serving of other labs' open models ---
    "groq": ProviderSpec(
        "groq",
        "Groq",
        "openai_compat",
        ("GROQ_API_KEY",),
        base_url="https://api.groq.com/openai/v1",
        key_namespace="groq:",
    ),
    "regolo": ProviderSpec(
        "regolo",
        "Regolo AI",
        "openai_compat",
        ("REGOLO_API_KEY",),
        base_url="https://api.regolo.ai/v1",
        # Regolo silently serves a DIFFERENT model when the requested one is
        # unavailable unless this is sent — which makes model identity, pricing
        # and every capability gate a lie. Never remove.
        # See bebri-chat/.llm-memory/regolo-silent-fallback-gotcha.md
        extra_body={"disable_fallbacks": True},
        image_url_format_field=True,
        key_namespace="regolo:",
    ),
    # --- Azure Foundry: deployment-scoped, so not catalog-listable ---
    "azure_openai": ProviderSpec(
        "azure_openai",
        "Azure OpenAI",
        "azure_openai",
        ("AZURE_OPENAI_API_KEY",),
        listable=False,
    ),
    "azure_anthropic": ProviderSpec(
        "azure_anthropic",
        "Azure Anthropic",
        "azure_anthropic",
        ("AZURE_ANTHROPIC_API_KEY",),
        listable=False,
    ),
}


LISTABLE_PROVIDERS: tuple[str, ...] = tuple(
    tag for tag, spec in PROVIDERS.items() if spec.listable
)
