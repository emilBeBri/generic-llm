# ADR: `GLLM_BASE_URL_<TAG>` — one uniform way to redirect a provider's endpoint

Adapters hardcode their endpoint (`https://api.deepseek.com`, `https://api.x.ai/v1`, …), which is right for normal use but makes a provider unreachable through a proxy. `config.resolve_base_url(tag, default, *, legacy_env=None)` lets the environment redirect any provider via `GLLM_BASE_URL_<TAG>`, where `<TAG>` is the provider tag upper-cased (`GLLM_BASE_URL_DEEPSEEK`, `GLLM_BASE_URL_GROQ`).

`#architecture-decision-record` `#environment`

## Why one generic variable instead of per-provider ones

The alternative — eleven differently-spelled env vars — pushes the naming problem onto every caller. One predictable pattern keyed off the tag the registry already uses means a caller can redirect one provider or all of them without editing code. `ZAI_BASE_URL` predates this and is preserved via the `legacy_env` parameter; the generic name wins when both are set, and a blank/whitespace value is ignored (so an exported-but-empty var doesn't silently break the default).

## The concrete consumer: the jail credential broker

`.control-center/bb-scripts/llm-key-broker.py` keeps real API keys on the host and hands a sandboxed agent a **loopback URL plus a per-launch session token** instead. gllm running inside `claude-jail` therefore needs its base URLs pointed at `127.0.0.1:<port>/<provider>/` — which is exactly what the broker sets. Verified live from inside the jail: with no override in the shell, `resolve_base_url("deepseek", …)` returned the broker's `http://127.0.0.1:33401/deepseek/`, because the jail had already exported `GLLM_BASE_URL_DEEPSEEK`.

## Coverage: only the adapters that need it

Wired into `deepseek`, `grok`, `kimi`, `zai`, and `openai_compat` — the last one covers **every** OpenAI-compatible host (groq, regolo, …) in a single edit, since they all resolve their endpoint through the shared `ProviderSpec`. The `openai`, `anthropic` and `gemini` adapters need nothing: their SDKs already read their own base-URL env vars. This asymmetry is deliberate, not an oversight — don't "complete" it by adding a second override path where the SDK already has one.

Related: `ADR-provider-model-axis.md` (where the provider tag comes from), `GOTCHA-azure-foundry-constraints.md` (the other env-driven routing switch, `WORK=1`).
