# ADR: `GLLM_BASE_URL_<TAG>` — one uniform way to redirect a provider's endpoint

Adapters hardcode their endpoint (`https://api.deepseek.com`, `https://api.x.ai/v1`, …), which is right for normal use but makes a provider unreachable through a proxy. `config.resolve_base_url(tag, default, *, legacy_env=None)` lets the environment redirect any provider via `GLLM_BASE_URL_<TAG>`, where `<TAG>` is the provider tag upper-cased (`GLLM_BASE_URL_DEEPSEEK`, `GLLM_BASE_URL_GROQ`).

`#architecture-decision-record` `#environment`

## Why one generic variable instead of per-provider ones

The alternative — eleven differently-spelled env vars — pushes the naming problem onto every caller. One predictable pattern keyed off the tag the registry already uses means a caller can redirect one provider or all of them without editing code. `ZAI_BASE_URL` predates this and is preserved via the `legacy_env` parameter; the generic name wins when both are set, and a blank/whitespace value is ignored (so an exported-but-empty var doesn't silently break the default).

## The concrete consumer: the jail credential broker

`.control-center/bb-scripts/llm-key-broker.py` keeps real API keys on the host and hands a sandboxed agent a **loopback URL plus a per-launch session token** instead. gllm running inside `claude-jail` therefore needs its base URLs pointed at `127.0.0.1:<port>/<provider>/` — which is exactly what the broker sets. Verified live from inside the jail: with no override in the shell, `resolve_base_url("deepseek", …)` returned the broker's `http://127.0.0.1:33401/deepseek/`, because the jail had already exported `GLLM_BASE_URL_DEEPSEEK`.

**A host-side key change reaches a jailed session only on relaunch.** The broker captures credentials when it starts, and a running session keeps hitting the port it was given, so rotating a key outside the jail changes nothing inside it — a dead key keeps returning the same provider error and it looks like the code. Confirmed 2026-08-13: an exhausted `ZAI_API_KEY` kept returning `429 code 1113` after the key was replaced, and only a session restart fixed it. The tell is that **both the broker port and the session token change** across a restart (`127.0.0.1:42637` → `43717`, token prefix `bVvV` → `b-oB`); if they are unchanged, you are talking to the old broker. Nothing inside the sandbox can restart it, `!`-prefixed commands included.

Note which slot the broker occupies per provider, because it is not uniform: it exported `ZAI_BASE_URL` — the *legacy* name — for Z.AI, leaving `GLLM_BASE_URL_ZAI` unset. Since the broker owns that slot, gllm inside the jail **cannot** be pointed at Z.AI's coding-plan endpoint, which is the documented cause of a spurious `1113` (see [[CONVENTIONS-zai-glm-adapter]]). That has to be fixed broker-side, not in gllm.

## Coverage: only the adapters that need it

Wired into `deepseek`, `grok`, `kimi`, `zai`, `openai_compat` — the last one covers **every** OpenAI-compatible host (groq, regolo, …) in a single edit, since they all resolve their endpoint through the shared `ProviderSpec` — and now `openai` too.

**Every provider now goes through `resolve_base_url`.** This note used to say `openai`/`anthropic`/`gemini` "need nothing: their SDKs already read their own base-URL env vars" — true only while the SDKs were there. With the transport rewrite complete ([[ADR-stdlib-http-transport]]) each adapter owns its endpoint, and each kept the variable its SDK used to read as `legacy_env`, because those are exactly what the broker sets: `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_GEMINI_BASE_URL`. Omitting any one would have broken every jailed call for that provider while presenting as an auth failure.

One trap the Gemini case exposed: **the API version must stay out of the base URL.** The SDK appended `/v1beta` itself, so an override points at the host; folding the version into the default makes every override 404. Keep version segments in the request path, not the base.

Related: `ADR-provider-model-axis.md` (where the provider tag comes from), `GOTCHA-azure-foundry-constraints.md` (the other env-driven routing switch, `WORK=1`).
