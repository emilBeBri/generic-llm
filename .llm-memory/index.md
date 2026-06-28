# .llm-memory index

Persistent project knowledge for `generic-llm` (the `gllm` CLI).

## Conventions
- [CONVENTIONS-schemas-and-instructions.md](CONVENTIONS-schemas-and-instructions.md) — Two-tier library layout (bundled `data/` + future `~/.config/gllm/` overlay), the all-required + empty-string-sentinel schema convention (for OpenAI strict-mode portability), and the instruction-authoring patterns.
- [CONVENTIONS-multi-provider-routing.md](CONVENTIONS-multi-provider-routing.md) — `provider_for` model-name inference (the `-dev` suffix is the Azure Foundry marker), the OpenAIProvider subclass pattern that DeepSeek/Grok/Azure-OpenAI reuse, and the 4-6/4-7/4-8 thinking family on Azure Anthropic.
- [CONVENTIONS-file-attachments.md](CONVENTIONS-file-attachments.md) — `-f PATH` (with `-` for stdin, process substitution for free), the `Attachment` domain type, and the native-or-fail capability matrix.
- [CONVENTIONS-zai-glm-adapter.md](CONVENTIONS-zai-glm-adapter.md) — Z.AI/GLM provider (`glm-*` → `zai`, OpenAI Chat @ api.z.ai): standalone NOT an OpenAIProvider subclass (else routed to Responses API GLM can't speak), `thinking`+`reasoning_effort` (glm-5.2 only), vision-model split enforced in-adapter, json_object-only (`--schema` refused), listable via `models.list()`.
- [CONVENTIONS-porting-adapters-from-reference.md](CONVENTIONS-porting-adapters-from-reference.md) — how to port a provider adapter from bebri-chat (the reference): reshape not copy (mirror protocol facts; drop tools/web-search/registry; deviate to gllm's fail-loud + trust-the-docs house rules). The process rule: every port must read the reference + provider docs concretely AND **ask the user** if this reshape is right for that provider (noting it's the established pattern). GLM is the first worked example.
- [CONVENTIONS-usage-cost-emission.md](CONVENTIONS-usage-cost-emission.md) — `--usage` emits one `gllm-usage {json}` line to stderr (tokens/cache/reasoning + provider-verbatim `usage_raw`); per-provider mappers in `usage.py` normalise at the source; `Response` gained cache/reasoning fields; dollar cost is deliberately the consumer's job (not gllm's).

## Architecture / gotchas
- [ADR-reasoning-effort-ladder.md](ADR-reasoning-effort-ladder.md) — the `-r/--reasoning low/medium/high/xhigh` knob: one abstract ladder translated per provider (`reasoning.py`), `supports_reasoning` fail-loud gate, hands-off default, and `WORK=1` ≡ the `xhigh` rung.
- [GOTCHA-azure-foundry-constraints.md](GOTCHA-azure-foundry-constraints.md) — Azure Foundry specifics: it DOES expose `output_config` (`effort` verified, `format` an unverified native attempt — corrected 2026-06-17), endpoint rewriting, and the `WORK=1`/`WORK_ENV` direct-vs-Azure routing toggle (`effective_model` appends `-dev`).
- [ADR-model-listing-live-probe.md](ADR-model-listing-live-probe.md) — there is NO model allowlist (adapters forward the name verbatim, so hand-maintained catalogs drift and lie — `gemini-3-flash-preview` was wrongly called "retired"); `gllm --models [PROVIDER]` probes each provider's live `models.list()` and prints greppable text-gen `provider<TAB>id` rows (loud-skip on no key, Azure excluded as deployment-scoped).

## Ideas / future features
- [IDEAS-key-loading-secret-managers.md](IDEAS-key-loading-secret-managers.md) — `--keys-from pass:...` / sops / keyring integration. Not built in v1.
