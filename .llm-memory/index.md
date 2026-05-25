# .llm-memory index

Persistent project knowledge for `generic-llm` (the `gllm` CLI).

## Conventions
- [CONVENTIONS-schemas-and-instructions.md](CONVENTIONS-schemas-and-instructions.md) — Two-tier library layout (bundled `data/` + future `~/.config/gllm/` overlay), the all-required + empty-string-sentinel schema convention (for OpenAI strict-mode portability), and the instruction-authoring patterns.
- [CONVENTIONS-multi-provider-routing.md](CONVENTIONS-multi-provider-routing.md) — `provider_for` model-name inference (the `-dev` suffix is the Azure Foundry marker), and the OpenAIProvider subclass pattern that DeepSeek/Grok/Azure-OpenAI reuse.

## Architecture / gotchas
- [GOTCHA-azure-foundry-constraints.md](GOTCHA-azure-foundry-constraints.md) — Azure Foundry quirks: no `output_config` (JSON emulated via instruction), endpoint rewriting, and the `WORK=1`/`WORK_ENV` forced-thinking mechanism for Azure Anthropic.

## Ideas / future features
- [IDEAS-key-loading-secret-managers.md](IDEAS-key-loading-secret-managers.md) — `--keys-from pass:...` / sops / keyring integration. Not built in v1.
