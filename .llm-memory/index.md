# .llm-memory index

Persistent project knowledge for `generic-llm` (the `gllm` CLI).

## Conventions
- [CONVENTIONS-schemas-and-instructions.md](CONVENTIONS-schemas-and-instructions.md) — Two-tier library layout (bundled `data/` + future `~/.config/gllm/` overlay), the all-required + empty-string-sentinel schema convention (for OpenAI strict-mode portability), and the instruction-authoring patterns.

## Ideas / future features
- [IDEAS-key-loading-secret-managers.md](IDEAS-key-loading-secret-managers.md) — `--keys-from pass:...` / sops / keyring integration. Not built in v1.
