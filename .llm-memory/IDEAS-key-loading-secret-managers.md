# Future feature: load API keys from secret managers

Not built yet. Drop here for when it matters.

## Current behavior (v1)

`gllm` resolves provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) in this order:
1. Process environment
2. `~/.config/gllm/.env` (key=value lines; chmod 600 recommended)

Rationale: no `.zshrc` exports needed, so keys aren't broadcast to every child process the user ever spawns. Loading is gllm-local — only the gllm process and its children see the resolved keys.

## Future: `--keys-from` flag

Add a flag that fetches keys on demand from a secret manager instead of reading them from a file at rest:

```
gllm --keys-from pass:api/anthropic         # pass(1) — GPG-encrypted
gllm --keys-from sops:~/secrets.yaml         # mozilla/sops
gllm --keys-from keyring:gllm                # OS keyring (Secret Service / macOS Keychain)
gllm --keys-from 1p:op://Personal/Anthropic  # 1Password CLI
```

Implementation sketch:
- Parse `scheme:locator` syntax
- Dispatch to a small registry of fetchers (each shells out to the relevant CLI or uses the keyring Python package)
- Apply fetched values as env vars *for the gllm process only* (no leak into parent shell)
- One key per call OR a single locator pointing at a multi-key blob — both shapes useful

Why later, not now:
- v1's chmod-600 `~/.config/gllm/.env` is already a big improvement over `.zshrc` exports
- Real secret-manager integration is per-user (everyone has a different setup: `pass`, 1Password, Bitwarden, sops, age, systemd-creds...) — pick the one(s) you actually use rather than implementing all five speculatively
- Need a real use case (e.g. running gllm on a shared box, or wanting per-invocation audit logs) before deciding which to support first

## Related considerations
- If we later add a `gllm serve` / daemon mode, keys-in-env becomes more of a liability (long-lived process, broader blast radius from a memory leak / coredump). Daemon mode is the natural trigger for prioritizing this work.
- Don't ever log resolved key values, even at `-v`. Mask in any error path.
