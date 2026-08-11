"""Process-level configuration read from the environment.

gllm has no settings object (it is a one-shot CLI); the few cross-cutting
toggles ported from bebri-chat live here as plain env lookups. Keys are
loaded into os.environ by cli._load_user_env_file before anything reads them.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def work_env() -> bool:
    """Corporate/Azure "work" mode toggle. Default off.

    In bebri-chat this is the `WORK_ENV` setting; here `WORK=1` also works as an
    ergonomic per-invocation flag (`WORK` wins over `WORK_ENV`). It selects the
    Azure Foundry adapters over the direct providers. It has nothing to do with
    reasoning — that is `--reasoning` (see gllm.reasoning).
    """
    val = (os.environ.get("WORK") or os.environ.get("WORK_ENV") or "").strip().lower()
    return val in _TRUTHY


def resolve_base_url(
    tag: str, default: str | None, *, legacy_env: str | None = None
) -> str | None:
    """Let the environment redirect a provider's base URL.

    Most adapters hardcode their endpoint, which is correct for normal use but
    makes the provider unreachable through a proxy. `GLLM_BASE_URL_<TAG>` is a
    uniform override so a caller can point one provider — or all of them — at a
    local gateway without editing code or juggling eleven differently-spelled
    env vars.

    The concrete consumer is `.control-center/bb-scripts/llm-key-broker.py`,
    which keeps real API keys on the host and hands a sandboxed agent a
    loopback URL plus a session token. Providers driven by an SDK that already
    reads its own base-URL env var (openai, anthropic, gemini) need nothing
    here; this covers the ones that do not.

    `legacy_env` preserves a provider-specific variable that predates the
    generic one (currently only `ZAI_BASE_URL`).
    """
    names = (f"GLLM_BASE_URL_{tag.upper()}", *((legacy_env,) if legacy_env else ()))
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return default
