"""Process-level configuration read from the environment.

gllm has no settings object (it is a one-shot CLI); the few cross-cutting
toggles ported from bebri-chat live here as plain env lookups. Keys are
loaded into os.environ by cli._load_user_env_file before anything reads them.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def work_env() -> bool:
    """Corporate/Azure "work" mode.

    In bebri-chat this is the `WORK_ENV` setting; here it doubles as an
    ergonomic per-invocation flag, so `WORK=1 gllm -m claude-opus-4-7-dev ...`
    works too. When on, the Azure Anthropic adapter forces maximum extended
    thinking (see adapters/azure_anthropic.py). `WORK` wins over `WORK_ENV`.
    """
    val = (os.environ.get("WORK") or os.environ.get("WORK_ENV") or "").strip().lower()
    return val in _TRUTHY
