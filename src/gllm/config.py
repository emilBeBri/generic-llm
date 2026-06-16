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
