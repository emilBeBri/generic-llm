"""Append-only JSONL log of every gllm call, for measuring REAL usage.

Cost and latency estimates made from rate cards have been wrong here by 3x —
a rate card cannot tell you how much a model thinks, how verbose it is, or how
often it is slow. Only the calls you actually make can. This writes one JSON
object per call so those questions become `jq` queries over your own history
instead of a benchmark someone has to remember to run.

**Opt-in, because it is not free.** Pricing a call loads the llm-price-tracker
book, ~150 ms of pydantic import that plain runs deliberately avoid (see
`pricing._load_book`). Enabling the log accepts that cost per call; leaving it
off costs nothing at all, not even an import.

    GLLM_CALL_LOG=1                     # on, at the default path below
    GLLM_CALL_LOG=/tmp/experiment.jsonl # on, at a path you choose
    GLLM_CALL_LOG=off                   # off (also: unset, "", 0, false, no)

The default path is `$XDG_STATE_HOME/gllm/calls.jsonl`, i.e. normally
`~/.local/state/gllm/calls.jsonl` — state, not config and not bundled data.
It lives OUTSIDE the repo on purpose: this is per-machine history that must
never be committed, and the surest way to guarantee that is for it to be
somewhere `git add` cannot reach. (`.gitignore` also covers `*.jsonl` for the
case where you point the variable at a path inside the checkout.)

**Metadata by default; text is a second, separate opt-in.** The log
accumulates silently across every call, so a file that quietly becomes a
transcript of everything you have ever asked an LLM should be a thing you
switched on deliberately, not a side effect of wanting cost numbers. Lengths
alone already answer what this log exists for (cost, latency, verbosity), so
that stays the default:

    GLLM_CALL_LOG_TEXT=1     # record prompt, system and completion in full
    GLLM_CALL_LOG_TEXT=2000  # record them, capped at 2000 chars each
    GLLM_CALL_LOG_TEXT=off   # lengths only (the default)

Note `1` means ON, not "one character" — it is a truthy word like everywhere
else here. Pass a larger integer to mean a cap. A cap is worth setting if you
use `-f` attachments: an inlined file turns one call into a megabyte of log.

Whatever you capture lands in a plaintext file on disk. That is fine on a
personal machine and is the point of the feature; it is worth knowing before
enabling it on a shared one.

JSONL rather than one JSON array: appending to an array means rewriting the
whole file, which races between concurrent gllm processes and truncates the
history if one is interrupted mid-write. One object per line appends, survives
a kill, and pipes straight into `jq -s`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_OFF = {"", "0", "off", "false", "no"}
_ON = {"1", "on", "true", "yes"}
_TEXT_ENV = "GLLM_CALL_LOG_TEXT"
_UNCAPPED = 0


def default_path() -> Path:
    """`$XDG_STATE_HOME/gllm/calls.jsonl`, falling back to ~/.local/state."""
    base = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "gllm" / "calls.jsonl"


def log_path() -> Path | None:
    """Where to append, or None when logging is off.

    A truthy word means "the default path"; anything else is taken as a path,
    so `GLLM_CALL_LOG=./calls.jsonl` works without a second variable.
    """
    raw = os.environ.get("GLLM_CALL_LOG", "").strip()
    if raw.lower() in _OFF:
        return None
    if raw.lower() in _ON:
        return default_path()
    return Path(os.path.expandvars(raw)).expanduser()


def enabled() -> bool:
    """True when a call should be priced and logged. Checked before the record
    is built, so a disabled log costs nothing beyond one env lookup."""
    return log_path() is not None


def text_limit() -> int | None:
    """Chars of prompt/completion to record: None when text capture is off,
    `_UNCAPPED` (0) for the whole thing, else a positive cap.

    A malformed value warns rather than defaulting to off — someone who typed
    `GLLM_CALL_LOG_TEXT=ture` asked for text and should not silently get a log
    without any.
    """
    raw = os.environ.get(_TEXT_ENV, "").strip().lower()
    if raw in _OFF:
        return None
    if raw in _ON or raw == "full":
        return _UNCAPPED
    try:
        return max(int(raw), 1)
    except ValueError:
        print(
            f"gllm: {_TEXT_ENV}={raw!r} is neither on/off nor a number of "
            f"characters; prompt and completion text NOT logged.",
            file=sys.stderr,
        )
        return None


def text_fields(*, prompt=None, response=None, system=None) -> dict:
    """The text half of a record: `{}` unless text capture is on.

    Each field is capped independently, and a capped one is flagged so a short
    completion is never mistaken for a truncated record when you read the log
    back.
    """
    limit = text_limit()
    if limit is None:
        return {}
    out: dict = {}
    for key, value in (("prompt", prompt), ("system", system), ("response", response)):
        if value is None:
            continue
        text = str(value)
        if limit != _UNCAPPED and len(text) > limit:
            out[key] = text[:limit]
            out[f"{key}_truncated"] = True
        else:
            out[key] = text
    return out


def append(record: dict) -> None:
    """Append one record. Warns on failure; never raises.

    A log write must not turn a successful LLM call into a failed command —
    the answer is already on stdout by the time this runs. It still complains
    on stderr rather than swallowing the error, because a log that silently
    stopped recording is worse than one that never started.

    Written as a single `write()` to a file opened O_APPEND, which is what
    keeps concurrent gllm processes from interleaving each other's lines.
    """
    path = log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # default=str so an unexpected non-JSONable value degrades to its repr
        # instead of losing the whole record.
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        print(f"gllm: call log write failed ({path}): {exc}", file=sys.stderr)
