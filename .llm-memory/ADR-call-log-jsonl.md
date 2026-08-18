# Call log: opt-in JSONL of every gllm call, in XDG state, metadata only
tags = #architecture-decision-record #observability #cost-estimation #privacy

`GLLM_CALL_LOG` makes gllm append one JSON object per call to a JSONL file.
Built 2026-08-19 because estimating cost and speed from rate cards was wrong by
3x, and because every model comparison so far has rested on synthetic
micro-benchmarks that keep producing confident wrong answers — see
[[TODO-open-items]]. Real usage is the only corrective.

Record: `ts` (UTC ISO), `elapsed_s`, `ok`, the whole `--usage` record
(provider, model, reasoning, token counts, `cost_usd`, `priced_as`,
`price_source`, `price_window`, `stop_reason`, `truncated`, `usage_raw`), plus
`prompt_chars` / `response_chars` / `attachments`. A failed call logs `ok:
false` with the error string and its elapsed time — a call that died after 30s
is exactly what the log is for.

## Four decisions

**Opt-in, not always-on.** Pricing a call loads the price book: ~150 ms of
pydantic that plain runs deliberately avoid (`pricing._load_book` is lazy for
this reason, and [[ADR-stdlib-http-transport]] fought to get per-call startup
down to ~0.12 s). Always-on logging would have quietly given that back on every
invocation. `calllog.enabled()` is checked BEFORE the record is built, so a
disabled log costs one env lookup — verified: zero pydantic imports on the
plain path.

**`$XDG_STATE_HOME/gllm/calls.jsonl`, not `data/`.** `data/` is bundled,
version-controlled content (prices.json, schemas, instructions) that ships with
the package; this is per-machine history that must never be committed. State,
not config either — it is not something you edit. Living outside the repo is
the guarantee: `git add` cannot reach it. `.gitignore` covers `*.jsonl` as a
backstop for pointing the variable at a path inside the checkout.

**JSONL, not one JSON array.** Appending to an array means rewriting the file,
which races between concurrent gllm processes and truncates the history if one
is interrupted mid-write. One object per line appends under `O_APPEND`,
survives a kill, and pipes into `jq -s`.

**Metadata by default; text behind a SECOND opt-in.** `GLLM_CALL_LOG_TEXT`
adds `prompt`, `system` and `response` to each record. Two knobs rather than
one because the log accumulates silently across every call: turning it into a
transcript of everything you have ever asked an LLM should be a decision, not
a side effect of wanting cost numbers. Lengths alone already answer cost,
latency and verbosity, so that stays the default.

Requested by the user 2026-08-19 for a personal machine where a plaintext
transcript is not a concern and reproducing a call from the log is the point.
The agent's privacy objection was raised once, overruled with a reason, and
implemented in full — recorded here so it is not re-litigated next session.

    GLLM_CALL_LOG_TEXT=1     # full text (1 is truthy, NOT "one character")
    GLLM_CALL_LOG_TEXT=2000  # capped at 2000 chars per field
    GLLM_CALL_LOG_TEXT=off   # lengths only (default)

Each field caps independently and a capped one gets `<field>_truncated: true`,
so a genuinely short completion is never misread as a cut-off record. A cap is
worth setting alongside `-f`: one inlined attachment turns a call into a
megabyte of log. A malformed value (`ture`) warns and logs no text rather than
silently defaulting to off — someone who set the variable asked for text.

A failed call records the prompt but has no `response` key. What was sent is
the interesting half of a failure.

## Interface

    GLLM_CALL_LOG=1                     # on, default path
    GLLM_CALL_LOG=/tmp/experiment.jsonl # on, chosen path (expands ~ and $VARS)
    GLLM_CALL_LOG=off                   # off; also unset, "", 0, false, no

A write failure warns on stderr and does not raise: the answer is already on
stdout by then, so a log problem must not turn a successful call into a failed
command — but a log that silently stopped recording is worse than one that
never started, so it stays audible. Same reasoning as
`_load_user_env_file`'s missing-key-file warning.

Related: [[CONVENTIONS-usage-cost-emission]] (the record shape this reuses, and
`price_window`), [[CONVENTIONS-one-shot-workload-no-cache]] (the verdict this
exists to re-derive from real data).
