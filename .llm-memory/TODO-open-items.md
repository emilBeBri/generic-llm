# Project TODO — open items as of 2026-08-13

Standing work, roughly by value. Everything here is *known* work, not
speculation: each item names why it is unfinished, because in two cases the
reason was a wrong belief rather than a real blocker.

`#todo` `#moc`

# Verify the three adapters that no live call has ever touched

`azure_openai`, `azure_anthropic` and `anthropic` are unit-tested only. This box
has no Azure keys, and the jail withholds Anthropic credentials from the agent,
so all three went through the stdlib-transport rewrite
([[ADR-stdlib-http-transport]]) without a single real request.

The commands, and what each failure mode would mean, are written up at the end of
`AZURE-FOUNDRY-SMOKE-TEST.md`. The riskiest piece is `azure_anthropic`'s SSE
reassembly: a 200 with empty text means the event handling is wrong for Foundry,
and non-zero *input* tokens with zero *output* tokens means the `message_delta`
event was missed specifically.

# Backfill `ModelSpec.max_output` for the ~100 unsourced rows

27 of 130 rows carry a documented output ceiling (Anthropic 10 + their 8 Foundry
mirrors, 9 Gemini rows). The rest fall back to `DEFAULT_MAX_OUTPUT`, which is
correct-but-timid: those models can produce far more than 4096 tokens and gllm
will not ask them to.

**This stalled on a false belief and is not blocked.** An earlier note claimed
`~/source-docs/` was corrupt because reads showed labels replaced by a bare `n`.
The corpus is fine — that is a tool-output display artefact, and the same lines
read through `repr()` are intact. See the correction in
[[ADR-output-budget-resolution]]. Where an API reports the limit (Gemini's
`models.list()` returns `outputTokenLimit`), prefer probing to reading.

# Re-probe the 19 unverified zai context windows

`glm-4.5-flash` was registered at 128,000 and is really 131,072 — proved by
bracketing (a 65,017-token input plus `max_tokens=66,000` succeeded at 131,017;
66,100 failed at 131,117). Its output ceiling, 98,304, came out of a 400 that
named its own range. The other 19 zai rows still carry the unverified 128,000 and
are suspect for exactly the same reason. Two Gemini rows were wrong too, one
understated 5×.

# Decide what `pause_turn` should do

Anthropic's docs say to pass the response back as-is in a subsequent request to
let the model continue. gllm is one-shot and does not, so a paused turn currently
reads as a short answer with nothing flagged. Untested and unhandled — see
[[GOTCHA-stop-reasons-that-mean-no-answer]].

# Dropped deliberately, recorded so it is not re-proposed

An **input-size pre-flight refusal** (estimate the prompt's tokens, refuse before
the socket opens). Superseded by the clamp in
[[ADR-output-budget-resolution]]: providers already 400 loudly on input-alone
overflow, the clamp handles the case that actually bites, and a refusal is the one
use where an over-estimate causes *false rejections* instead of being harmless.

# Cross-repo: fixes filed against bebri-chat

Reading bebri-chat found bugs in both directions. Its `AGENT-TODO.md` now carries
four items from this session — the `max_tokens`/context clamp, two remaining fixes
from its own long-open truncation note, and re-probing its `context_window`
column. Those live in that repo's memory, not here.
