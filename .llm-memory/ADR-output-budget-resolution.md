# ADR: the output budget is resolved once, in the CLI, and never silently overridden

`--max-tokens` defaults to `None`. `cli._resolve_max_tokens` turns that into the number every adapter sends, and adapters forward `request.max_tokens` verbatim — no adapter applies a floor of its own any more.

`#architecture-decision-record` `#convention` `#multi-provider`

## What was wrong

`--max-tokens` defaulted to a hardcoded 4096, which is far too small once reasoning is on, so **nine adapters each bolted on their own floor** — seven flat `max(request.max_tokens, 16000)`, plus Anthropic's budget-derived `min_max_tokens`. Consequences, all of them quiet:

- An explicit `--max-tokens 500` was discarded in nine places. With `-r xhigh` on Claude that meant 64000 went on the wire — a 128x override of a stated instruction.
- `--usage` emitted `request.max_tokens`, the number **asked for**, not the number **sent**. For a record whose stated job is exact cost accounting, that field was a lie. Verified before/after: `-r high` on deepseek reported `"max_tokens":4096` while 16000 went out; it now reports 16000.
- Nothing announced the override, unlike the effort-remap notices of [[ADR-reasoning-effort-ladder]].

The floors themselves were *correct*, which is why this is a restructuring and not a removal. Reasoning tokens are spent from the output budget — proven live on Gemini 2026-08-13: `max_output_tokens=120` with `thinking_budget=2048` returned `thoughts=115`, `candidates=1`, `finish_reason=MAX_TOKENS`, and printed `1` for `23*47`. A budget sized for a plain answer really does starve the answer.

## The rule: gllm's guess is overridable, the user's number is not

Three cases in `_resolve_max_tokens`, and the whole reason `--max-tokens` defaults to `None` is to tell the first two apart:

1. **No flag** → gllm picks: `ModelSpec.max_output` when the registry knows it, else `cli.DEFAULT_MAX_OUTPUT` (4096), raised to `reasoning.min_output_tokens` when `-r` is on. Silent, because the user expressed no preference.
2. **Explicit and adequate** → sent verbatim.
3. **Explicit but below the reasoning floor** → sent anyway, with a stderr warning (`-q`-silenced). A stated number is a decision.

The one exception is a minimum the **API enforces**: Anthropic's old `enabled`+`budget_tokens` interface requires `budget_tokens` strictly less than `max_tokens`, so a smaller value is a guaranteed 400. `reasoning.hard_min_output_tokens` detects only that case and the CLI **refuses** with the minimum that would work (`--max-tokens 500 -r high` on claude-opus-4-5 → "the API needs at least 32001", exit 2, no request sent). The adaptive dialect sends no budget, so its 64000 is gllm's headroom preference and only warns.

## `ModelSpec.max_output` — and why most rows are still `None`

A different axis from `context_window`, and on every provider tested it covers thinking **plus** answer. Populated for 27 rows: Anthropic 10 (128k/64k per platform.claude.com's "Max output" row) and their 8 Foundry `-dev` mirrors (same model artifact), plus 9 Gemini rows at 65536 from `models.list().output_token_limit`, probed live.

`None` means **not sourced**, never "unlimited" — a value guessed too high is a hard 400 and one guessed too low truncates, so an unsourced row falls back rather than inventing. Do not "complete" the column from vibes.

Two things blocked the rest, worth knowing before retrying:
- **`~/source-docs/` is unusable for this field.** The crawler mangles the literal string `max` — `**Output token limit** 65536` arrives as `**n** 65536`, `max_tokens=800` as `n_tokens=800`, `Max Output Tokens` as `n Tokens`. The *numbers* survive, the labels do not. Claude's overview table happened to escape it.
- **Ask the API instead where possible.** Gemini's `models.list()` returns `input_token_limit`/`output_token_limit` per model — authoritative, one call, no scraping. OpenAI-compatible `/models` responses carry no limits, which is why deepseek/zai/kimi/grok/groq/regolo/openai remain `None`.

## Truncation is detected, because a cap that IS hit was invisible

Sizing the budget correctly does not help when something still hits it, and until now nothing noticed: `finish_reason` was read nowhere in the tree, so a cut-off answer reached stdout looking exactly like a complete one.

`Response.stop_reason` carries the provider's own word **verbatim** (`end_turn`, `stop`, `length`, `MAX_TOKENS`, `max_output_tokens`) — un-normalised for the same reason `usage_raw` is. `Response.truncated` is a property matching it case-folded against `domain._TRUNCATION_REASONS` = {`max_tokens`, `max_output_tokens`, `length`}, which is what lets Gemini's enum name share an entry with Anthropic's string. An **unrecognised** reason is *not* reported as truncation: a missed warning beats a false one.

Three different extraction shapes, one per surface, and none of them interchangeable:

| surface | where the reason lives |
|---|---|
| Anthropic Messages | `msg.stop_reason` → `"max_tokens"` |
| OpenAI-compatible chat (deepseek, zai, kimi, compat, openai chat) | `choices[0].finish_reason` → `"length"` |
| OpenAI **Responses** | `status="incomplete"` + `incomplete_details.reason` → `"max_output_tokens"`. There is no `finish_reason` on this surface. |
| Gemini | `candidates[0].finish_reason` is an **enum** — use `.name` (`"MAX_TOKENS"`); `str()` yields `"FinishReason.MAX_TOKENS"` and matches nothing. |

The warning is deliberately **not** silenced by `-q`: that flag is `--quiet-effort`, scoped to the effort-remap notice, and truncation is a correctness signal rather than chatter. Verified live on deepseek both ways — `--max-tokens 12` warned with `stop_reason='length'`, a complete answer reported `"truncated":false` and printed nothing.

## The output budget shares the context window — measured, not assumed

`input_tokens + max_tokens <= context_window` is **enforced**, and the generous per-model default above is exactly what makes it bite. Established against GLM (`glm-4.5-flash`) 2026-08-13 by separating the causes:

| probe | input | max_tokens | sum | result |
|---|---|---|---|---|
| A | tiny | 120,000 | — | 400 `1210`: `max_tokens ... 限制数值范围[1,98304]` — the **output ceiling**, a different axis |
| B1 | 65,017 | 4,096 | 69,113 | ok — so that input alone is legal |
| B2 | 65,017 | 98,304 | 163,321 | 400 `1261 Prompt exceeds max length` |
| — | 65,017 | 66,000 | 131,017 | ok |
| — | 65,017 | 66,100 | 131,117 | 400 |

Both halves of B2 are legal on their own (A fixes the output ceiling at 98,304; B1 clears the input), so only the sum explains it. The last two rows bracket the true window at **131,072**.

Two traps in that data:
- **The provider's error misattributes the cause.** `Prompt exceeds max length` when the prompt was 65,017 of 131,072. Nobody debugging that would suspect `max_tokens`, which is the case for gllm doing the arithmetic itself.
- **The registry was wrong**, and a pre-flight check inherits whatever it says. `glm-4.5-flash` was registered at 128,000 — disproved outright, since sum 131,017 succeeded. `gemini-3-flash-preview` was registered at 200,000 against the live API's 1,048,576 (understated 5x), and `gemini-3.1-pro-preview` at 1,000,000 vs 1,048,576. All three corrected. **The other 19 zai rows still carry the unverified 128,000 and are therefore suspect.** Where a provider's `models.list()` reports limits (Gemini does, OpenAI-compatible hosts do not), probing beats hardcoding — the same argument as [[ADR-model-listing-live-probe]].

## `_clamp_to_context`: only lowers, and only gllm's own number

The default budget is shrunk to `context_window - estimated_input` when that is smaller. An explicit `--max-tokens` never reaches the clamp — same principle as everything else here.

`_CHARS_PER_TOKEN = 3.0`, deliberately pessimistic against the usual rule of thumb of 4: the B1 input measured 222,499 chars → 65,017 tokens, i.e. **3.42 chars/token**, so /4 would have *under*-counted by 17% and under-counting is the direction that 400s. Verified end to end: the exact B2 request that failed now succeeds on the default budget, clamped to 56,889 (sum 121,906). The estimate ran 14% high, which is the harmless direction.

**Attachments disable the clamp.** Image and PDF-page cost is a function of pixel dimensions and page count, not character length, so there is no honest estimate to make; the budget is left alone and the API stays the backstop. Do not "improve" this with an invented per-attachment constant.

When the clamp lands below the reasoning floor, it warns — the trace itself may not fit, and the answer will likely come back truncated (which `Response.truncated` now catches).

## Behaviour change to know about

Kimi's floor was **unconditional** (16000 even without `-r`), so a plain Kimi call now defaults to `DEFAULT_MAX_OUTPUT` like every other provider. Kimi publishes no fixed ceiling — k3's limit is `context - prompt_tokens` — so there is no honest `max_output` to put in the registry for it yet.

Related: [[ADR-reasoning-effort-ladder]] (where the effort and thinking budgets come from), [[CONVENTIONS-usage-cost-emission]] (the `max_tokens` field this made truthful), [[ADR-provider-model-axis]] (why per-model facts live in the registry).
