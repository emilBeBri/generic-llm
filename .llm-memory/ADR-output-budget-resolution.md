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

## Behaviour change to know about

Kimi's floor was **unconditional** (16000 even without `-r`), so a plain Kimi call now defaults to `DEFAULT_MAX_OUTPUT` like every other provider. Kimi publishes no fixed ceiling — k3's limit is `context - prompt_tokens` — so there is no honest `max_output` to put in the registry for it yet.

Related: [[ADR-reasoning-effort-ladder]] (where the effort and thinking budgets come from), [[CONVENTIONS-usage-cost-emission]] (the `max_tokens` field this made truthful), [[ADR-provider-model-axis]] (why per-model facts live in the registry).
