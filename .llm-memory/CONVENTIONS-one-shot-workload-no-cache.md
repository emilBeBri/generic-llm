# gllm's workload is one-off shots: cache rates are NOISE when choosing a model
tags = #convention #cost-estimation #user-preference #model-selection

Stated by the user 2026-08-18 and true structurally: gllm calls are **one-off
shots**, not conversations. `rg cache_control src/gllm/adapters/` returns
nothing — gllm never requests caching (Anthropic's explicit opt-in block is
absent), and the automatic prefix caching OpenAI and DeepSeek do needs a
repeated prefix that a one-shot invocation does not have. `pricing.py` reads
`cache_read_tokens` back off the usage object because providers report it, not
because gllm engineers a hit.

**So a model's cache-hit rate must not enter a gllm cost comparison.** Every
call bills full input + full output. DeepSeek's cache-hit price is `$0.007/M`
— roughly 20x below Gemini's — and it is worth exactly nothing here. Quoting
it as an advantage produced a recommendation the user had to correct.

Corollary: **the peak/off-peak swing hits gllm at full weight.** In a
cache-heavy client a doubled input rate is blunted by the discounted majority
of tokens; here there is nothing to blunt it. See
[[CONVENTIONS-usage-cost-emission]] for the `price_window` field that makes
the swing visible.

# The same price book gives gllm and bebri-chat OPPOSITE answers

Worth stating because both repos read the same llm-price-tracker book and it
is tempting to reuse a verdict:

- **gllm** — one-shot, no cache, cost is `input x rate + output x rate` every
  time. Output rate dominates, since a reasoning model emits far more output
  than a short prompt sends input.
- **bebri-chat** — long conversations that re-send a growing prefix, so cache
  reads are the *majority* of billed input. A cheap cache-hit rate is the
  single biggest lever there, and the model rankings reorder accordingly.

A model recommendation is therefore not portable between the two repos. Derive
it per workload, from the book, at the moment it is asked.

# MODELLED 2026-08-18 ($ per 1,000 calls, no cache, assumed token counts)

| job | gemini-3.5-flash | ds-v4-flash off-peak | ds-v4-flash peak | gpt-5.6-luna |
|---|---|---|---|---|
| 2k in, 500 out, no thinking | $7.50 | $0.77 | $1.54 | $1.00 |
| 2k in, 500 visible + 2k thought | $25.50 | $2.09 | $4.18 | $3.40 |
| 100k in, 1k visible + 2k thought | $177.00 | $23.98 | $47.96 | $23.60 |

Two things that decide it, neither visible in a rate table:

* **Gemini bills thinking ON TOP of output** (thoughts are a separate count),
  where OpenAI and DeepSeek fold reasoning into `output_tokens`. At $9/M
  output that makes `-r low` on gemini-3.5-flash cost triple its own no-think
  price. This is the same asymmetry `compute_cost` encodes per provider.
* **deepseek-v4-flash has no `low` rung** — `native_efforts` is
  `('high', 'max')` (see [[ADR-reasoning-effort-ladder]]). Its cheapest
  thinking is expensive thinking, so a like-for-like "low effort" comparison
  flatters it by holding thought-tokens equal when DeepSeek would emit more.

That table is ARITHMETIC ON RATE CARDS, and it understated the real gap by
3x. See the measured section below — keep the two apart.

# MEASURED 2026-08-18, live calls via `scripts/latency-bench.zsh`

Off-peak, `-r low`, identical prompt ("explain a B-tree...~200 words"), one
call each. Latency is the median of 3, wall clock from send to full answer.

| model | short | long | tok/s | in | visible out | thought | cost |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | **0.90s** | **4.15s** | **68.9** | 29 | 330 | 31 | **$0.000402** |
| deepseek-v4-flash | 1.24s | 5.98s | 51.3 | 106 | 301 | 44 | **$0.000222** |
| gemini-3.5-flash | 2.38s | 5.86s | 41.8 | 27 | 264 | **631** | **$0.008096** |

**The rate card cannot tell you how much a model THINKS, and that dominates.**
gemini-3.5-flash at `low` burned 631 thought tokens where luna burned 31 — 20x
the thinking for a slightly shorter answer — and Gemini bills thoughts on top
of output at $9/M, so the thinking cost more than the answer ($0.0057 of
$0.0081). Actual gap: **20x luna, 36x deepseek**, against the 7.5x the rates
implied. Any future model comparison here must measure thought tokens, not
reason from $/M alone.

Luna also won every latency cell, including against DeepSeek. gemini-3.5-flash
was the SLOWEST of the three as well as the dearest, and the least predictable
(short-prompt median 2.38s, worst case 4.66s).

## `-r low` is NOT Gemini's floor — omitting `-r` entirely is

Measured on gemini-3.5-flash-lite, same prompt: `-r low` produced **627-656
thought tokens**; **no `-r` flag produced ZERO**, and cost fell 3.5x ($2.29 ->
$0.65 per 1k). Gemini's `native_efforts` has no `none` rung, so nothing in the
effort ladder reaches this — only the absence of the flag does. Passing `-r
low` "to be cheap" on a Gemini model is a 3.5x cliff in the wrong direction.

luna is not the same shape: with no flag it still emits 40-106 thought tokens,
and costs about what `-r low` costs. Its ladder does have `none`.

And the thinking tax is a Gemini-FAMILY trait, not a flash quirk: flash-lite
thought 627-656 where flash thought 878-885. lite is cheaper because of its
rate card, NOT because it thinks less.

## MEASURED 2026-08-18 evening, no `-r` flag (the one-shot configuration)

Four samples each, same B-tree prompt. `$` is per 1,000 calls.

| model | visible out | thought | $ /1k | short | long | tok/s |
|---|---|---|---|---|---|---|
| gemini-3.5-flash-lite | 248-273 | **0** | $0.63-0.69 | **0.83s** | **1.93s** | **132.8** |
| gpt-5.6-luna | 311-392 | 40-106 | $0.38-0.48 | 1.36s | 6.38s* | 103.1 |
| deepseek-v4-flash | 369-419 | 105-129 | $0.27-0.30 | 1.41s | 5.08s | 63.4 |

*luna's 6.38s came from a run that emitted 658 tokens — a verbosity outlier
against its typical 311-392. Normalised per token it is ~0.97 s/100 tok against
flash-lite's 0.75, so call it ~1.3x slower per token, not 3.3x.

**flash-lite is the speed winner and the most concise** — it alone hit the
"around 200 words" brief; luna and DeepSeek overshot by 25-60%. It is also
~1.5x luna's cost and ~2.4x DeepSeek's, which matches the rate-card floor
predicted before measuring (1.7x at zero thinking; 1.5x measured, the gap being
its concision).

**Perspective the ranking hides: these are fractions of a cent per call.**
$0.27 vs $0.43 vs $0.66 per *thousand* calls. Unless the volume runs to tens of
thousands, price should not decide this — latency and capability should. Price
only became decisive against gemini-3.5-flash, which is a genuine 15-25x
outlier, not against any of these three.

## flash-lite's cheap/fast configuration is the one that CANNOT do arithmetic

The finding that reverses the ranking above. A multi-step word problem (buy 17
pens at 3-for-$2 plus $0.80 per leftover; correct answer $11.60), 4 runs each:

| model / config | result |
|---|---|
| gemini-3.5-flash-lite, **no `-r`** | **0/4** — $12.20, $11.80, $12.00, $11.80 |
| gemini-3.5-flash-lite, `-r low` | **4/4** correct (~255 thought tokens) |
| gpt-5.6-luna, no `-r` | **4/4** correct |
| deepseek-v4-flash, no `-r` | **4/4** correct |

Note it is not merely wrong, it is *inconsistently* wrong — four runs, three
different answers. That is the signature of no deliberation at all rather than
a stable mistake, and it means a retry does not save you.

So flash-lite's entire advantage — fastest, most concise, 0 thought tokens —
exists only in the configuration that fails multi-step reasoning. Buying the
fix costs the 3.5x thinking cliff ($0.65 -> $2.29 per 1k), which lands it at
~5x luna, and luna gets the same answer right for free with no flag at all.

**Easy traps do not discriminate.** Bat-and-ball ($0.05), counting r's in
"strawberry raspberry" (6), and a transitive height ordering were all answered
correctly by all three including flash-lite with zero thinking. A quality probe
that uses only classic one-step traps will conclude, wrongly, that these models
are equivalent. Reach for something needing two or three chained operations.

Scope: n=4 on one problem type, deliberately superficial. It is enough to
disqualify a default, not enough to rank luna against deepseek on quality —
neither was ever wrong here.

Standing verdict for gllm: **gpt-5.6-luna** as `DEFAULT_MODEL` — right for
free where flash-lite's cheap configuration is not (see the arithmetic section), ~20x cheaper than gemini-3.5-flash, strict `--schema` (DeepSeek has
none: gllm REFUSES `--schema` there rather than fake enforcement), full effort
ladder, largest context. deepseek-v4-flash is for off-peak batch work where
latency does not matter and nothing needs a schema: ~1.8x cheaper than luna
off-peak, but MORE expensive at peak ($0.44 vs $0.40 per 1k of the measured
call) and slower in both cases.

## Omitting `-r` is NOT "no reasoning" — it is the provider's default

gllm sends no effort parameter when `-r` is absent (`if reasoning_on:` guards
the kwarg), so the SERVER default applies, and that differs per provider.
OpenAI's reasoning guide is explicit: *"If you omit `reasoning.effort`, GPT-5.6
defaults to `medium` in both modes"*, and defaults are "model-dependent rather
than universal". So bare `gllm -m gpt-5.6-luna` runs at **medium**, while bare
`gllm -m gemini-3.5-flash-lite` runs at **zero** thought tokens (measured).
Same absent flag, opposite meaning.

Measured, pens problem, per 1k calls:

| luna config | thought | $/1k | correct |
|---|---|---|---|
| no flag (= medium) | 45 | $0.08 | yes |
| `-r low` | 33 | $0.06 | yes |
| `-r none --native-effort` | **0** | **$0.02** | **4/4 yes** |

**luna is still right with reasoning switched fully off**, on the exact problem
flash-lite failed 0/4 without thinking. That is the cleanest statement of the
gap between them: it is not that one thinks and the other does not.

Why an effort probe looks flat on luna: the same guide says the models "reason
adaptively across reasoning [efforts], using fewer tokens for simpler tasks".
Sweeping low/medium/high produced 20-50 thought tokens at every rung — the knob
is a ceiling, not a quota, so you cannot infer the active level from token
counts.

`DEFAULT_EFFORT` gotchas, both verified:

* gllm's ladder is `LEVELS = (low, medium, high, xhigh)` — **no `none` rung**.
  `DEFAULT_EFFORT="none"` fails every call with exit 2 (loudly — house style
  working). Use `""` or omit the variable.
* `none` is reachable only as `-r none --native-effort`, and `--native-effort`
  **refuses to inherit `$DEFAULT_EFFORT` by design**, so zero-reasoning cannot
  be set as an ambient default at all. It is a per-call flag or nothing.
* `--native-effort` validates against the registry: `-r minimal
  --native-effort` on luna is rejected with the model's real vocabulary
  (`none, low, medium, high, xhigh, max`) — GPT-5.6 has no `minimal` rung even
  though older GPT-5 docs describe one.

**Untested:** DeepSeek's latency *during* its peak window — every sample above
is off-peak. The hypothesis that it is slower when it is dearest is unmeasured;
rerun inside 06:00-10:00 UTC and the `WINDOW` column will say `peak` so the
sample is self-labelling. No quality benchmark backs any of this — price,
latency and stated capability only.
