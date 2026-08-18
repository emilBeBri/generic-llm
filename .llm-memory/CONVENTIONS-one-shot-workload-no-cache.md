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

Standing verdict for gllm: **gpt-5.6-luna** as `DEFAULT_MODEL` — fastest
measured, ~20x cheaper than gemini-3.5-flash, strict `--schema` (DeepSeek has
none: gllm REFUSES `--schema` there rather than fake enforcement), full effort
ladder, largest context. deepseek-v4-flash is for off-peak batch work where
latency does not matter and nothing needs a schema: ~1.8x cheaper than luna
off-peak, but MORE expensive at peak ($0.44 vs $0.40 per 1k of the measured
call) and slower in both cases.

**Untested:** DeepSeek's latency *during* its peak window — every sample above
is off-peak. The hypothesis that it is slower when it is dearest is unmeasured;
rerun inside 06:00-10:00 UTC and the `WINDOW` column will say `peak` so the
sample is self-labelling. No quality benchmark backs any of this — price,
latency and stated capability only.
