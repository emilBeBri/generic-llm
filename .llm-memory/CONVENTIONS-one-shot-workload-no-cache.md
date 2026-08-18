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

# Measured 2026-08-18 ($ per 1,000 calls, no cache, same work)

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

Standing verdict for gllm: **gpt-5.6-luna** as `DEFAULT_MODEL` — cheapest or
within ~1.3x of cheapest on every one-shot job, the full effort ladder
including `none`/`low`, largest context of the three, and immune to the peak
clock. deepseek-v4-flash wins off-peak on small prompts and is still 4-6x
cheaper than gemini-3.5-flash even at peak. Price and stated capability only
— no quality benchmark backs this.
