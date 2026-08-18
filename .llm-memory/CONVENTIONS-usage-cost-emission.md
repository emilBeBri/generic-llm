# `--usage`: machine-readable token/cost emission, normalised per provider

`gllm --usage` prints ONE JSON object to **stderr**, prefixed `gllm-usage `,
after the call. stdout stays the model text only (callers parse stdout as the
completion, so usage must never pollute it). It is the machine-readable sibling
of `-v/--verbose` (which prints a human `gllm: tokens in= out=` line); the two
compose.

Record shape (`cli.py`, end of `main()`):
```
gllm-usage {"provider":..,"model":..,"reasoning":<level|null>,
            "input_tokens":..,"output_tokens":..,
            "cache_read_tokens":..,"cache_write_tokens":..,"reasoning_tokens":..,
            "cost_usd":..,"priced_as":..,"price_source":..,"price_window":..,
            "max_tokens":..,"stop_reason":<provider's own word|null>,
            "truncated":<bool>,"schema":<bool>,"json":<bool>,
            "usage_raw":{<provider's own usage dict, verbatim>}}
```

`stop_reason` is the provider's own word, verbatim and un-normalised (`end_turn`, `stop`, `length`, `MAX_TOKENS`, `max_output_tokens`) — same rule as `usage_raw`. `truncated` is gllm's reading of it, so a consumer needn't know that Gemini says `MAX_TOKENS` where OpenAI chat says `length`. See [[ADR-output-budget-resolution]].

`max_tokens` is the budget **actually sent**, not the one asked for. It used to be `args.max_tokens` while nine adapters silently raised what went on the wire, so the field disagreed with the request it claimed to describe — see [[ADR-output-budget-resolution]].

## Extraction lives in the adapters (`gllm/usage.py`)

Per the "maximum precision at the source" rule: each adapter has its provider's
native usage object in scope at `Response`-build time, so a per-provider mapper
(`from_anthropic`, `from_openai_chat`, `from_openai_responses`, `from_gemini`,
`from_deepseek`) normalises it and the adapter splats the result:
`Response(..., raw=resp, **from_openai_chat(resp.usage))`. `Response` gained
`cache_read_tokens` / `cache_write_tokens` / `reasoning_tokens` / `usage_raw`
(additive, non-breaking — existing `input_tokens`/`output_tokens` stay).

Provider field map (the precision that's otherwise lost):
- **Anthropic**: `cache_read_input_tokens` → read, `cache_creation_input_tokens`
  → write (the pricier one). Thinking is folded into `output_tokens`, not broken out.
- **OpenAI chat / Z.AI-GLM**: `prompt_tokens_details.cached_tokens` → read,
  `completion_tokens_details.reasoning_tokens` → reasoning.
- **OpenAI responses**: `input_tokens_details.cached_tokens`,
  `output_tokens_details.reasoning_tokens`.
- **Gemini**: `cached_content_token_count` → read, `thoughts_token_count` →
  reasoning. NOTE thinking is billed ON TOP of `candidates_token_count`, so
  `output_tokens` here EXCLUDES it (unlike OpenAI/Anthropic).
- **DeepSeek**: `prompt_cache_hit_tokens` → read. Grok + azure_openai inherit
  the OpenAI mappers via subclassing.

## Two layers: normalised vs ground truth

The normalised fields are a lowest-common-denominator view whose semantics do
NOT fully agree across providers (the Gemini reasoning caveat above). For exact
per-model billing, use **`usage_raw`** — the provider's own numbers, untouched.

## Dollar cost IS baked in (`gllm/pricing.py`)

(Earlier this said cost belongs downstream — reversed on request 2026-06-28:
gllm owns the token counts, so it owns the $-conversion too.)

`--usage` adds `cost_usd`, `priced_as` (the entry matched), and `price_source`
(`override` | `book` | `none`) to the record. Since 2026-08-01 the source is
the **llm-price-tracker book** (editable path dependency on
`~/prog/prj/llm-price-tracker`): a committed, daily cross-checked store of
first-party vendor prices, keyed by vendor id, USD per 1M tokens. Its read
path is pure/offline — the old llm-prices.com runtime fetch (24h cache, 15s
worst-case offline stall) is DELETED, and `~/.cache/gllm/llm-prices-v1.json`
is inert. The CLI's match candidates are `[response.model, request.model,
request.wire_model]` — the book is vendor-id-keyed and `Response.model` is the
registry key on some adapters, the vendor's returned id on others.

Three separable pieces (the cost half is pure + unit-tested offline):
- `_book_entry(model, at)` — exact id, then the tracker's UNIQUE dot/dash-folded
  match (`claude-haiku-4-5` ↔ book page-slug `claude-haiku-4.5`); `_load_book`
  is the monkeypatch seam. Ambiguous fold → no match (null, never a wrong price).
  `at` picks the time-of-day rate — see below.
- `compute_cost(provider, entry, usage)` — **provider-aware**, because the token
  conventions differ: Anthropic `input_tokens` EXCLUDES cache (so don't subtract;
  writes bill at the book's published `cache_write` rate, ≈1.25× input only as
  the backstop for overrides); Gemini bills thoughts ON TOP of output (add
  `reasoning_tokens`); OpenAI-family/DeepSeek/GLM fold cache into prompt and
  reasoning into output (subtract cache_read, don't add reasoning).

## Time-of-day rates: `price_window`, stamped at DISPATCH

Some vendors bill by the clock — DeepSeek doubles input, output AND cache-hit
inside published UTC windows — and the book carries that structurally: the
scalar fields are OFF-PEAK, `peak` nests the peak rate, `peak_windows` selects
between them. `_book_entry(model, at)` resolves it with the book's own
`for_time`, so the hours stay a vendor fact and gllm never writes a clock time
down; a vendor moving its window needs a tracker `refresh --write` and nothing
here. Before this, `--usage` reported the off-peak figure at every hour of the
day — a silent 2x under-report for 7 hours out of 24.

`at` is stamped in `cli.main` **immediately before `provider.generate`**, not
after the response: vendors bill by when the request lands, and a call that
starts at 03:58 UTC and returns at 04:02 must not be repriced by having been
slow. It is `datetime.now(UTC)`, unambiguous by construction — but
`_book_entry` still normalises a tz-aware `at` to UTC, because the book's
`contains()` compares `.hour` WITHOUT converting, so 03:00+02:00 would read as
hour 3 when the instant is 01:00 UTC. That off-by-a-zone is a silent 2x and
nothing downstream can detect it.

`price_window` in the record names which rate was applied — `"peak"`,
`"off_peak"`, or `null` when the row has no windows (the common case). Without
it a consumer seeing `cost_usd` swing 2x between two identical calls would file
a gllm bug rather than read DeepSeek's policy.

**An override always reports `price_window: null`, and that is the tell.**
Overrides are consulted BEFORE the book and their schema is a flat
`{input, output, input_cached}` with no time dimension — so overriding a model
the book prices by the hour pins it to one rate permanently. That is allowed
(the overlay is the escape hatch) but it must be *visible*: `price_source:
"override"` beside a null window is the only signal that an hourly rate was
flattened. Teaching overrides their own windows was deliberately NOT built —
no one needs it, and a second window implementation is a second thing to drift.

## Local overrides fill book gaps (GLM) and fix mispricings

**GLM/Zhipu once had no tracker source** (the book alone gave `glm-5.2` → null
cost). Closed with a two-tier override (mirrors the schema/instruction layout,
[[CONVENTIONS-schemas-and-instructions]]):
- bundled `<repo>/data/prices.json` (version-controlled, syncs across machines)
- user overlay `~/.config/gllm/prices.json` (per-machine, **wins** per model)

`load_overrides()` merges them ({model_lower: {input, output, input_cached}},
USD/1M); `_`-prefixed keys are comments; an entry activates only with numeric
input AND output, so a null-valued stub never fabricates a $0 — fill it to
activate. `price_report()` consults **overrides BEFORE the book**, so they also
override a book mispricing; matched override → `price_source: "override"`.

### The book keeps absorbing rows — deleting the local copy is the maintenance

On 2026-08-18 the tracker gained zai and moonshot sources, and the 13 bundled
rows they now cover (kimi-k3 + twelve `glm-*`) were deleted. Every one was
**bit-identical** to the book, so nothing repriced: only the provenance
improved and `price_source` flipped `override` → `book`. Compute the set
mechanically (every key `llm_price_tracker.get_entry` resolves) rather than
hand-picking, and expect it again each time the tracker adds a vendor —
`test_bundled_price_overrides_do_not_shadow_the_book` is what tells you.

Watch for prose the book has no field for: `glm-4-32b-0414-128k` carried a
"No prompt caching" note, which cost nothing to lose here (with `input_cached`
null, cached tokens bill at the input rate — and a model with no caching
reports none), but the fact was moved into `_source` rather than dropped
silently. A `note` is never grounds to keep a row: `"override": true` means
"I disagree with the book's price", and using it to smuggle documentation is a
lie about the price. bebri-chat runs the identical layer and hit the identical
13 rows the same day.

Related: [[ADR-reasoning-effort-ladder]] (the `reasoning` level echoed in the
record), [[CONVENTIONS-zai-glm-adapter]] (GLM uses the chat mapper, and has no
feed price).
