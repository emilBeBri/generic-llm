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
            "max_tokens":..,"schema":<bool>,"json":<bool>,
            "usage_raw":{<provider's own usage dict, verbatim>}}
```

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

`--usage` adds `cost_usd`, `priced_as` (the feed entry matched), and
`price_source` to the record. Source is the **llm-prices.com** feed
(`https://www.llm-prices.com/current-v1.json`, Simon Willison's project — the
same one bebri-chat uses), fetched with stdlib urllib (no new dep) and cached to
`~/.cache/gllm/llm-prices-v1.json` for 24h with stale fallback. Prices are USD
per 1M tokens; `input_cached` may be null.

Three separable pieces (the matching/cost halves are pure + unit-tested offline):
- `load_prices()` — fetch + 24h cache + stale fallback.
- `match_price(model, prices)` — exact id → dot/dash-normalised (`gemini-3.1-...`
  ↔ feed `gemini-3-1-...`) → unique token-set (`claude-haiku-4-5` ↔ feed
  `claude-4.5-haiku`). Ambiguous token-set → no match (null, never a wrong price).
- `compute_cost(provider, entry, usage)` — **provider-aware**, because the token
  conventions differ: Anthropic `input_tokens` EXCLUDES cache (so don't subtract;
  writes ≈1.25× input); Gemini bills thoughts ON TOP of output (add
  `reasoning_tokens`); OpenAI-family/DeepSeek/GLM fold cache into prompt and
  reasoning into output (subtract cache_read, don't add reasoning).

## Local overrides fill feed gaps (GLM) and fix mispricings

**GLM/Zhipu is absent from the llm-prices feed** (so the feed alone gives
`glm-5.2` → null cost). Closed with a two-tier override (mirrors the
schema/instruction layout, [[CONVENTIONS-schemas-and-instructions]]):
- bundled `<repo>/data/prices.json` (version-controlled, syncs across machines)
- user overlay `~/.config/gllm/prices.json` (per-machine, **wins** per model)

`load_overrides()` merges them ({model_lower: {input, output, input_cached}},
USD/1M); `_`-prefixed keys are comments; an entry activates only with numeric
input AND output, so the shipped `glm-5.2` stub (null values) never fabricates a
$0 — fill it to activate. `price_report()` consults **overrides BEFORE the feed**,
so they also override a feed mispricing; matched override → `price_source:
"override"`. The bundled file ships glm-5.2/glm-5v-turbo stubs awaiting real
z.ai rates.

Related: [[ADR-reasoning-effort-ladder]] (the `reasoning` level echoed in the
record), [[CONVENTIONS-zai-glm-adapter]] (GLM uses the chat mapper, and has no
feed price).
