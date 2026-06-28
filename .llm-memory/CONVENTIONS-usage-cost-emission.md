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

## Dollar cost is deliberately NOT here

gllm emits the token/cache/reasoning *ingredients* and stops. Per-model prices
drift and are the consumer's policy, so $-conversion lives downstream (e.g.
book-agent's price table + SQLite cost log), not in gllm.

Related: [[ADR-reasoning-effort-ladder]] (the `reasoning` level echoed in the
record), [[CONVENTIONS-zai-glm-adapter]] (GLM uses the chat mapper).
