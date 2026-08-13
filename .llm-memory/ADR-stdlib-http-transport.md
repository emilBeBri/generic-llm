# ADR: the vendor SDKs are being replaced by a stdlib transport (`gllm._http`)

`openai`, `anthropic` and `google-genai` are being removed as dependencies. Adapters POST their request body themselves through `gllm._http` (stdlib `http.client` + `json`). The reason is startup latency, which for a one-shot CLI is a fixed tax on every invocation.

`#architecture-decision-record` `#performance` `#transport` `#multi-provider`

## The measurement that decided it

Profiled 2026-08-12 on the uv-tool install, warm, three runs each:

| what | cost |
|---|---|
| bare interpreter | 26 ms |
| `import gllm.cli` (own code, all stdlib) | 52 ms |
| `import openai` | 0.69–0.84 s |
| `import anthropic` | 0.89–0.95 s |
| `from google import genai` | 1.02–1.46 s |
| `ssl, http.client, json, base64, argparse` | 0.06–0.08 s (total, incl. interpreter) |

`import anthropic` builds **462 pydantic `BaseModel` subclasses** and loads **1247 modules** (bare interpreter: 36). `anthropic.types` alone is 333 ms, `anthropic.lib.tools` 254 ms — a tool-runner gllm never calls.

**Lazy-importing cannot fix this.** Pydantic compiles a validator/serialiser when the *class body executes*, so the cost lands at import, not instantiation. `cli.py` already defers adapter imports (which is why `--help` is 87 ms), and a real call still pays in full. The only fix is not importing it.

The A/B after converting one adapter, same tree, same interpreter: `deepseek` (converted) **105–124 ms**, `anthropic` (still on the SDK) **891–1021 ms**.

## Why the SDKs were free to remove

The entire SDK surface across all ten adapters was five operations: the client constructor, `chat.completions.create(**kwargs)`, `responses.create(**kwargs)`, `messages.create/stream(**kwargs)`, and `models.list()`. Note the `**kwargs` — every adapter already hand-built a plain dict. The SDK validated that dict into pydantic models and serialised them back to the same JSON. Nothing was gained on the request side; the response side reads two or three fields.

## `http.client`, not `urllib.request`

`urllib.request` costs ~49 ms cumulative against `http.client`'s ~25 ms; the difference is opener/handler machinery for redirects, proxies, auth and cookies that an API call never uses. `http.client` also exposes status, headers and the raw socket read — needed for retry decisions, `Retry-After`, and the SSE reader the Anthropic adapter will want.

## The lever that made the migration cheap: `wrap` / `Obj`

`_http.post_json` returns decoded JSON verbatim; `_http.wrap` puts a lazy attribute-access view over it, so `resp.choices[0].message.content` and `getattr(usage, "cached_tokens", 0)` keep working against dicts. **`gllm.usage` therefore needed zero changes**, and each adapter's diff is one line at the call site.

`Obj.to_dict()` is deliberately named to be what `usage._to_plain` finds after `model_dump` misses. This makes `usage_raw` *more* faithful than before: the provider's own bytes rather than a pydantic re-serialisation, and nested detail dicts (`completion_tokens_details`) survive — the old `dir()`-scraping fallback dropped them because they aren't scalars. See [[CONVENTIONS-usage-cost-emission]].

A missing key raises `AttributeError` listing the available keys, so a changed response shape names itself.

## Traps found while converting

- **`extra_body` was never a wire concept.** It existed to get non-OpenAI params past the SDK's kwarg validation, and merged into the same top-level request object. Porting an adapter means hoisting its contents to the top level (`kwargs["thinking"] = {...}`), not preserving the key. Silent wrong-request otherwise.
- **What we now own:** retry/backoff with `Retry-After` (integer seconds only — an HTTP-date falls through to backoff rather than dragging in `email.utils`), per-provider auth headers, and refusing redirects. A 3xx is a misconfigured base URL and says so.
- **`http://` is supported on purpose** — the loopback key broker of [[ADR-base-url-env-override]] serves plain HTTP. It is not a TLS fallback.

## Live verification (2026-08-13)

Verified against the real API, through the jail's loopback key broker:

- `--models deepseek` → both v4 rows. The `get_json` path.
- plain call → text back, `tokens in=89 out=19`. The `post_json` path plus the usage mapper reading through `Obj`.
- `-r high --usage` → `reasoning_tokens: 6`. **This is the proof the `extra_body` hoist is right**: had top-level `thinking`/`reasoning_effort` been the wrong wire shape, DeepSeek would have 400'd or silently ignored them and reasoning_tokens would be 0.
- `usage_raw` came back with `completion_tokens_details`, `prompt_tokens_details` and `prompt_cache_hit_tokens`/`_miss` all intact — the nested dicts the old SDK `dir()`-scraping fallback dropped.
- `--usage` degrades cleanly with no `llm_price_tracker` installed: `cost_usd: null, price_source: "none"`. Only its unit tests fail on the missing dep, not the runtime path.

**Live provider testing IS available inside claude-jail** — the broker exports working session tokens plus `GLLM_BASE_URL_<TAG>` (see [[ADR-base-url-env-override]], which already documented this). Commit `ae5351f` claimed "this jail holds no provider keys" and skipped live verification on that basis; the claim was never checked with `env` and was false. Do not repeat it.

## Migration status and the note this will falsify

**Complete. All ten adapters, and `pyproject.toml` now declares no vendor SDK at all** — only `llm-price-tracker`. Measured per-adapter startup afterwards: 0.111–0.124s, against 0.89s (anthropic), 0.88s (openai) and 1.02–1.46s (gemini) before.

## Two hazards when a provider's SDK renamed things

Gemini was the only conversion where the SDK did more than transport the call, and it failed in **two different ways that need different defences**:

1. **Case.** The wire is camelCase (`usageMetadata`, `promptTokenCount`, `finishReason`); the SDK exposed snake_case; `gllm.usage.from_gemini` reads snake_case. `_snake_keys` rewrites response keys once, so the usage mapper, truncation detection and their tests need no Gemini special case.
2. **Genuinely different names, which no transform can bridge.** The SDK's `supported_actions` is the wire's **`supportedGenerationMethods`** — not a case variant. Reading the SDK's name returned an **empty catalog rather than an error**, so `gllm --models gemini` silently printed nothing. Only a live call caught it; no unit test written against the same wrong assumption ever would.

Also non-obvious on Gemini: `systemInstruction` is a **top-level** field (400s inside `generationConfig`), and the API version must stay **out** of the base URL — the SDK appended `/v1beta` itself, so every override in the wild (the jail broker's `GOOGLE_GEMINI_BASE_URL` included) points at the host and 404s if the base already contains it. That one also only showed up live.

## Anthropic: streaming is not optional

Anthropic documents a **10-minute ceiling on non-streaming Messages requests** (504 `timeout_error`, whose own remedy text says to use the streaming API), and gllm now defaults Claude to a 128k output budget — well capable of reaching it. So `_http.post_sse` exists, and the reasoning path streams and rebuilds the message via `final_message_from_events`.

The reassembly is where a bug hides quietly: each event contributes something no other repeats. `message_start` is the only source of `input_tokens` and the cache counters; `message_delta` is the only source of `stop_reason` and the final `output_tokens`. Miss the latter and usage reads **zero output tokens** while the text still looks right. An `error` event can also arrive mid-stream after a 200, so a success status is not the end of error handling.

`post_sse` is deliberately **not retried** — a stream can fail half-consumed, and replaying it would duplicate content or silently drop the first half.

## What is NOT verified live

`anthropic` and `azure_anthropic` are unit-tested only: the jail withholds Anthropic credentials from this agent and no Azure keys exist on this machine. `azure_openai` likewise. The live checks a work-box run should perform — including the SSE reassembly and the Azure URL shapes — are written up at the end of `AZURE-FOUNDRY-SMOKE-TEST.md`.

**`openai` leaving took `grok` and `azure_openai` with it** — both subclass `OpenAIProvider` and inherit its `generate`, so they were never separately convertible. The "OpenAI-compatible chat" adapters are independent of each other; the Responses-API ones are one unit. With that batch done, **`openai` is out of `pyproject.toml`**. (`.venv` still has it installed until someone syncs, which is fine and better than running `uv sync` in the jail — see [[GOTCHA-jail-uv-venv-artifactory]].)

## The Responses API needed something rebuilt, not just re-posted

Every other conversion was a call-site swap. `openai.py` was not: **`resp.output_text` is an SDK convenience, not a wire field.** The raw response is a list of typed items (`reasoning`, `message`, `web_search_call`, …); only `message` items carry `content`, and within that only `output_text` parts are answer text. `_output_text` reassembles it.

While rebuilding it, one deliberate deviation from SDK parity: a `refusal` content part with no accompanying text now **raises**. The SDK's `output_text` would have been `""`, so gllm would have exited 0 having printed a blank line — indistinguishable from an empty answer, and the same class of silent-wrong-output as the truncation gap in [[ADR-output-budget-resolution]].

## `OPENAI_BASE_URL` had to be wired, and this falsifies an older note

[[ADR-base-url-env-override]] said `openai`/`anthropic`/`gemini` "need nothing here: their SDKs already read their own base-URL env vars". True until the SDK left. `OpenAIProvider` now calls `resolve_base_url("openai", …, legacy_env="OPENAI_BASE_URL")`, because **the jail's key broker sets exactly that variable** — dropping the SDK without wiring it would have broken every jailed OpenAI call while looking like an auth problem. Corrected in that note; the same correction is still pending for `anthropic` and `gemini` when they convert.

## Verifying a hoisted key needs a DIFFERENTIAL, not a successful call

The `extra_body` hoists are the risky part of every conversion, and a single successful call cannot verify one: a silently-dropped key looks exactly like an honoured one. DeepSeek happened to give a positive signal (`reasoning_tokens: 6`); on Z.AI that signal is **per-tier** — `glm-4.5-flash` reports 0 even with thinking on, while `glm-4.7` and `glm-5.2` report real values (315 and 95 on the same prompt). Do not read one GLM model's 0 as a provider-wide fact, as this note first did.

What works: send the same request twice through `_http`, differing **only** in the hoisted key, and diff the replies. On `glm-4.5-flash` 2026-08-13:

| `thinking.type` | `reasoning_content` | completion_tokens | answer to 23*47 |
|---|---|---|---|
| `enabled` | 847 chars | 300 (hit the cap) | `''` |
| `disabled` | 0 chars | 3 | `1081` |

So top-level `thinking` **is** honoured by Z.AI. Use this shape for the remaining conversions rather than trusting a 200. Confirmed again on the tiers that report reasoning tokens: `glm-5.2` and `glm-4.7` at `-r high` returned 95 and 315 reasoning tokens with the correct answer.

What this method does **not** settle is whether an *effort value* is honoured. `glm-5.2` at `-r low` (→`high`) vs `-r xhigh` (→`max`) gave 377 vs 411 reasoning tokens — a 9% gap on one sample each, indistinguishable from sampling noise. A binary key flips a behaviour you can see; a graded one needs repeated sampling per rung, and an easy prompt may not separate the rungs at all. `reasoning_effort` **gating** (which models receive the parameter) is unit-tested instead; whether `max` thinks harder than `high` on GLM is unverified.

Incidental but useful: the `enabled` row is the output-starvation of [[ADR-output-budget-resolution]] reproduced on a third provider — 300 tokens of budget, all of it spent on the trace, **empty** answer.

## GOTCHA: `Retry-After` can stall a one-shot CLI for three minutes in silence

`_sleep_before_retry` honours `Retry-After` up to 60s, and `DEFAULT_RETRIES` is 3 — so a rate-limited call can sleep ~180s while printing nothing. Found the hard way: the first attempt at the probe above was killed at a 2-minute timeout, and the API was never slow, my own backoff was. Z.AI answers `429 code 1302` (rate limit) distinctly from `429 code 1113` (insufficient balance / wrong endpoint), and 1302 is what triggered it.

Unfixed as of this note. The shape of a fix: cap total retry wall time, and say something on stderr before the first sleep — silence is the actual defect, not the waiting. Pass `max_retries=0` when probing so the backoff cannot eat the budget before an error is visible.

Per-adapter gotchas found while converting the chat batch:
- **`zai`**: `ZAI_DEFAULT_BASE_URL` ends in `/`, so the join needs `rstrip("/")` or you POST to `//chat/completions`.
- **`kimi`**: k2.6's binary thinking block lived in `_reasoning_kwargs` as `{"extra_body": {"thinking": ...}}` — the hoist has to happen inside that helper, not at the call site.
- **`openai_compat`**: `ProviderSpec.extra_body` becomes `kwargs.update(extra_body)`. Regolo's `disable_fallbacks` MUST survive that or the host may silently answer with a different model, which makes model identity and cost accounting a lie. It now has a test.
- **`ProviderSpec.image_url_format_field` is currently unreachable through `generate`**: no registered regolo row has `supports_vision`, so `_user_content` refuses every one of them first. The flag is live config awaiting a model row. Tested at the unit instead. Anthropic additionally needs an SSE line reader in `_http` (it streams only to dodge socket timeouts on long thinking, then takes the final message). **`gemini` is the only real rewrite** — it is genuinely coupled to `types.GenerateContentConfig` / `types.ThinkingConfig` / `types.Part.from_bytes` / `resp.text`, and must be redone against the REST `:generateContent` endpoint. It is also the biggest single win at ~1.45 s.

When `openai`/`anthropic`/`gemini` convert, [[ADR-base-url-env-override]]'s claim that those three "need nothing, their SDKs read their own base-URL env vars" becomes **false** — they will all need `config.resolve_base_url`. Correct that note in the same commit.

Related: [[CONVENTIONS-multi-provider-routing]] (the adapter-shape rules this transport sits under), [[CONVENTIONS-porting-adapters-from-reference]] (the reshape-not-copy process each conversion follows).
