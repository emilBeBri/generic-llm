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

Done: `deepseek`. Remaining: `grok`, `zai`, `kimi`, `openai_compat`, `openai`, `azure_openai`, `anthropic`, `azure_anthropic`, `gemini`. Anthropic additionally needs an SSE line reader in `_http` (it streams only to dodge socket timeouts on long thinking, then takes the final message). **`gemini` is the only real rewrite** — it is genuinely coupled to `types.GenerateContentConfig` / `types.ThinkingConfig` / `types.Part.from_bytes` / `resp.text`, and must be redone against the REST `:generateContent` endpoint. It is also the biggest single win at ~1.45 s.

When `openai`/`anthropic`/`gemini` convert, [[ADR-base-url-env-override]]'s claim that those three "need nothing, their SDKs read their own base-URL env vars" becomes **false** — they will all need `config.resolve_base_url`. Correct that note in the same commit.

Related: [[CONVENTIONS-multi-provider-routing]] (the adapter-shape rules this transport sits under), [[CONVENTIONS-porting-adapters-from-reference]] (the reshape-not-copy process each conversion follows).
