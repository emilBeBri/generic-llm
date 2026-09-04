# Z.AI / GLM adapter (ZaiProvider)

#architecture-decision-record #provider #zai #glm #multi-provider-routing

Zhipu's GLM family added to gllm. Wire protocol is **OpenAI Chat Completions**
at `base_url=https://api.z.ai/api/paas/v4/` (trailing slash matters for the SDK
URL join). Env key `ZAI_API_KEY`. Model ids are bare `glm-*`; the substring
router maps `'glm' in model -> 'zai'` (collides with nothing). Mirrors
bebri-chat's `ZaiAdapter` (commit 572995f), adapted to gllm's one-shot,
synchronous, no-tools shape — the first worked example of
[[CONVENTIONS-porting-adapters-from-reference]]. See also
[[CONVENTIONS-multi-provider-routing]].

## Standalone, NOT an OpenAIProvider subclass

Critical: GLM is DeepSeek-shaped (standalone `LLMProvider`, direct
`chat.completions.create`), NOT a `OpenAIProvider` subclass like Grok. Reason:
gllm's `use_responses_api()` returns True for any non-`gpt-4`/`gpt-3.5` slug, so
`glm-*` would be wrongly dispatched to the **Responses API**, which Z.AI does
not speak. Standalone forces Chat Completions.

## GLM-specific handling (the traps)

- **Thinking**: `extra_body={"thinking": {"type": "enabled"}}` when `--reasoning`
  is set. Omitting `--reasoning` = hands-off (provider default, which is
  thinking-ON for the forced-thinking 4.5+ line). gllm has no "thinking off"
  path — consistent with its hands-off-by-default stance.
- **reasoning_effort** is a recognised OpenAI SDK kwarg (top-level, NOT
  extra_body) and is honoured by the **glm-5.2 and glm-5.3 lines**
  (`glm_supports_reasoning_effort`, registry-backed via `thinking_dialect ==
  'zai_effort'`). They share the dialect, NOT the vocabulary: 5.2 is
  `high|max` and collapses low/medium→high, xhigh→max itself; 5.3 publishes
  `low|high|max` with `max` as the default, so `low` is a real rung there and
  `_GLM53_EFFORT` exists to stop `resolve_effort` promoting it to `high`.
- **The 5.3 line cannot stop reasoning.** `thinking.type` accepts only
  `enabled`, and `disabled` is a hard request failure. That costs gllm nothing
  today — it has no thinking-off path and omits the block without
  `--reasoning` — but it is why bebri-chat needed an explicit gate, and why a
  future `--no-reasoning` here would need one too.
- **No thinking at all** on `glm-ocr` / `glm-4-32b` (`glm_supports_thinking`).
  `supports_reasoning('zai', model)` returns that, so `--reasoning` on them fails
  loudly at the CLI gate (exit 2).
- **Vision is split into separate models** (`glm-5v*`, `glm-4.6v*`, `glm-4.5v`,
  `glm-ocr`; `is_glm_vision_model`) — with one exception, `glm-5.3-flash`,
  the first natively multimodal GLM, which takes images in the same model as
  code and tools. Text GLMs reject image content. The CLI
  image gate is provider-level (`supports_image('zai')` = True), so the
  per-model enforcement lives in the adapter: a non-vision model + image raises
  a loud RuntimeError naming the vision models. Images go as `image_url` base64
  data URIs.
- **No native PDF** (Z.AI `file_url` needs a hosted URL, not base64) → adapter
  and `supports_pdf('zai', …)`=False both reject PDFs.
- **Structured output is `json_object` only** — NO native json_schema. `zai` is
  deliberately OUT of `_STRICT_SCHEMA_PROVIDERS`, so `--schema` is refused
  (exit 2, same as DeepSeek); `--json` works. See [[CONVENTIONS-schemas-and-instructions]].

## Listing

Z.AI **does** support `client.models.list()` (returns 8 text ids: glm-4.5,
-4.5-air, -4.6, -4.7, -5, -5-turbo, -5.1, -5.2 — narrower than the full
marketing lineup, but it's the live truth). So `zai` is in `_LISTABLE_PROVIDERS`
and appears in `gllm --models`. See [[ADR-model-listing-live-probe]].

## Files

`adapters/zai.py` (new) · `routing.py` (glm→zai branch) · `cli.py` (_build_provider
+ _LISTABLE_PROVIDERS) · `adapters/_capabilities.py` (GLM vision/thinking/effort
helpers, supports_reasoning + _IMAGE_PROVIDERS) · `reasoning.py` (zai_effort).

## sources

External docs at `~/source-docs/zai-docs/` (thinking, struct_output,
chat_completion reference, vlm/glm_4_6v). bebri-chat reference adapter +
`.llm-memory/zai-glm-integration.md`.

## 2026-07-29: the family splits are registry data now

The GLM capability splits used to be prefix tuples in `_capabilities.py`
(`_GLM_VISION_PREFIXES`, `_GLM_NO_THINKING_PREFIXES`, and a
`startswith("glm-5.2")` check for `reasoning_effort`). They are now
`ModelCaps` on each GLM row in `models.py`:

| caps preset | rows | thinking |
|---|---|---|
| `_GLM53_VISION_EFFORT` | glm-5.3-flash | `thinking` + `reasoning_effort` (low/high/max), + images |
| `_GLM53_EFFORT` | glm-5.3 | same, text only |
| `_GLM_EFFORT` | glm-5.2 | `thinking` + `reasoning_effort` (high/max) |

(The binary-thinking and vision-only presets — `_GLM_THINK`, `_GLM_NO_THINK`,
`_GLM_VISION_THINK`, `_GLM_VISION_NO_THINK` — were deleted with the rows that
used them on 2026-09-04; see below.)

The consequence worth knowing: **`supports_image` now takes a model**, so the
vision split is refused by the CLI's native-or-fail gate before dispatch rather
than by an in-adapter raise. `is_glm_vision_model` / `glm_supports_thinking` /
`glm_supports_reasoning_effort` still exist and still work — they read the
registry row first and fall back to the old prefixes for an unregistered
`glm-*` name. See [[ADR-provider-model-axis]].

Z.AI's GLM models are also served third-party: `regolo:glm5.2-beta` is the same
family via [[ADR-provider-model-axis]]'s `openai_compat` adapter, `family='glm-5.2'`.

## 2026-09-04: the registry keeps four GLM rows, not twenty-one

Everything below GLM-5.2 was removed — the 4.x line, `glm-5`/`glm-5.1`/
`glm-5-turbo`, the free flash rows and the five vision-only models. Not
deprecations: Z.AI still serves all of them. They went because `--models`
listing 21 GLM ids for three anyone would pick is exactly the noise a curated
registry exists to prevent, and `glm-5.3-flash` at $0.075/$0.25 beats every
paid 4.x row while also reading images.

Two things deliberately did NOT change:

- **The `_legacy_*` prefix tuples stay.** A hand-typed `glm-4.6v` still routes
  to zai and still gets the right wire shape — it just has no registry row, so
  no context window, no `max_output`, and price via the book or nothing. That
  is the whole point of `_legacy_caps` being a fallback rather than a fixture,
  and `test_glm_vision_split_still_guessed_for_unregistered_ids` pins it.
- **The prices stay in the llm-price-tracker book.** The book prices a call
  made in July; deleting rows there would break that, and the next refresh
  would restore them anyway. `data/prices.json` now carries one GLM row, the
  regolo rental — the free-tier rows went with their models.

## 2026-08-07: two endpoints, one key — and the error names the wrong cause

Z.AI serves pay-as-you-go credit and **coding-plan subscriptions from different
base URLs**, and the same `ZAI_API_KEY` authenticates on both. So pointing a
coding-plan key at the credit endpoint does not fail as auth — it fails as:

```
429 code 1113: Insufficient balance or no resource package. Please recharge.
```

which reads as "you are out of money" and sends you to check your billing page,
not your base URL. Same key on `https://api.z.ai/api/coding/paas/v4/` answered
immediately.

`ZAI_BASE_URL` now overrides the default (`zai_base_url()` in `adapters/zai.py`),
because the endpoint is a property of the user's *plan*, which no amount of
model-registry data can tell you. It is env-var rather than a flag deliberately:
it is set once per machine alongside the key, not chosen per call.

Worth remembering as a class of bug, not a GLM fact: **a provider error that
names a plausible cause is still not evidence for it.** 1113 is generated by an
endpoint that has no idea a subscription exists elsewhere on the same account.
