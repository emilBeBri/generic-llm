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
  extra_body) and is honoured **only by glm-5.2+** (`glm_supports_reasoning_effort`
  = `startswith('glm-5.2')`). GLM-5.2 accepts our ladder verbatim
  (low/medium/high/xhigh all valid), so `zai_effort` is identity-with-validation
  like `openai_effort` — the API collapses low/medium→high, xhigh→max itself.
- **No thinking at all** on `glm-ocr` / `glm-4-32b` (`glm_supports_thinking`).
  `supports_reasoning('zai', model)` returns that, so `--reasoning` on them fails
  loudly at the CLI gate (exit 2).
- **Vision is split into separate models** (`glm-5v*`, `glm-4.6v*`, `glm-4.5v`,
  `glm-ocr`; `is_glm_vision_model`). Text GLMs reject image content. The CLI
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
