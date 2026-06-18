# Porting a provider adapter from bebri-chat (the reference)

#architecture-decision-record #convention #provider #process

bebri-chat is the **reference implementation** of LLM adapters. When gllm gains
a new provider, we port from bebri-chat's adapter — but the two are different
*hosts*: bebri-chat is a multi-turn, async, stateful, tool-using chat agent;
gllm is a one-shot, synchronous, stdin→stdout pipe. So a port is a **reshape,
not a copy**. The Z.AI/GLM port (2026-06-18) established the pattern below;
[[CONVENTIONS-zai-glm-adapter]] is the first worked example.

## Suggested default reshape

**Mirror faithfully — the provider's protocol facts:**
- base_url, env-key name, wire format (Chat Completions vs Responses), and
  whether it's standalone or an `OpenAIProvider` subclass (check what gllm's
  `use_responses_api()` would do to the model ids — wrong dispatch is the trap
  that forced GLM to be standalone).
- per-model capability splits (vision-only models, no-thinking models,
  effort-gated models) — same prefix sets as the reference.
- structured-output mechanism (`json_object` vs native `json_schema`) and the
  attachment-encoding shape.

**Drop — no surface in a one-shot CLI:**
- function calling / tools / `tool_choice`, native web search.
- `reasoning_content` round-tripping (single-turn → nothing to echo forward).
- streaming partials, token/cache meters.
- the hardcoded model registry + prices. gllm has **no** registry; discover live
  via `gllm --models` (see [[ADR-model-listing-live-probe]]). Porting the
  registry reintroduces the stale-catalog disease that command exists to kill.
- the thinking-disabled toggle (gllm has no off-switch; omitting `--reasoning`
  is hands-off → provider default).

**Deviate to gllm's house rules — deliberate, not bugs:**
- **Fail-loud, not warn-and-skip.** bebri-chat keeps the conversation alive
  (warn + skip an unsupported attachment, inject a placeholder). gllm fails fast
  (exit 2 / RuntimeError). Native-or-fail, no silent fallback.
- **Trust the API/docs over client-side guessing.** e.g. GLM effort: bebri-chat
  pre-collapses low/medium→high client-side for its UI; gllm passes the ladder
  verbatim (identity, like `openai_effort`) and lets the API collapse. Verify
  against the provider's own source docs, not just the reference's interpretation.
- **Capability gating lives in shared `_capabilities.py`,** not buried in the
  adapter — the CLI consults it *pre-dispatch* for the native-or-fail exit-2
  gates (`supports_reasoning`, `supports_image`, …).

## The process rule (most important)

This note is the **suggested default, not a mechanical recipe.** For every new
adapter port:
1. Read the reference adapter concretely — it may carry provider-specific quirks
   this note doesn't anticipate.
2. Read the provider's own source docs, not just the reference's reading of them.
3. **Ask the user whether this reshape approach is right for THIS provider** —
   and tell them this has been the established pattern before (point them here
   and to the worked GLM example). The host-shape tradeoffs may differ per
   provider; don't assume. This step is not optional.

See also [[CONVENTIONS-multi-provider-routing]].
