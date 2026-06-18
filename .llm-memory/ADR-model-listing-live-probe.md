# `gllm --models` probes the live API; there is no model allowlist

#architecture-decision-record #model-routing #gotcha

## The invariant: adapters forward the model name verbatim

gllm has **no catalog gate**. The model string flows `cli → provider_for()`
(substring match to pick the adapter) `→ adapter.generate()`, and each adapter
hands `request.model` straight to the SDK call (`client.models.generate_content`,
`messages.create`, `responses.create`, …). Nothing checks the name against a
known-models list. Any id the provider's API serves works without a code change;
any id it doesn't 404s at call time. See [[CONVENTIONS-multi-provider-routing]].

Corollary: the "Known models" lists in the README/tests are **orientation, not
truth**. They are hand-maintained and drift.

## The drift that motivated this

cdf6619 ("drop dead Gemini models") declared `gemini-3-flash` and
`gemini-3-pro-preview` "retired / 404" based on a one-off probe — but it probed
the **wrong id**. Google's preview models carry a `-preview` suffix: the live
ids are `gemini-3-flash-preview` and `gemini-3-pro-preview` (confirmed live via
`models.list()` on 2026-06-18). The bare `gemini-3-flash` 404s because it's a
malformed id, not because the model is gone. A later agent trusted the stale
README, told the user the model was dead, and substituted a different one. The
stale catalog actively caused a wrong answer.

## Decision: `gllm --models [PROVIDER]`

A discovery command that asks each provider's live `models.list()` endpoint
instead of trusting ourselves. Prints greppable `provider<TAB>id` rows; pipe to
`rg`/`fzf`. `--models` alone probes every listable provider; `--models gemini`
restricts to one. It short-circuits in `main()` before any prompt/attachment
handling (needs neither).

- **Listable providers**: anthropic, openai, gemini, grok, deepseek. Azure
  Foundry is excluded — it's *deployment*-scoped (you list your deployments, not
  a global catalog), so it has no equivalent endpoint. See
  [[GOTCHA-azure-foundry-constraints]].
- **Loud skip, never silent**: a provider with no key or a failing call prints
  `gllm: <name>: skipped (...)` to stderr and continues. Matches the project's
  fail-loud, no-silent-fallback stance.
- **Text-generation filter**: `list_models()` returns text-gen models only.
  Gemini uses a two-stage filter — the API's `supported_actions` must include
  `generateContent` (drops embeddings), AND a name-based blocklist
  (`is_text_generation_model` in `_capabilities.py`) because TTS/image/music/
  robotics models *also* advertise `generateContent`. OpenAI-family catalogs
  carry no capability metadata at all, so only the name blocklist applies.
  `_NON_TEXT_GEN_MARKERS` is a substring blocklist (embedding/tts/image/video/
  audio/sora/imagine/lyria/robotics/computer-use/moderation/…). Heuristic by
  necessity — a false negative hides a row from `--models` but can never block a
  real `generate()` call, since dispatch never consults the filter.

## Mechanics

`list_models() -> list[str]` is an optional method on the `LLMProvider` port
(base default raises `NotImplementedError`), implemented per adapter. Grok
inherits OpenAIProvider's implementation unchanged.
