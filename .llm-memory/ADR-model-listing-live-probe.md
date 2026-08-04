# `gllm --models` probes the live API; there is no model allowlist

#architecture-decision-record #model-routing #gotcha

## The invariant: adapters forward the model name verbatim

gllm has **no catalog gate**. The model string flows `cli → provider_for()`
(registry lookup to pick the adapter) `→ adapter.generate()`, and each adapter
hands `request.wire_model or request.model` straight to the SDK call (`client.models.generate_content`,
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

- **Configured providers only**: bare `--models` first filters by the key and
  required endpoint variables declared on `ProviderSpec`. Missing credentials
  mean unavailable, not an error to print nine times. An explicitly requested
  provider still fails loudly when it is incomplete or its API fails.
- **WORK follows actual routing**: in work mode, direct Anthropic/OpenAI
  discovery is replaced by `azure_anthropic`/`azure_openai`; otherwise the
  picker can advertise public models that the subsequent call will redirect
  away from.
- **Azure deployment exception**: Foundry inference APIs do not expose a live
  deployment-listing endpoint (the Anthropic SDK explicitly disables
  `AnthropicFoundry.models`). Azure rows therefore come from the registry's
  explicit `-dev` deployment entries, gated on the matching Azure key plus
  `AZURE_FOUNDRY_ENDPOINT`. All other providers remain live API probes. See
  [[GOTCHA-azure-foundry-constraints]].
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

## 2026-07-29: a registry arrived, and it is STILL not an allowlist

[[ADR-provider-model-axis]] added `models.MODELS`, which sounds like exactly the
hand-maintained catalog this note argues against. It isn't, and the distinction
is load-bearing:

- The registry answers **how to drive** a model — provider, wire id, thinking
  dialect, effort rungs, vision/pdf/schema. `--models` answers **what exists**.
  The second question is the one that rots, and it is still asked live.
- A name with no row **still runs**. `routing._legacy_guess_provider` guesses the
  provider from the name, warns once on stderr, and dispatches. Unknown is a red
  flag, never a refusal — precisely so the cdf6619 failure above cannot repeat in
  a new costume ("gllm says it's not a real model, so it must be dead").
- `--models` output is now printed with the host `key_namespace` prefix
  (`groq:`/`regolo:`) so a row is pasteable straight into `-m`. The rows still
  come from the live API, not from the registry.

The one thing the registry legitimately buys here: a hallucinated slug now says
so out loud instead of silently becoming an OpenAI call via the old catch-all.
