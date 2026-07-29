# Models and providers are two axes; routing is a registry lookup, never a name guess

Decision 2026-07-29. `routing.provider_for` was a substring ladder over the
model name (`if "deepseek" in m: return "deepseek"`). It is now
`MODELS[key].provider` — a lookup in `src/gllm/models.py`, backed by a provider
registry in `src/gllm/providers.py`. Ported from bebri-chat's
`provider-model-axis-adr`, reshaped to gllm's sync one-shot CLI.

`#architecture-decision-record` `#model-registry` `#provider` `#multi-provider`

## Why the name cannot decide the provider

The lab that **trained** a model says nothing about the host that **serves** it.
`openai/gpt-oss-120b` is an OpenAI model answered by Groq; `glm5.2-beta` is a
Z.AI model answered by Regolo. Worse, host ids embed other vendors' names
verbatim: `groq:deepseek-r1-distill-llama-70b` contains "deepseek", so the old
ladder would have sent it to `api.deepseek.com`, and `groq:qwen/qwen3-32b`
matched nothing and fell through to the OpenAI catch-all.

The catch-all was the deeper problem: **every** unrecognised name silently
became an OpenAI call. There was no "unknown model" path at all.

## The two registries

`providers.py` — `PROVIDERS: dict[tag, ProviderSpec]`, ten hosts.
`adapter_kind` is a **string** key into `cli._build_provider`'s
kind→constructor map, never a class reference: config must not import adapters,
whose lazy per-branch imports are what stop a missing SDK breaking unrelated
providers. Several tags share a kind — `groq` and `regolo` are both
`openai_compat`. Host quirks are data on the spec (`base_url`, `extra_body`,
`image_url_format_field`, `listable`, `key_namespace`), not branches in code.

`models.py` — `MODELS: dict[key, ModelSpec]`, ~120 rows. The dict key is the
app-facing identity: what you type after `-m`, what `--usage` reports, and what
`data/prices.json` is keyed by. `wire_id` is the literal string sent over the
wire; the two differ only for namespaced host rows
(`groq:openai/gpt-oss-120b` → `openai/gpt-oss-120b`). `family` ties rows serving
the same underlying open model across hosts.

Invariants, enforced by `tests/test_registry.py` — this is the file that stops
the two dicts drifting, and it is the load-bearing half of the change:
lowercase keys; `-dev` iff Azure (and there key == wire); host rows namespaced
`<provider>:` with the bare id in `wire_id`; `alt_model`/`azure_alias` targets
resolve; effort tuples are ladder values in ladder order; a thinking dialect iff
there are rungs to translate; every `data/prices.json` key names a real model.

## Unknown is warned, not fatal

`_legacy_guess_provider` keeps the old substring ladder for names with no row,
`lru_cache`d so its stderr warning fires once per name per process. This is
deliberate, and it is the one place gllm's philosophy pushes back on the
registry: models ship faster than this file gets updated, and `gllm --models` is
the authority on what **exists** (see [[ADR-model-listing-live-probe]] — the
registry is emphatically *not* an allowlist). The registry is the authority on
how to **drive** what exists.

But a silent mis-route is how an agent ends up reporting success for a model
nobody asked for, so the guess announces itself. Read the warning as a red flag
on the model name.

## Capabilities moved onto the rows

`ModelCaps` (api surface, reasoning rungs, thinking dialect, vision, pdf, strict
schema) is read by every gate in `adapters/_capabilities.py`, which kept its
function names and became a thin facade. `models._legacy_caps` reconstructs a
guess from the old substring predicates for unregistered names, granting the
full effort ladder — gllm must not block a capability it merely hasn't been told
about; let the API 400.

This is what fixed the Claude 5 thinking bug: see
[[ADR-reasoning-effort-ladder]].

Two capability facts became declarative that used to be adapter-internal raises:
the GLM vision split (`supports_image` now takes a model — see
[[CONVENTIONS-zai-glm-adapter]]) and the per-model reasoning vocabulary.

## WORK mode stopped doing string surgery

`effective_model` used to append `-dev`. It now reads `ModelSpec.azure_alias`,
so the registry states which models actually have a Foundry deployment. A
registered Anthropic/OpenAI model with no alias still falls back to the `-dev`
append — but loudly, because **Azure deployment inventory is live data the
registry cannot know**, and a confident guess there 404s as
`DeploymentNotFound`. See [[GOTCHA-azure-foundry-constraints]].

## Adding a host is config, not code

`adapters/openai_compat.py` is one adapter parameterised by a `ProviderSpec`.
Host number three should be a `PROVIDERS` row + `MODELS` rows + prices and zero
new code. It is standalone rather than an `OpenAIProvider` subclass for the same
reason the Z.AI adapter is: the parent routes by name to the Responses API,
which these hosts do not speak. See
[[CONVENTIONS-porting-adapters-from-reference]].

**Regolo's `extra_body={'disable_fallbacks': True}` is load-bearing.** Without
it Regolo silently answers with a *different* model than requested, which makes
model identity, cost accounting and every capability gate a lie.

## Known cost: two catalogues, one truth

gllm and bebri-chat now each carry their own copy of essentially the same
catalogue, and they will drift. Accepted deliberately — gllm must not import
bebri-chat, and a shared data file is a bigger change than this one. When a
vendor refreshes, update both. A shared JSON is the obvious future move.

## Related
- [[CONVENTIONS-multi-provider-routing]] — the mechanics the registry replaced.
- [[ADR-model-listing-live-probe]] — why `--models` stays a live probe.
- [[ADR-reasoning-effort-ladder]] — the `max` rung and the dialect dispatch.
- [[CONVENTIONS-usage-cost-emission]] — `data/prices.json` is keyed by registry key.
