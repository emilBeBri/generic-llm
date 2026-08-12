# Convention: provider routing & OpenAI-compatible adapters

How `gllm` picks a provider for a model, and how OpenAI-compatible backends are added cheaply. Ported from bebri-chat's `get_model_provider` / `MultiProviderRouter` on 2026-05-25, reduced to gllm's sync one-shot `Request -> Response` shape (no tools, no conversation, no thinking config).

`#convention` `#architecture-decision-record`

## Routing — SUPERSEDED 2026-07-29 by [[ADR-provider-model-axis]]

`routing.provider_for(model)` is now a lookup in the `models.MODELS` registry, not name inference. The table below is the **historical** ladder; in code it survives only as `routing._legacy_guess_provider`, for names with no registry row, and it warns on stderr when it fires.

| Model name matches | Provider | Adapter |
|---|---|---|
| starts `groq:` / `regolo:` | that host | `openai_compat.py` |
| ends `-dev` + contains `claude` | `azure_anthropic` | `azure_anthropic.py` |
| ends `-dev` (else) | `azure_openai` | `azure_openai.py` |
| contains `claude` | `anthropic` | `anthropic.py` |
| contains `gemini` | `gemini` | `gemini.py` |
| contains `deepseek` | `deepseek` | `deepseek.py` |
| contains `grok` | `grok` | `grok.py` |
| contains `glm` | `zai` | `zai.py` |
| else (`gpt-*`, `o1/o3/o4`, `codex`) | `openai` | `openai.py` |

Why it had to go: the ladder assumes a model name names its *vendor*, but hosts serve other labs' models — `groq:deepseek-r1-distill-llama-70b` would have routed to `api.deepseek.com`, and `groq:qwen/qwen3-32b` matched nothing and fell through to the OpenAI catch-all. The host-namespace branch at the top of the fallback is a patch so it stops misfiring; the real fix is the registry.

`cli._build_provider` now dispatches on `PROVIDERS[name].adapter_kind` (a string key), still via lazily-imported adapter classes so a missing SDK only breaks its own provider. As adapters move to the stdlib transport ([[ADR-stdlib-http-transport]]) the "missing SDK" half of that rationale dissolves, but keep the lazy import: it is what holds the non-network paths (`--help`, `--models`) at ~87 ms.

`routing.effective_model(model, work)` is the WORK-mode Azure redirect. It reads `ModelSpec.azure_alias` instead of appending `-dev`; a registered Anthropic/OpenAI model with no alias still gets the append, but loudly. `cli.main` calls it right after resolving `-m`. WORK is **only** this routing toggle — it has nothing to do with reasoning (see [[GOTCHA-azure-foundry-constraints]] and [[ADR-reasoning-effort-ladder]]).

## OpenAI-compatible backends: subclass, or the generic compat adapter

Two shapes, and the question that picks between them is **"is this host's wire protocol the OpenAI Responses API with only `base_url` differing?"**

**Subclass `OpenAIProvider`** when yes. `OpenAIProvider.__init__` takes optional `base_url=` and `name=`; the subclass supplies a base_url, its own key env var, and a provider tag:

- `grok.py` — `GrokProvider(OpenAIProvider)`, `base_url=https://api.x.ai/v1`, key `XAI_API_KEY`. Grok speaks the **Responses** API.
- `azure_openai.py` — `AzureOpenAIProvider(OpenAIProvider)`, base_url from `AZURE_FOUNDRY_ENDPOINT` (+`/v1/`), key `AZURE_OPENAI_API_KEY`. The `-dev` suffix doesn't disturb dispatch.

**Standalone** when no — the parent would route the model to Responses, which the host can't speak:

- `deepseek.py` — Chat-Completions only, no native json_schema. `openai.OpenAI` at `https://api.deepseek.com`.
- `zai.py` — same reasoning plus GLM's own thinking/vision shape. See [[CONVENTIONS-zai-glm-adapter]].
- `openai_compat.py` — the **generic** one, parameterised by a `ProviderSpec` (base_url, extra_body, image dialect). Serves `groq` and `regolo` today; the next host should be a `PROVIDERS` row + `MODELS` rows + prices and **zero new code**.

`_capabilities.use_responses_api` remains the single source of truth for Responses-vs-Chat dispatch, but now reads `ModelCaps.api_surface` from the registry (falling back to the old prefix check for unregistered names). Unknown slugs still default to Responses, the strict superset.

## Anthropic family thinking — dialect is registry data, not a substring

Historically `reasoning._is_adaptive_family` matched the literal strings `4-6`/`4-7`/`4-8`. **That was a bug the moment the Claude 5 line shipped**: `claude-opus-5`, `claude-sonnet-5` and `claude-fable-5` fell through to the retired `enabled`+`budget_tokens` shape, which those models reject with a 400. Every `-r` call against Claude 5 was broken until 2026-07-29.

The dialect now comes from `ModelCaps.thinking_dialect` on the model's row, so a new family is a registry row, not a string edit:

- `anthropic_adaptive` (Claude 4.6/4.7/4.8, Sonnet 4.6, and the whole 5 line) — `thinking={type:"adaptive", display:"summarized"}`, `max_tokens=64000`, effort graded by `output_config.effort` on **both** the direct API and Azure Foundry.
- `anthropic_budget` (4.5 and older) — `enabled`+`budget_tokens`.

`display:"summarized"` stays mandatory on 4-7+ (the default flipped to `omitted`, which suppresses streaming thinking deltas — the terminal looks hung).

## Related
- [[ADR-provider-model-axis]] — the registry that replaced the ladder.
- [[GOTCHA-azure-foundry-constraints]] — Azure adapter specifics (Foundry DOES expose `output_config`; `effort` verified, `format` an unverified attempt; the `WORK` routing toggle; endpoint rewriting).
- [[CONVENTIONS-schemas-and-instructions]] — the json_schema strict-mode convention these adapters consume.
