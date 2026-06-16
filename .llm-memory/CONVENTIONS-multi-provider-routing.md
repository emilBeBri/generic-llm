# Convention: provider routing & OpenAI-compatible adapters

How `gllm` picks a provider from a model name, and how OpenAI-compatible backends are added cheaply. Ported from bebri-chat's `get_model_provider` / `MultiProviderRouter` on 2026-05-25, reduced to gllm's sync one-shot `Request -> Response` shape (no tools, no conversation, no thinking config).

`#convention` `#architecture-decision-record`

## Routing — `routing.provider_for(model)`

Pure model-name inference, no config. Order matters: the Azure `-dev` suffix is checked **first**.

| Model name matches | Provider | Adapter |
|---|---|---|
| ends `-dev` + contains `claude` | `azure_anthropic` | `azure_anthropic.py` |
| ends `-dev` (else) | `azure_openai` | `azure_openai.py` |
| contains `claude` | `anthropic` | `anthropic.py` |
| contains `gemini` | `gemini` | `gemini.py` |
| contains `deepseek` | `deepseek` | `deepseek.py` |
| contains `grok` | `grok` | `grok.py` |
| else (`gpt-*`, `o1/o3/o4`, `codex`) | `openai` | `openai.py` |

**The `-dev` suffix is the Azure Foundry marker.** This is the one non-obvious bit: `claude-opus-4-7` hits the direct Anthropic API, `claude-opus-4-7-dev` hits Azure Foundry. Mirrors bebri-chat exactly. `cli._build_provider` maps the provider string to a lazily-imported adapter class.

`routing.effective_model(model, work)` is the WORK-mode front door to the same suffix: under `WORK=1` it appends `-dev` to direct Anthropic/OpenAI names (set `_AZURE_REDIRECTABLE`), so a clean `claude-opus-4-8` routes to Azure with deployment name `claude-opus-4-8-dev`. `cli.main` calls it right after resolving `-m`. WORK is **only** this routing toggle — it has nothing to do with reasoning (see [[GOTCHA-azure-foundry-constraints]] and [[ADR-reasoning-effort-ladder]]).

## OpenAI-compatible backends subclass `OpenAIProvider`

DeepSeek, Grok, and Azure OpenAI all speak the OpenAI wire protocol. Rather than duplicate the Responses/Chat-Completions logic, `OpenAIProvider.__init__` takes optional `base_url=` and `name=`. The subclass just supplies a base_url, its own key env var, and a provider tag:

- `grok.py` — `GrokProvider(OpenAIProvider)`, `base_url=https://api.x.ai/v1`, key `XAI_API_KEY`. Grok speaks the **Responses** API, so `grok` is in `_capabilities._RESPONSES_API_PREFIXES`.
- `azure_openai.py` — `AzureOpenAIProvider(OpenAIProvider)`, base_url from `AZURE_FOUNDRY_ENDPOINT` (+`/v1/`), key `AZURE_OPENAI_API_KEY`. The `-dev` suffix doesn't disturb dispatch — `use_responses_api` keys off the prefix (`gpt-5.1-dev`->Responses, `gpt-4o-dev`->Chat).
- `deepseek.py` — does *not* subclass (it's Chat-Completions-only and has no native json_schema; see body), but uses the same `openai.OpenAI` client pointed at `https://api.deepseek.com`.

`_capabilities.use_responses_api` is the single source of truth for Responses-vs-Chat dispatch, shared by `openai.py` and `azure_openai.py` (via inheritance). Unknown slugs default to Responses (the strict superset).

## Anthropic family thinking — 4-6 / 4-7 / 4-8 are one bucket

`reasoning._is_adaptive_family` treats Claude **4-6, 4-7, and 4-8** identically: `thinking={type:"adaptive", display:"summarized"}`, `max_tokens=64000`, with the effort graded by `output_config.effort` (= the `--reasoning` level) on **both** the direct API and Azure Foundry (Foundry supports `effort` — see [[GOTCHA-azure-foundry-constraints]]). These models **reject** the old `enabled`+`budget_tokens` shape (live-verified 400 — see [[ADR-reasoning-effort-ladder]]); 4-5 & older still use it. When a new family lands (4-9, 5-x), add it to `_is_adaptive_family`. `display:"summarized"` is mandatory on 4-7+ (default flipped to `omitted`, which suppresses streaming thinking deltas — terminal looks hung).

## Related
- [[GOTCHA-azure-foundry-constraints]] — Azure adapter specifics (Foundry DOES expose `output_config`; `effort` verified, `format` an unverified attempt; the `WORK` routing toggle; endpoint rewriting).
- [[CONVENTIONS-schemas-and-instructions]] — the json_schema strict-mode convention these adapters consume.
