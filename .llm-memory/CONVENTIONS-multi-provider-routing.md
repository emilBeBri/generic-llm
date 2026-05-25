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

## OpenAI-compatible backends subclass `OpenAIProvider`

DeepSeek, Grok, and Azure OpenAI all speak the OpenAI wire protocol. Rather than duplicate the Responses/Chat-Completions logic, `OpenAIProvider.__init__` takes optional `base_url=` and `name=`. The subclass just supplies a base_url, its own key env var, and a provider tag:

- `grok.py` — `GrokProvider(OpenAIProvider)`, `base_url=https://api.x.ai/v1`, key `XAI_API_KEY`. Grok speaks the **Responses** API, so `grok` is in `_capabilities._RESPONSES_API_PREFIXES`.
- `azure_openai.py` — `AzureOpenAIProvider(OpenAIProvider)`, base_url from `AZURE_FOUNDRY_ENDPOINT` (+`/v1/`), key `AZURE_OPENAI_API_KEY`. The `-dev` suffix doesn't disturb dispatch — `use_responses_api` keys off the prefix (`gpt-5.1-dev`->Responses, `gpt-4o-dev`->Chat).
- `deepseek.py` — does *not* subclass (it's Chat-Completions-only and has no native json_schema; see body), but uses the same `openai.OpenAI` client pointed at `https://api.deepseek.com`.

`_capabilities.use_responses_api` is the single source of truth for Responses-vs-Chat dispatch, shared by `openai.py` and `azure_openai.py` (via inheritance). Unknown slugs default to Responses (the strict superset).

## Related
- [[GOTCHA-azure-foundry-constraints]] — what the two Azure adapters have to work around (no `output_config`, WORK-mode thinking, endpoint rewriting).
- [[CONVENTIONS-schemas-and-instructions]] — the json_schema strict-mode convention these adapters consume.
