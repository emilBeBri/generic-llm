# Gotcha: Azure Foundry adapter constraints & WORK mode

Things the two Azure Foundry adapters (`azure_openai.py`, `azure_anthropic.py`) have to work around that the direct providers don't. Ported from bebri-chat on 2026-05-25.

`#gotcha` `#architecture-decision-record` `#environment`

## Auth & endpoint

Both Azure adapters share `AZURE_FOUNDRY_ENDPOINT`; keys differ (`AZURE_OPENAI_API_KEY`, `AZURE_ANTHROPIC_API_KEY`). The adapters read these from env in `__init__` (like the other gllm adapters), so they only construct when actually selected.

`azure_anthropic._normalize_foundry_url` rewrites an **Agents** endpoint (`*.services.ai.azure.com`, `*.cognitiveservices.azure.com`) to the resource's `*.openai.azure.com` MaaS host, then appends `/anthropic`. Azure OpenAI just appends `/v1/` (NOT the classic `/openai/deployments/...` — that's Azure OpenAI *Service*, not Foundry MaaS).

## Azure Anthropic has no `output_config`

Direct Anthropic uses native `output_config.format = json_schema` for structured output. **Azure Foundry does not support it.** So `azure_anthropic.py` emulates `--schema`/`--json` by injecting an instruction into the system prompt (pasting the schema text for `--schema`) — the same fallback the direct adapter uses for bare `--json`. There is also no effort/`output_config` thinking control on Azure.

## WORK mode (`config.work_env`) — a routing toggle, NOT a thinking knob

`WORK=1` or `WORK_ENV=1` (truthy: 1/true/yes/on; `WORK` wins). In bebri-chat this is `WORK_ENV` in `.env`; gllm also accepts the bare `WORK=1` per-invocation.

**History / correction (2026-06-16):** the original gllm port wrongly made `WORK` *force maximum extended thinking* on Azure Anthropic (`_force_work_env_thinking`). That coupling is **removed** — reasoning is `--reasoning` only (see [[ADR-reasoning-effort-ladder]]). `WORK` is now what it is in bebri-chat: the corporate/Azure switch.

**Implemented semantic:** `routing.effective_model(model, work)` — under `WORK=1` a *direct* Anthropic/OpenAI model name gets `-dev` appended (`claude-opus-4-8` → `claude-opus-4-8-dev`), so `provider_for` then routes it to the Azure adapter and the `-dev` string is the deployment name sent to Foundry. Already-`-dev` names, `WORK=0`, and non-Azure providers (Gemini/Grok/DeepSeek) pass through unchanged; an explicit `-dev` name still selects Azure regardless of `WORK`. `cli.main` applies `effective_model` right after resolving `-m`, so the Request, routing, capability gates and verbose log all see the effective name. The redirect set is `routing._AZURE_REDIRECTABLE = {"anthropic", "openai"}` (the only two direct providers with a Foundry counterpart).

## Related
- [[CONVENTIONS-multi-provider-routing]] — how `-dev` models route to these adapters in the first place.
