# Gotcha: Azure Foundry adapter constraints & WORK mode

Things the two Azure Foundry adapters (`azure_openai.py`, `azure_anthropic.py`) have to work around that the direct providers don't. Ported from bebri-chat on 2026-05-25.

`#gotcha` `#architecture-decision-record` `#environment`

## Auth & endpoint

Both Azure adapters share `AZURE_FOUNDRY_ENDPOINT`; keys differ (`AZURE_OPENAI_API_KEY`, `AZURE_ANTHROPIC_API_KEY`). The adapters read these from env in `__init__` (like the other gllm adapters), so they only construct when actually selected.

`azure_anthropic._normalize_foundry_url` rewrites an **Agents** endpoint (`*.services.ai.azure.com`, `*.cognitiveservices.azure.com`) to the resource's `*.openai.azure.com` MaaS host, then appends `/anthropic`. Azure OpenAI just appends `/v1/` (NOT the classic `/openai/deployments/...` — that's Azure OpenAI *Service*, not Foundry MaaS).

## Azure Anthropic has no `output_config`

Direct Anthropic uses native `output_config.format = json_schema` for structured output. **Azure Foundry does not support it.** So `azure_anthropic.py` emulates `--schema`/`--json` by injecting an instruction into the system prompt (pasting the schema text for `--schema`) — the same fallback the direct adapter uses for bare `--json`. There is also no effort/`output_config` thinking control on Azure.

## WORK mode — forced extended thinking (`config.work_env`)

`WORK=1` or `WORK_ENV=1` (truthy: 1/true/yes/on; `WORK` wins). In bebri-chat this is `WORK_ENV` in `.env`; gllm also accepts the bare `WORK=1` as an ergonomic per-invocation flag (`WORK=1 gllm -m claude-opus-4-7-dev ...`).

Effect is **only** in `azure_anthropic.py` (`_force_work_env_thinking`), scaled by model family:
- `4-6`/`4-7` -> `thinking={type: adaptive, display: summarized}`, `max_tokens=64000`. `display:summarized` is **required** on 4.7 (its default flipped to `omitted`) or the thinking is suppressed.
- `4-5` -> `{type: enabled, budget_tokens: 32000}`, `max_tokens=64000`.
- else -> `{type: enabled, budget_tokens: 16000}`, `max_tokens=32000`.

When thinking is forced we **drop `temperature`** (extended thinking pins it to 1) and **stream** (`messages.stream().get_final_message()`) so a long reasoning generation doesn't outrun the non-streaming socket timeout. Only `text` blocks are returned; thinking blocks are discarded (gllm prints final text only).

bebri-chat's WORK_ENV also gated `-dev` model *visibility* in its picker and disabled SSL verification (corporate proxy). gllm has no picker and does no web fetching, so neither was ported — WORK mode here is purely the thinking knob.

## Related
- [[CONVENTIONS-multi-provider-routing]] — how `-dev` models route to these adapters in the first place.
