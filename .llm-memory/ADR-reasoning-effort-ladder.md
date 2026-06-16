# Reasoning-effort ladder: one abstract knob, per-provider translation

`-r/--reasoning low|medium|high|xhigh` is gllm's single reasoning control. Added 2026-06-16. The ladder vocabulary deliberately mirrors OpenAI's `reasoning.effort` so that path is the identity; the other providers translate.

`#architecture-decision-record` `#convention` `#multi-provider`

## Why a ladder and not pass-through

Providers disagree on the wire shape (OpenAI = effort string; Anthropic = `thinking` budget/adaptive; Gemini = `thinking_budget` int; DeepSeek = no control). One abstract level, translated per adapter, keeps the CLI Unix-y (one knob, composable) instead of leaking each provider's vocabulary. No `off`/`max`/`minimal` rungs — just the four the user asked for.

## Where the logic lives

`src/gllm/reasoning.py` — pure, SDK-free, unit-tested directly (see `tests/test_reasoning.py`):
- `LEVELS = ("low","medium","high","xhigh")`
- `openai_effort(level)` → the string (validates; identity).
- `anthropic_thinking(level, model)` → `{"thinking": <block>, "min_max_tokens": N}`. Lower rungs are `enabled` budgets 8k/16k/32k; `xhigh` is the per-family max (adaptive+summarized/64k for 4.6–4.8, enabled 32k/64k for 4.5, enabled 16k/32k else — these are the numbers the old WORK forced-thinking used). `display:summarized` is forced on the 4.6/4.7/4.8 family for *any* block.
- `gemini_thinking_budget(level, model)` → 4096/8192/16384/`-1` (dynamic).

`supports_reasoning(provider, model)` lives in `adapters/_capabilities.py` next to `supports_pdf`: Anthropic/Gemini families → True; OpenAI/Grok/Azure-OpenAI → `use_responses_api(model)` (so gpt-4o → False); DeepSeek → False.

## Behavioural rules

- **Default = hands-off.** `reasoning=None` → no param sent → provider default. Zero regression.
- **Fail loud on unsupported.** CLI gate (`cli.py`, beside the attachment native-or-fail loop) exits 2 if `--reasoning` is given but `supports_reasoning` is False. Mirrors the attachment matrix in [[CONVENTIONS-file-attachments]].
- **Token/temperature side-effects when a level is set:** Anthropic & OpenAI bump `max_tokens`/`max_output_tokens` (so reasoning doesn't starve the answer) and **drop `temperature`** (reasoning models reject a custom one). The direct Anthropic adapter also switches to `messages.stream()` when thinking is on (long generations outrun the non-streaming socket timeout). Gemini keeps temperature.
- **`xhigh` may 400** on models that cap lower (some o-series, `grok-3-mini`); passed through, the API rejects loudly. Not pre-guarded.

## Reasoning is fully decoupled from WORK

Earlier this adapter had `WORK=1` force-max thinking on Azure Anthropic (`_force_work_env_thinking`). That was wrong — `WORK` is a provider-routing toggle (direct vs Azure, via `routing.effective_model`), not a reasoning lever. As of 2026-06-16 the forced-thinking mechanism is **removed**; reasoning is `--reasoning` only, on every adapter equally. See [[GOTCHA-azure-foundry-constraints]].

## Related
- [[GOTCHA-azure-foundry-constraints]] — the two Azure adapters; `WORK` no longer touches thinking.
- [[CONVENTIONS-multi-provider-routing]] — how each model name reaches the adapter that does the translation.
