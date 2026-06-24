# Reasoning-effort ladder: one abstract knob, per-provider translation

`-r/--reasoning low|medium|high|xhigh` is gllm's single reasoning control. Added 2026-06-16. The ladder vocabulary deliberately mirrors OpenAI's `reasoning.effort` so that path is the identity; the other providers translate.

`#architecture-decision-record` `#convention` `#multi-provider`

## Why a ladder and not pass-through

Providers disagree on the wire shape (OpenAI = effort string; Anthropic = `thinking` budget/adaptive; Gemini = `thinking_budget` int; DeepSeek = no control). One abstract level, translated per adapter, keeps the CLI Unix-y (one knob, composable) instead of leaking each provider's vocabulary. No `off`/`max`/`minimal` rungs — just the four the user asked for.

## Where the logic lives

`src/gllm/reasoning.py` — pure, SDK-free, unit-tested directly (see `tests/test_reasoning.py`):
- `LEVELS = ("low","medium","high","xhigh")`
- `openai_effort(level)` → the string (validates; identity).
- `anthropic_thinking(level, model)` → `{"thinking": <block>, "min_max_tokens": N}` plus, for the adaptive family, an `"effort"` string. **Live-verified API constraint (2026-06-16):** Claude 4.6/4.7/4.8 *reject* `thinking.type=enabled` with a 400 (`"thinking.type.enabled" is not supported... Use thinking.type.adaptive and output_config.effort`). So the adaptive family maps every rung to `{type:adaptive, display:summarized}` + `output_config.effort=<level>` (the level, 1:1, set via `extra_body` and merged with any json_schema `format`). 4.5 & older keep the old `enabled`+`budget_tokens` (8k/16k/32k; xhigh 32k/16k by family). **Azure Foundry also supports `output_config.effort`** (corrected 2026-06-17 against Microsoft's docs — see [[GOTCHA-azure-foundry-constraints]]), so `azure_anthropic` sends the same graded `effort` as the direct adapter (earlier it wrongly dropped it). Verified on direct Opus 4.8: output tokens scale 734→1514 from low→xhigh on a hard prompt; the Azure path needs a work-box smoke test (`AZURE-FOUNDRY-SMOKE-TEST.md`).
- `gemini_thinking_budget(level, model)` → 4096/8192/16384/`-1` (dynamic).

`supports_reasoning(provider, model)` lives in `adapters/_capabilities.py` next to `supports_pdf`: Anthropic/Gemini families → True; OpenAI/Grok/Azure-OpenAI → `use_responses_api(model)` (so gpt-4o → False); DeepSeek → False.

## Behavioural rules

- **Default = hands-off.** `reasoning=None` → no param sent → provider default. Zero regression.
- **Fail loud on unsupported — but only for an *explicit* level.** CLI gate (`cli.py`, just after `effective_model`/`provider_for`, before the status print) checks `supports_reasoning`. An *explicit* `-r/--reasoning` on a non-reasoning model exits 2 (mirrors the attachment native-or-fail matrix in [[CONVENTIONS-file-attachments]]). A level inherited from `$DEFAULT_EFFORT` is an *ambient default*, not a request, so it is **silently dropped** (`args.reasoning = None`) instead — otherwise a global `DEFAULT_EFFORT=low` would make every pipe to gpt-4.1/deepseek error. Provenance is tracked via `reasoning_was_defaulted`, mirroring the existing `model_was_defaulted` pattern. The gate runs *before* the `model_was_defaulted` status print so the printed `model:reasoning` line reflects the dropped value (prints bare model, no `:low`). Tests: `tests/test_cli_reasoning_gate.py` (the only main()-level tests; mock `_build_provider`/`_read_stdin_if_piped`/`_load_user_env_file`).
- **Token/temperature side-effects when a level is set:** Anthropic & OpenAI bump `max_tokens`/`max_output_tokens` (so reasoning doesn't starve the answer) and **drop `temperature`** (reasoning models reject a custom one). The direct Anthropic adapter also switches to `messages.stream()` when thinking is on (long generations outrun the non-streaming socket timeout). Gemini keeps temperature.
- **`xhigh` may 400** on models that cap lower (some o-series, `grok-3-mini`); passed through, the API rejects loudly. Not pre-guarded.

## Reasoning is fully decoupled from WORK

Earlier this adapter had `WORK=1` force-max thinking on Azure Anthropic (`_force_work_env_thinking`). That was wrong — `WORK` is a provider-routing toggle (direct vs Azure, via `routing.effective_model`), not a reasoning lever. As of 2026-06-16 the forced-thinking mechanism is **removed**; reasoning is `--reasoning` only, on every adapter equally. See [[GOTCHA-azure-foundry-constraints]].

## Related
- [[GOTCHA-azure-foundry-constraints]] — the two Azure adapters; `WORK` no longer touches thinking.
- [[CONVENTIONS-multi-provider-routing]] — how each model name reaches the adapter that does the translation.
