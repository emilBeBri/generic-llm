# Reasoning-effort ladder: one abstract knob, per-provider translation

`-r/--reasoning low|medium|high|xhigh|max` is gllm's single reasoning control. Added 2026-06-16; the `max` rung 2026-07-29. The ladder vocabulary deliberately mirrors OpenAI's `reasoning.effort` so that path is the identity; the other providers translate.

`#architecture-decision-record` `#convention` `#multi-provider`

## Why a ladder and not pass-through

Providers disagree on the wire shape (OpenAI = effort string; Anthropic = `thinking` budget/adaptive; Gemini = `thinking_budget` int; DeepSeek = no control). One abstract level, translated per adapter, keeps the CLI Unix-y (one knob, composable) instead of leaking each provider's vocabulary. Still no `off`/`minimal` rungs.

`max` was added 2026-07-29 because it is real and otherwise unreachable: Claude Fable 5 / Opus 5 / 4.8 / 4.7 / 4.6 / Sonnet 5 / 4.6, GPT-5.6 and GLM all take it, and without it those models cannot be driven at full effort. It was safe to add only *because* the registry now knows which models have which rungs — a global fifth rung with no per-model data would just 400 everywhere else.

## Where the logic lives

`src/gllm/reasoning.py` — pure, SDK-free, unit-tested directly (see `tests/test_reasoning.py`):
- `LEVELS = ("low","medium","high","xhigh","max")`
- `openai_effort(level)` → the string (validates; identity).
- `anthropic_thinking(level, model, dialect=None)` → `{"thinking": <block>, "min_max_tokens": N}` plus, on the adaptive dialect, an `"effort"` string. **Live-verified API constraint (2026-06-16):** the adaptive family *rejects* `thinking.type=enabled` with a 400 (`"thinking.type.enabled" is not supported... Use thinking.type.adaptive and output_config.effort`). So adaptive maps every rung to `{type:adaptive, display:summarized}` + `output_config.effort=<level>` (the level, 1:1, set via `extra_body` and merged with any json_schema `format`). Budget-dialect models keep the old `enabled`+`budget_tokens` (8k/16k/32k; xhigh 32k/16k by family). **Azure Foundry also supports `output_config.effort`** (corrected 2026-06-17 against Microsoft's docs — see [[GOTCHA-azure-foundry-constraints]]), so `azure_anthropic` sends the same graded `effort` as the direct adapter (earlier it wrongly dropped it). Verified on direct Opus 4.8: output tokens scale 734→1514 from low→xhigh on a hard prompt; the Azure path needs a work-box smoke test (`AZURE-FOUNDRY-SMOKE-TEST.md`).
- `gemini_thinking_budget(level, model)` → 4096/8192/16384/`-1` (dynamic for both xhigh and max).
- `compat_effort(level)` → Groq/Regolo's `reasoning_effort`, clamping xhigh/max down to `high` (those hosts publish only three rungs).

**Which dialect a Claude model uses is registry data, not a substring match.** It used to be `_is_adaptive_family`, matching the literal strings `4-6`/`4-7`/`4-8` — which silently broke the entire Claude 5 line the day it shipped: `claude-opus-5`/`sonnet-5`/`fable-5` got the retired budget shape and 400'd on every `-r` call, until 2026-07-29. `anthropic_thinking` now takes the dialect from `ModelCaps.thinking_dialect` and the two anthropic adapters pass it in. See [[ADR-provider-model-axis]].

`supports_reasoning(provider, model, level=None)` lives in `adapters/_capabilities.py` next to `supports_pdf`, and now reads `ModelCaps.reasoning_efforts`: an empty tuple means no control at all; a non-empty one is the exact accepted set. Passing a `level` answers the sharper question the `max` rung needs — grok tops out at `high`, gpt-5.1 at `xhigh`, Sonnet 4.6 has `max` but not `xhigh`.

## Behavioural rules

- **Default = hands-off.** `reasoning=None` → no param sent → provider default. Zero regression.
- **Fail loud on unsupported — but only for an *explicit* level.** CLI gate (`cli.py`, just after `effective_model`/`provider_for`, before the status print) checks `supports_reasoning`. An *explicit* `-r/--reasoning` on a non-reasoning model exits 2 (mirrors the attachment native-or-fail matrix in [[CONVENTIONS-file-attachments]]). A level inherited from `$DEFAULT_EFFORT` is an *ambient default*, not a request, so it is **silently dropped** (`args.reasoning = None`) instead — otherwise a global `DEFAULT_EFFORT=low` would make every pipe to gpt-4.1/deepseek error. Provenance is tracked via `reasoning_was_defaulted`, mirroring the existing `model_was_defaulted` pattern. The gate runs *before* the `model_was_defaulted` status print so the printed `model:reasoning` line reflects the dropped value (prints bare model, no `:low`). Tests: `tests/test_cli_reasoning_gate.py` (the only main()-level tests; mock `_build_provider`/`_read_stdin_if_piped`/`_load_user_env_file`).
- **Token/temperature side-effects when a level is set:** Anthropic & OpenAI bump `max_tokens`/`max_output_tokens` (so reasoning doesn't starve the answer) and **drop `temperature`** (reasoning models reject a custom one). The direct Anthropic adapter also switches to `messages.stream()` when thinking is on (long generations outrun the non-streaming socket timeout). Gemini keeps temperature.
- **A rung the model lacks is now caught before the wire, not by a 400.** An explicit level outside `ModelCaps.reasoning_efforts` exits 2 and names the accepted set (`does not accept --reasoning xhigh; it accepts low, medium, high`). An *ambient* `$DEFAULT_EFFORT` is never fatal: it is **clamped down** to the model's top rung, or dropped entirely if the model can't reason. (Before the registry, `xhigh` was simply passed through and the API rejected it.) Unregistered names are granted the full ladder and still fall back to the old behaviour — gllm must not block a capability it merely hasn't been told about.

## Reasoning is fully decoupled from WORK

Earlier this adapter had `WORK=1` force-max thinking on Azure Anthropic (`_force_work_env_thinking`). That was wrong — `WORK` is a provider-routing toggle (direct vs Azure, via `routing.effective_model`), not a reasoning lever. As of 2026-06-16 the forced-thinking mechanism is **removed**; reasoning is `--reasoning` only, on every adapter equally. See [[GOTCHA-azure-foundry-constraints]].

## Related
- [[GOTCHA-azure-foundry-constraints]] — the two Azure adapters; `WORK` no longer touches thinking.
- [[CONVENTIONS-multi-provider-routing]] — how each model name reaches the adapter that does the translation.
- [[ADR-provider-model-axis]] — the registry supplying `reasoning_efforts` and `thinking_dialect`.
