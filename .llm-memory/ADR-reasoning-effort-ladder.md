# Reasoning-effort ladder: one abstract knob, per-provider translation

`-r/--reasoning low|medium|high|xhigh` is gllm's single reasoning control. Added 2026-06-16. The ladder vocabulary deliberately mirrors OpenAI's `reasoning.effort` so that path is the identity; the other providers translate.

`#architecture-decision-record` `#convention` `#multi-provider`

## Why a ladder and not pass-through

Providers disagree on the wire shape (OpenAI = effort string; Anthropic = `thinking` budget/adaptive; Gemini = `thinking_budget` int; DeepSeek = no control). One abstract level, translated per adapter, keeps the CLI Unix-y (one knob, composable) instead of leaking each provider's vocabulary. Still no `off`/`minimal` rungs.

# The ladder is NORMALISED, not literal (2026-07-31)

A `max` rung was briefly added (2026-07-29) so Claude 5 / GPT-5.6 / GLM could be driven at full effort, then **removed the next day**. Chasing a DeepSeek question showed why the whole approach was wrong: provider vocabularies do not merely differ in length, they disagree completely. DeepSeek publishes `{high, max}`; GLM-5.2 honours only those two; grok-4.5 takes `{low..xhigh}`; gpt-5.6 takes `{none..max}`; Gemini takes `{minimal..high}`. No single literal ladder covers that, and growing gllm's vocabulary to the union would make `-r` mean different things on different models — fatal for a CLI whose whole value is being scriptable.

**Decision: four fixed rungs, normalised per model.** `reasoning.resolve_effort(level, native)`:

- `xhigh` ALWAYS resolves to `native_efforts[-1]` — "the most this model has", whatever it is called there;
- any other rung keeps its own name where the provider has it;
- otherwise it clamps to the nearest by `_RANK`, breaking ties toward the cheaper value.

The invariant that makes this safe to leave in a script: **`-r low` is always the cheapest setting a model offers**, never silently upgraded (`test_low_is_never_upgraded`). Only the top rung is ever remapped.

`ModelCaps.native_efforts` therefore holds the PROVIDER's vocabulary, cheapest first — not gllm's rungs. It was renamed from `reasoning_efforts` deliberately, to force every registry row to be revisited rather than silently reinterpreted.

**Translation is announced.** When the resolved value differs from what was typed, `cli.py` prints `gllm: -r xhigh -> 'max' (deepseek-v4-pro offers: high, max)` to stderr — silent on a pass-through, suppressed by `-q/--quiet-effort`. A rung that quietly means something else is exactly the soft degradation gllm refuses for schemas and attachments; it does not get a pass here.

**bebri-chat deliberately does the opposite.** Its `/think` offers each model's real levels, because an interactive REPL user is choosing per-model in the moment and wants the truth, not a normalisation. Two tools, two correct answers. See bebri-chat's `[[grok-per-model-effort-vocabulary]]`.

## Where the logic lives

`src/gllm/reasoning.py` — pure, SDK-free, unit-tested directly (see `tests/test_reasoning.py`):
- `LEVELS = ("low","medium","high","xhigh")` — four rungs, stable forever.
- `resolve_effort(level, native)` — normalises a rung onto the model's own vocabulary.
- `openai_effort(level)` → the string (validates; identity).
- `anthropic_thinking(level, model, dialect=None)` → `{"thinking": <block>, "min_max_tokens": N}` plus, on the adaptive dialect, an `"effort"` string. **Live-verified API constraint (2026-06-16):** the adaptive family *rejects* `thinking.type=enabled` with a 400 (`"thinking.type.enabled" is not supported... Use thinking.type.adaptive and output_config.effort`). So adaptive maps every rung to `{type:adaptive, display:summarized}` + `output_config.effort=<level>` (the level, 1:1, set via `extra_body` and merged with any json_schema `format`). Budget-dialect models keep the old `enabled`+`budget_tokens` (8k/16k/32k; xhigh 32k/16k by family). **Azure Foundry also supports `output_config.effort`** (corrected 2026-06-17 against Microsoft's docs — see [[GOTCHA-azure-foundry-constraints]]), so `azure_anthropic` sends the same graded `effort` as the direct adapter (earlier it wrongly dropped it). Verified on direct Opus 4.8: output tokens scale 734→1514 from low→xhigh on a hard prompt; the Azure path needs a work-box smoke test (`AZURE-FOUNDRY-SMOKE-TEST.md`).
- `gemini_thinking_budget(level, model)` → 4096/8192/16384/`-1` (dynamic for both xhigh and max).
- Dialects no longer translate levels themselves. The CLI resolves once and stores the provider's own value on `Request.wire_effort` (mirroring `wire_model`); each adapter just sends it. Only shape differs per dialect: a string for OpenAI/Grok/GLM/DeepSeek/compat hosts, a `thinking_budget` int for Gemini, a `thinking` block for Anthropic.

**Which dialect a Claude model uses is registry data, not a substring match.** It used to be `_is_adaptive_family`, matching the literal strings `4-6`/`4-7`/`4-8` — which silently broke the entire Claude 5 line the day it shipped: `claude-opus-5`/`sonnet-5`/`fable-5` got the retired budget shape and 400'd on every `-r` call, until 2026-07-29. `anthropic_thinking` now takes the dialect from `ModelCaps.thinking_dialect` and the two anthropic adapters pass it in. See [[ADR-provider-model-axis]].

`supports_reasoning(provider, model)` lives in `adapters/_capabilities.py` next to `supports_pdf` and reads `bool(ModelCaps.native_efforts)`. `native_efforts(provider, model)` returns the vocabulary itself, for the translation notice.

## Behavioural rules

- **Default = hands-off.** `reasoning=None` → no param sent → provider default. Zero regression.
- **Fail loud on unsupported — but only for an *explicit* level.** CLI gate (`cli.py`, just after `effective_model`/`provider_for`, before the status print) checks `supports_reasoning`. An *explicit* `-r/--reasoning` on a non-reasoning model exits 2 (mirrors the attachment native-or-fail matrix in [[CONVENTIONS-file-attachments]]). A level inherited from `$DEFAULT_EFFORT` is an *ambient default*, not a request, so it is **silently dropped** (`args.reasoning = None`) instead — otherwise a global `DEFAULT_EFFORT=low` would make every pipe to gpt-4.1/deepseek error. Provenance is tracked via `reasoning_was_defaulted`, mirroring the existing `model_was_defaulted` pattern. The gate runs *before* the `model_was_defaulted` status print so the printed `model:reasoning` line reflects the dropped value (prints bare model, no `:low`). Tests: `tests/test_cli_reasoning_gate.py` (the only main()-level tests; mock `_build_provider`/`_read_stdin_if_piped`/`_load_user_env_file`).
- **Token/temperature side-effects when a level is set:** Anthropic & OpenAI bump `max_tokens`/`max_output_tokens` (so reasoning doesn't starve the answer) and **drop `temperature`** (reasoning models reject a custom one). The direct Anthropic adapter also switches to `messages.stream()` when thinking is on (long generations outrun the non-streaming socket timeout). Gemini keeps temperature.
- **The gate asks ONE question: does this model have an effort knob at all?** Since every rung always resolves onto a non-empty vocabulary, "a level this model cannot take" no longer exists, and the short-lived per-level refusal (and the ambient-effort clamp that went with it) were both deleted. What remains: an explicit `-r` on a knobless model exits 2; an ambient `$DEFAULT_EFFORT` is dropped there instead, so a global `DEFAULT_EFFORT=low` never breaks a pipe to `gpt-4.1` or `grok-build-0.1`.

## Escape hatch: `--native-effort` (2026-08-04)

The normalised ladder is right for portability and wrong when effort is the
thing under study. Because `xhigh` is *defined* as "the most this model has", it
resolves to `max` on DeepSeek, GLM, Claude-5 and GPT-5.6 — so a benchmark
comparing "xhigh vs max" silently runs one arm twice, and no output distinguishes
them. That happened: a book-agent effort sweep was designed around `high` vs
`max` before anyone noticed `-r max` is not even accepted and `xhigh` already
meant `max`.

`--native-effort` passes `-r` through as the model's own value, untranslated.
Off by default. Consequences worth remembering:

- **argparse `choices` had to go.** The valid vocabulary depends on the flag,
  which argparse cannot consult, so `-r` is validated after parsing. The payoff:
  the error can name the right vocabulary for the mode in use, and a plain
  `-r max` now tells the user how to get what they obviously wanted.
- **A value the model lacks is refused before the payload is built.**
  `-r xhigh --native-effort` fails on `deepseek-v4-flash` (ladder `high|max`) and
  succeeds on `claude-opus-5`, which really has an `xhigh` below its `max`. That
  asymmetry IS the confusion the flag removes, so it is asserted in tests.
- **It will not inherit `$DEFAULT_EFFORT`** — an ambient portable rung fed to a
  provider as a native value would recreate the ambiguity.
- The knobless-model gate is unchanged.

`Request` carries both `reasoning` (the rung asked for) and `wire_effort` (what
is actually sent); only the latter changes under this flag. Assert on
`wire_effort` when testing what reached the provider.

**Agent guidance:** before running a benchmark, an effort sweep, or anything
where the effort must be reported exactly, say that the default ladder makes
results ambiguous and ask whether to use `--native-effort` — *before* spending
tokens. For ordinary one-shot calls the default is correct and this never comes
up. This belongs in the `gllm-cli` skill too; the skill dir is read-only, so it
has to be applied by hand.

Tests: `tests/test_cli_native_effort.py`.

## Reasoning is fully decoupled from WORK

Earlier this adapter had `WORK=1` force-max thinking on Azure Anthropic (`_force_work_env_thinking`). That was wrong — `WORK` is a provider-routing toggle (direct vs Azure, via `routing.effective_model`), not a reasoning lever. As of 2026-06-16 the forced-thinking mechanism is **removed**; reasoning is `--reasoning` only, on every adapter equally. See [[GOTCHA-azure-foundry-constraints]].

## Related
- [[GOTCHA-azure-foundry-constraints]] — the two Azure adapters; `WORK` no longer touches thinking.
- [[CONVENTIONS-multi-provider-routing]] — how each model name reaches the adapter that does the translation.
- [[ADR-provider-model-axis]] — the registry supplying `reasoning_efforts` and `thinking_dialect`.
