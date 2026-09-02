# Gemini: gllm sends `thinkingBudget`, which Google now documents as legacy

`adapters/gemini.py` sets `generation_config["thinkingConfig"] =
{"thinkingBudget": gemini_thinking_budget(request.wire_effort)}` for every
Gemini row, mapping the ladder to ints (4096 / 8192 / 16384 / -1 dynamic; see
[[ADR-reasoning-effort-ladder]]). That was the whole API when the adapter was
written. It no longer is.

Google's current thinking doc (corpus refreshed 2026-09-02) splits the two
parameters cleanly:

- The **`thinkingLevel`** table lists every Gemini 3.x row — `minimal` / `low`
  / `medium` / `high`, per-model, and the rows disagree with each other.
- The **`thinkingBudget`** table lists **only the 2.5 series** (2.5 Pro, 2.5
  Flash, 2.5 Flash-Lite, Robotics-ER, the 2.5 live audio preview). No Gemini 3
  row appears in it at all.
- Verbatim note on the page: *"Use the `thinkingLevel` parameter with Gemini 3
  models. While `thinkingBudget` is accepted for backwards compatibility, using
  it with Gemini 3 Pro may result in unexpected performance."*

So gllm is on a supported-but-deprecated path for the ten `gemini-3.*` rows,
with an explicit performance warning on Pro. Not broken — measurements in
[[CONVENTIONS-one-shot-workload-no-cache]] were taken over this path and the
`-r` cliff they found was real — but it is the legacy wire shape, and
"backwards compatibility" is a clock.

## Why this has not bitten us the way it bit bebri-chat

`_GEMINI.native_efforts` is `("low", "medium", "high", "xhigh")` and the ladder
has no `minimal` or `off` rung, so **gllm structurally cannot send `minimal`**.
That is the parameter value newer Flash models reject outright:
`gemini-3.8-flash`, `gemini-3.7-flash` and `gemini-3.1-pro-preview` all 400 on
it, while 3.5/3.6 accept it. bebri-chat had to add per-model caps rows to stop
its picker offering `minimal` (its
`.llm-memory/gemini-per-model-thinking-level-vocabulary.md`); gllm needs no
equivalent split, and one shared `_GEMINI` preset stays correct for every row
including 3.8.

## The blocker if we do migrate

`thinkingLevel` has no equivalent for `xhigh`. The ladder's top rung means "the
most this model has", and on Gemini that is `thinkingBudget = -1` — *dynamic*,
model-chosen, potentially above `high`. Levels stop at `high`. Migrating
therefore forces a decision the budget path never had to make: collapse `xhigh`
onto `high` and lose dynamic thinking, or keep the budget int for that one rung
and send two different wire shapes. Not decided; not urgent while backwards
compatibility holds.

**Unverified:** no Gemini key is reachable from the claude-jail, so
`gemini-3.8-flash` has never been exercised live from gllm — the registry row
is sourced from the model page, not from a call. The `thinkingBudget` path on
3.8 specifically is likewise untested.

`#gotcha` `#gemini` `#reasoning` `#deprecation`
