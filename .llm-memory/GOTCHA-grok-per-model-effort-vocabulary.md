# xAI's reasoning_effort support is per-model and the docs get it wrong

Probed against the live API 2026-07-29 (`reasoning={"effort": ...}` on
`POST /v1/responses`). Every row here was verified by an actual request, not
read off a page.

`#gotcha` `#grok` `#model-registry` `#reasoning`

| model | none | low | medium | high | xhigh | max |
|---|---|---|---|---|---|---|
| `grok-4.5` | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `grok-4.3` | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `grok-4.20-multi-agent-0309` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `grok-4.20-0309-reasoning` | — parameter rejected — |
| `grok-4.20-0309-non-reasoning` | — parameter rejected — |
| `grok-build-0.1` | — parameter rejected — |

The three rejecting models return
`400 invalid-argument: Model <id> does not support parameter reasoningEffort.`

## Why the docs are not usable here

`docs_x_ai_developers_model_capabilities_text_reasoning.md` documents the
parameter for **`grok-4.5` only**, listing `low|medium|high` and stating
"Reasoning cannot be disabled". Taken literally that is wrong in both
directions: `grok-4.5` also accepts `xhigh`, and `grok-4.3` still accepts
`none`. Meanwhile each model's own page says "**Reasoning:** Yes" — including
the three that reject the parameter outright.

So the docs answer "does it reason?" while the registry needs to answer "can
its reasoning be **graded**?" Those differ, and only the second belongs in
`ModelCaps.reasoning_efforts`.

## The trap

`grok-4.20-0309-reasoning` is a reasoning model, reasons perfectly well with no
`reasoning` key at all, and 400s the moment you try to grade it — structurally
identical to DeepSeek. A name containing the word "reasoning" says nothing
about the control surface. Do not infer either way; probe.

Corollary for [[ADR-reasoning-effort-ladder]]: `max` reaches exactly one xAI
model, the multi-agent one, where it grades **agent count** (4 vs 16) rather
than depth.

## Both codebases had this wrong

gllm gave four of six grok rows a `(low, medium, high)` preset — wrong on
`grok-4.5`/`grok-4.3` (missing `xhigh`), wrong on multi-agent (missing `max`),
and actively broken on `grok-build-0.1` and `grok-4.20-0309-reasoning`, which
would have 400'd on any `-r`.

bebri-chat's `openai_model_capabilities.supports_reasoning_effort` special-cased
only `'build' in m`, so it too sent an effort to both `grok-4.20-0309-*`
variants. It was, however, right about `grok-build-0.1` where gllm was wrong —
the two registries disagreed and each held one half of the truth. That is the
argument for probing rather than porting.

See [[ADR-provider-model-axis]] for why capability facts live on registry rows
at all.
