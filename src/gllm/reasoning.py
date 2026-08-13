"""The reasoning ladder: four stable rungs, normalised onto each provider.

gllm exposes exactly one knob with exactly four values —
`--reasoning low|medium|high|xhigh` — and that vocabulary never changes. This is
a deliberate contract for a scriptable CLI: a pipeline written against
`-r high` keeps working when you swap the model, even though the providers
underneath share almost no vocabulary at all.

They really don't agree. DeepSeek publishes `{high, max}`; GLM-5.2 honours only
those two as well; grok-4.5 takes `{low..xhigh}`; gpt-5.6 takes `{none..max}`;
Gemini takes `{minimal..high}` (or a token budget); Claude takes a thinking
block. An earlier attempt to make one literal ladder cover all of that failed —
hence this normalisation layer.

The rule (see `resolve_effort`):
  * `xhigh` ALWAYS means "the most this model has", whatever it is called there;
  * every other rung keeps its own name where the provider has it, and clamps to
    the nearest one where it doesn't.

So `-r low` is the cheapest available setting everywhere, and only the top rung
is ever remapped. `gllm.cli` prints a one-line notice when a translation
actually happens (silence with `-q`), because a level that silently means
something else is exactly the kind of quiet lie gllm refuses elsewhere.

`ModelCaps.native_efforts` holds each model's OWN vocabulary, cheapest first —
not gllm's rungs. `ModelCaps.thinking_dialect` says how the resolved value
reaches the wire:
  * `openai_effort` — OpenAI / Grok / Azure OpenAI (Responses): `reasoning.effort`.
  * `anthropic_adaptive` — Claude 4.6+ and the 5 family: `thinking.type=adaptive`
    (+ `display:summarized`, since their default flipped to `omitted`), graded by
    `output_config.effort`. These models REJECT the old enabled+budget shape.
  * `anthropic_budget` — Claude 4.5 and older: `thinking.type=enabled` with a
    `budget_tokens` int, plus a `max_tokens` floor (the budget must be strictly
    below max_tokens) and an unset temperature (thinking pins it to 1).
  * `gemini_budget` — a `thinking_budget` int (-1 = dynamic / model-capped).
  * `zai_effort` / `zai_thinking` — GLM: `thinking.type=enabled`, plus a
    `reasoning_effort` string on glm-5.2+ only.
  * `deepseek_effort` — `thinking.type=enabled` + `reasoning_effort`.
  * `compat_effort` / `compat_thinking_flag` — OpenAI-compatible hosts: a bare
    `reasoning_effort` (Groq), or one alongside a top-level `thinking` flag
    (Regolo).
  * `None` — no effort knob at all. The CLI gates these out before we get here.

These functions are pure (no SDK, no network) and unit-test directly; see
tests/test_reasoning.py.
"""

from __future__ import annotations

# gllm's own ladder. Four rungs, stable across every provider and model.
LEVELS = ("low", "medium", "high", "xhigh")

# Every effort word any provider uses, ordered cheapest to most expensive. Used
# only to measure distance when a rung has to clamp — it is NOT gllm's ladder
# and is never exposed to the user.
_RANK = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _check(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(
            f"unknown reasoning level {level!r}; expected one of {', '.join(LEVELS)}"
        )
    return level


def resolve_effort(level: str, native: tuple[str, ...]) -> str:
    """Translate a gllm rung into the model's own effort value.

    `native` is `ModelCaps.native_efforts` — the provider's vocabulary for this
    model, cheapest first. Raises if it is empty; a model with no effort knob
    must be refused by the caller, not silently given one.

        resolve_effort("xhigh", ("high", "max"))                 -> "max"
        resolve_effort("low",   ("high", "max"))                 -> "high"
        resolve_effort("low",   ("low", "medium", "high"))       -> "low"
        resolve_effort("xhigh", ("low", "medium", "high"))       -> "high"
    """
    _check(level)
    if not native:
        raise ValueError(f"model has no effort vocabulary to map {level!r} onto")
    # `xhigh` is defined as "the most this model offers", not as the literal
    # string "xhigh" — that is the whole point of the rung.
    if level == "xhigh":
        return native[-1]
    if level in native:
        return level
    want = _RANK.index(level)
    # Nearest by rank; ties break toward the cheaper value.
    return min(native, key=lambda n: (abs(_RANK.index(n) - want), _RANK.index(n)))


# --------------------------------------------------------------------------- #
# Dialect-specific wire shaping. Each takes an ALREADY-RESOLVED native value.
# --------------------------------------------------------------------------- #

# Anthropic budgets for the lower rungs on the OLD enabled+budget interface
# (4.5 and earlier). The top rung is special-cased per family below.
_ANTHROPIC_BUDGETS = {"low": 8000, "medium": 16000, "high": 32000}
_ANTHROPIC_HEADROOM = 8000  # answer tokens reserved above the thinking budget


def anthropic_thinking(effort: str, model: str, dialect: str) -> dict:
    """Build an Anthropic `thinking` block (+ a max_tokens floor) for a resolved
    effort value.

    Returns ``{"thinking": <block>, "min_max_tokens": <int>}`` and, on the
    adaptive dialect, the ``"effort"`` to grade with. The caller sets
    ``kwargs["thinking"]``, raises ``max_tokens`` to at least
    ``min_max_tokens``, and drops temperature. ``effort`` is graded via
    ``output_config.effort`` on both the direct API and Azure Foundry.

    `dialect` comes from the model's registry row. It is required rather than
    inferred: matching the strings 4-6/4-7/4-8 is what silently handed the whole
    Claude 5 line the retired budget shape it rejects.
    """
    if dialect == "anthropic_adaptive":
        return {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "effort": effort,
            "min_max_tokens": 64000,
        }

    # 4.5 and older: the original enabled + budget_tokens interface.
    if effort in ("xhigh", "max"):
        budget, floor = (32000, 64000) if "4-5" in model.lower() else (16000, 32000)
        return {
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "min_max_tokens": floor,
        }
    budget = _ANTHROPIC_BUDGETS.get(effort, _ANTHROPIC_BUDGETS["medium"])
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "min_max_tokens": budget + _ANTHROPIC_HEADROOM,
    }


# --------------------------------------------------------------------------- #
# Output-budget floors
#
# Reasoning tokens are spent from the OUTPUT budget, not alongside it — verified
# on Gemini, where max_output_tokens=120 with a 2048 thinking budget returned
# thoughts=115, candidates=1 and finish_reason=MAX_TOKENS. So a max_tokens sized
# for a plain answer starves the answer once thinking is on.
#
# This used to be a flat `max(request.max_tokens, 16000)` duplicated in seven
# adapters, which meant an explicit --max-tokens was silently overridden in
# seven places and reported wrongly by --usage. It resolves once in the CLI now.
# --------------------------------------------------------------------------- #

REASONING_MIN_OUTPUT = 16_000
_ANTHROPIC_DIALECTS = ("anthropic_adaptive", "anthropic_budget")


def min_output_tokens(model: str, effort: str, dialect: str | None) -> int:
    """gllm's PREFERRED output budget when reasoning is on — headroom, not law.

    Anthropic sizes it from the thinking budget; everyone else gets the flat
    floor. Falling below this makes for a cramped answer, not an error, so an
    explicit `--max-tokens` may go lower (loudly). See `hard_min_output_tokens`
    for the case where the API genuinely refuses.
    """
    if dialect in _ANTHROPIC_DIALECTS:
        return int(anthropic_thinking(effort, model, dialect)["min_max_tokens"])
    return REASONING_MIN_OUTPUT


def hard_min_output_tokens(model: str, effort: str, dialect: str | None) -> int | None:
    """A minimum the API ENFORCES, or None when there is none.

    Only Anthropic's old enabled+budget interface has one: `budget_tokens` must
    be **strictly less than** `max_tokens` (platform.claude.com, Messages API),
    so a smaller explicit value is a guaranteed 400 rather than a cramped
    answer — worth refusing locally instead of paying for the round trip. The
    adaptive dialect sends no budget, so it has no hard minimum and its
    `min_max_tokens` is purely gllm's headroom preference.
    """
    if dialect != "anthropic_budget":
        return None
    budget = anthropic_thinking(effort, model, dialect)["thinking"].get("budget_tokens")
    return budget + 1 if budget else None


# Gemini thinking_budget per native value. -1 = dynamic (the model self-budgets
# up to its cap), which is what the top rung resolves to.
_GEMINI_BUDGETS = {"minimal": 1024, "low": 4096, "medium": 8192, "high": 16384,
                   "xhigh": -1, "max": -1}


def gemini_thinking_budget(effort: str) -> int:
    """Translate a resolved Gemini effort value to a `thinking_budget` int."""
    return _GEMINI_BUDGETS.get(effort, -1)
