"""Abstract reasoning-effort ladder and per-provider translation.

gllm exposes one knob — `--reasoning low|medium|high|xhigh` — and each adapter
translates it to its provider-native wire shape. These functions are pure (no
SDK, no network) so they unit-test directly (see tests/test_reasoning.py).

Providers disagree on the shape:
  * OpenAI / Grok / Azure OpenAI (Responses API): an `effort` string, 1:1 with
    our ladder (the ladder was chosen to match OpenAI's vocabulary).
  * Anthropic / Azure Anthropic: a `thinking` block (a token budget, or adaptive
    at the top), plus a `max_tokens` floor — the budget must be strictly below
    max_tokens — and an unset temperature (extended thinking pins it to 1).
  * Gemini: a `thinking_budget` int (-1 = dynamic / model-capped).
  * DeepSeek: no control surface — gated out upstream, never reaches here.
"""

from __future__ import annotations

LEVELS = ("low", "medium", "high", "xhigh")


def _check(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(
            f"unknown reasoning level {level!r}; expected one of {', '.join(LEVELS)}"
        )
    return level


def openai_effort(level: str) -> str:
    """Map the ladder to an OpenAI/Grok Responses `reasoning.effort` string.

    Identity (with validation). `xhigh` only exists on newer models; older ones
    reject it with a loud 400, which is the intended fail-loud behaviour.
    """
    return _check(level)


# Anthropic budgets for the three lower rungs on the OLD enabled+budget
# interface (4.5 and earlier). xhigh is special-cased per family below.
_ANTHROPIC_BUDGETS = {"low": 8000, "medium": 16000, "high": 32000}
_ANTHROPIC_HEADROOM = 8000  # answer tokens reserved above the thinking budget


def _is_adaptive_family(model: str) -> bool:
    """Claude Opus/Sonnet 4.6/4.7/4.8 use the adaptive-thinking interface: they
    REQUIRE `thinking.type=adaptive` (+ `display:summarized`, since their default
    flipped to `omitted`) and reject the old `enabled`+`budget_tokens` shape.
    Effort is graded by `output_config.effort`, not a token budget."""
    m = model.lower()
    return "4-6" in m or "4-7" in m or "4-8" in m


def anthropic_thinking(level: str, model: str) -> dict:
    """Translate the ladder to an Anthropic `thinking` block (+ a max_tokens floor).

    Returns ``{"thinking": <block>, "min_max_tokens": <int>}`` and, for the
    adaptive family, an ``"effort"`` string (= our ladder, 1:1). The caller sets
    ``kwargs["thinking"]``, raises ``max_tokens`` to at least ``min_max_tokens``,
    and drops temperature. When ``"effort"`` is present it is graded via
    ``output_config.effort`` on both the direct Anthropic API and Azure Foundry
    (both expose `output_config`). 4.5 and older use ``enabled``+``budget_tokens``.
    """
    _check(level)
    m = model.lower()

    if _is_adaptive_family(model):
        return {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "effort": level,
            "min_max_tokens": 64000,
        }

    # 4.5 and older: the original enabled + budget_tokens interface.
    if level == "xhigh":
        budget, floor = (32000, 64000) if "4-5" in m else (16000, 32000)
        return {
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "min_max_tokens": floor,
        }
    budget = _ANTHROPIC_BUDGETS[level]
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "min_max_tokens": budget + _ANTHROPIC_HEADROOM,
    }


# Gemini thinking_budget per rung. Budgets are clamped per model; -1 = dynamic
# (the model self-budgets up to its cap), which we use for the top rung.
_GEMINI_BUDGETS = {"low": 4096, "medium": 8192, "high": 16384, "xhigh": -1}


def gemini_thinking_budget(level: str, model: str) -> int:
    """Translate the ladder to a Gemini `thinking_budget` int. `model` is taken
    for future per-model clamping; the API rejects out-of-range budgets loudly."""
    _check(level)
    return _GEMINI_BUDGETS[level]
