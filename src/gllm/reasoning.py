"""Abstract reasoning-effort ladder and per-provider translation.

gllm exposes one knob — `--reasoning low|medium|high|xhigh|max` — and each
adapter translates it to its provider-native wire shape. These functions are
pure (no SDK, no network) so they unit-test directly (see tests/test_reasoning.py).

Which rungs a given model actually accepts is registry data
(`ModelCaps.reasoning_efforts`), enforced by the CLI gate before anything here
runs: `max` exists on Claude Fable 5 / Opus 5 / 4.8 / 4.7 / 4.6 / Sonnet 5 / 4.6,
on GPT-5.6, and on GLM — while grok tops out at `high` and gpt-5.1 at `xhigh`.

Providers disagree on the shape, so `ModelCaps.thinking_dialect` names it:
  * `openai_effort` — OpenAI / Grok / Azure OpenAI (Responses API): an `effort`
    string, 1:1 with our ladder (the ladder was chosen to match its vocabulary).
  * `anthropic_adaptive` — Claude 4.6+ and the 5 family: `thinking.type=adaptive`
    (+ `display:summarized`, since their default flipped to `omitted`), graded by
    `output_config.effort`. These models REJECT the old enabled+budget shape.
  * `anthropic_budget` — Claude 4.5 and older: `thinking.type=enabled` with a
    `budget_tokens` int, plus a `max_tokens` floor (the budget must be strictly
    below max_tokens) and an unset temperature (extended thinking pins it to 1).
  * `gemini_budget` — a `thinking_budget` int (-1 = dynamic / model-capped).
  * `zai_effort` / `zai_thinking` — GLM: `thinking.type=enabled`, plus a
    `reasoning_effort` string on glm-5.2+ only.
  * `compat_effort` / `compat_thinking_flag` — OpenAI-compatible hosts: a bare
    `reasoning_effort` (Groq), or one alongside a top-level `thinking` flag
    (Regolo).
  * `None` — no control surface. DeepSeek reasons by default but exposes no
    knob; gated out upstream, never reaches here.
"""

from __future__ import annotations

LEVELS = ("low", "medium", "high", "xhigh", "max")


def _check(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(
            f"unknown reasoning level {level!r}; expected one of {', '.join(LEVELS)}"
        )
    return level


def openai_effort(level: str) -> str:
    """Map the ladder to an OpenAI/Grok Responses `reasoning.effort` string.

    Identity (with validation). Which rungs a model accepts is a registry
    question answered before we get here; a level the API doesn't know is a loud
    400, which is the intended fail-loud behaviour.
    """
    return _check(level)


# Anthropic budgets for the three lower rungs on the OLD enabled+budget
# interface (4.5 and earlier). xhigh/max are special-cased per family below.
_ANTHROPIC_BUDGETS = {"low": 8000, "medium": 16000, "high": 32000}
_ANTHROPIC_HEADROOM = 8000  # answer tokens reserved above the thinking budget


def anthropic_thinking(level: str, model: str, dialect: str | None = None) -> dict:
    """Translate the ladder to an Anthropic `thinking` block (+ a max_tokens floor).

    Returns ``{"thinking": <block>, "min_max_tokens": <int>}`` and, on the
    adaptive dialect, an ``"effort"`` string (= our ladder, 1:1). The caller sets
    ``kwargs["thinking"]``, raises ``max_tokens`` to at least ``min_max_tokens``,
    and drops temperature. When ``"effort"`` is present it is graded via
    ``output_config.effort`` on both the direct Anthropic API and Azure Foundry
    (both expose `output_config`).

    `dialect` comes from the model's registry row; pass it when you already have
    the caps in hand. Omitted, it is looked up — which is what stops a Claude 5
    model from silently getting the retired enabled+budget_tokens shape it
    rejects.
    """
    _check(level)
    m = model.lower()

    if dialect is None:
        from .models import caps_for

        dialect = caps_for(model, "anthropic").thinking_dialect

    if dialect == "anthropic_adaptive":
        return {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "effort": level,
            "min_max_tokens": 64000,
        }

    # 4.5 and older: the original enabled + budget_tokens interface. `max` is
    # only reachable here via the legacy guess for an unregistered name; treat
    # it as the top budget rung rather than crash.
    if level in ("xhigh", "max"):
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
# (the model self-budgets up to its cap), which we use for the top rungs.
_GEMINI_BUDGETS = {"low": 4096, "medium": 8192, "high": 16384, "xhigh": -1, "max": -1}


def gemini_thinking_budget(level: str, model: str) -> int:
    """Translate the ladder to a Gemini `thinking_budget` int. `model` is taken
    for future per-model clamping; the API rejects out-of-range budgets loudly."""
    _check(level)
    return _GEMINI_BUDGETS[level]


def zai_effort(level: str) -> str:
    """Translate the ladder to a GLM `reasoning_effort` string (glm-5.2+ only).

    Identity-with-validation, like `openai_effort`: GLM accepts our ladder values
    verbatim and collapses them itself (low/medium -> high, xhigh/max -> max).
    Only sent when `glm_supports_reasoning_effort(model)` is true.
    """
    return _check(level)


# OpenAI-compatible hosts (Groq, Regolo) publish only low/medium/high. The caps
# for those rows say so too, but clamp rather than 400 on a legacy-guessed name.
_COMPAT_CEILING = {"xhigh": "high", "max": "high"}


def compat_effort(level: str) -> str:
    """Translate the ladder to an OpenAI-compatible host's `reasoning_effort`."""
    _check(level)
    return _COMPAT_CEILING.get(level, level)
