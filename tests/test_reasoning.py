"""Unit coverage for the reasoning ladder and its per-provider normalisation.

Pure functions only — no SDK, no network.

The contract under test: gllm exposes exactly four rungs, and
`resolve_effort` maps them onto whatever vocabulary a model actually has.
`xhigh` always means "the most this model offers"; every other rung keeps its
own name where the provider has it and clamps to the nearest where it doesn't.
"""

import pytest

from gllm.adapters._capabilities import native_efforts, supports_reasoning
from gllm.models import MODELS
from gllm.reasoning import (
    LEVELS,
    anthropic_thinking,
    gemini_thinking_budget,
    resolve_effort,
)


def test_ladder_is_four_stable_rungs():
    # The scriptable contract: a pipeline written against `-r high` must keep
    # working when the model changes, so this vocabulary does not grow.
    assert LEVELS == ("low", "medium", "high", "xhigh")


# --- resolve_effort ----------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "native", "expected"),
    [
        # xhigh is always the top rung, whatever it is called there.
        ("xhigh", ("high", "max"), "max"),
        ("xhigh", ("low", "medium", "high"), "high"),
        ("xhigh", ("low", "medium", "high", "xhigh"), "xhigh"),
        ("xhigh", ("none", "low", "medium", "high", "xhigh", "max"), "max"),
        # Name match wins when the provider has the word.
        ("low", ("low", "medium", "high", "xhigh"), "low"),
        ("medium", ("none", "low", "medium", "high"), "medium"),
        ("high", ("high", "max"), "high"),
        # Otherwise clamp to the nearest by rank...
        ("low", ("high", "max"), "high"),
        ("medium", ("high", "max"), "high"),
        # ...preferring the cheaper of two equidistant options.
        ("medium", ("low", "high"), "low"),
    ],
)
def test_resolve_effort(level, native, expected):
    assert resolve_effort(level, native) == expected


def test_resolve_effort_rejects_unknown_rung():
    with pytest.raises(ValueError):
        resolve_effort("max", ("high", "max"))  # 'max' is not a gllm rung


def test_resolve_effort_refuses_empty_vocabulary():
    # A knobless model must be refused by the caller, never handed an effort.
    with pytest.raises(ValueError):
        resolve_effort("high", ())


def test_low_is_never_upgraded():
    """`-r low` must stay the cheapest thing a model offers, everywhere.

    This is the property that makes the ladder safe to leave in a script: the
    cheap rung never silently becomes an expensive one.
    """
    for key, spec in MODELS.items():
        native = spec.caps.native_efforts
        if not native:
            continue
        got = resolve_effort("low", native)
        assert native.index(got) == 0 or got == "low", f"{key}: low -> {got}"


# --- the per-provider matrix the design was chosen for -----------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # gllm rung ->            low       medium    high      xhigh
        ("gpt-5.6", ("low", "medium", "high", "max")),
        ("gpt-5.1", ("low", "medium", "high", "xhigh")),
        ("deepseek-v4-pro", ("high", "high", "high", "max")),
        ("deepseek-v4-flash", ("high", "high", "high", "max")),
        ("grok-4.5", ("low", "medium", "high", "xhigh")),
        ("grok-4.20-multi-agent-0309", ("low", "medium", "high", "max")),
        ("glm-5.2", ("high", "high", "high", "max")),
        ("claude-opus-5", ("low", "medium", "high", "max")),
        ("claude-opus-4-5", ("low", "medium", "high", "xhigh")),
        ("o3", ("low", "medium", "high", "high")),
        ("groq:openai/gpt-oss-120b", ("low", "medium", "high", "high")),
        ("kimi-k3", ("low", "low", "high", "max")),
        ("kimi-k2.6", ("high", "high", "high", "high")),
    ],
)
def test_resolution_matrix(model, expected):
    native = MODELS[model].caps.native_efforts
    got = tuple(resolve_effort(level, native) for level in LEVELS)
    assert got == expected


# --- anthropic_thinking ------------------------------------------------------


def test_adaptive_dialect_uses_adaptive_block_and_grades_by_effort():
    for effort in ("low", "medium", "high", "max"):
        r = anthropic_thinking(effort, "claude-opus-5", "anthropic_adaptive")
        assert r["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert r["effort"] == effort
        assert r["min_max_tokens"] == 64000


def test_claude_5_family_is_adaptive_via_the_registry():
    """Regression: matching the strings 4-6/4-7/4-8 silently handed the whole
    Claude 5 line the retired enabled+budget_tokens shape, which it rejects."""
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5"):
        assert MODELS[model].caps.thinking_dialect == "anthropic_adaptive"


def test_budget_dialect_uses_enabled_budget_no_effort():
    expected = {"low": 8000, "medium": 16000, "high": 32000}
    for effort, budget in expected.items():
        r = anthropic_thinking(effort, "claude-opus-4-5", "anthropic_budget")
        assert r["thinking"] == {"type": "enabled", "budget_tokens": budget}
        assert "effort" not in r
        assert r["min_max_tokens"] > budget


def test_budget_dialect_top_rung_per_family():
    r45 = anthropic_thinking("xhigh", "claude-opus-4-5", "anthropic_budget")
    assert r45["thinking"] == {"type": "enabled", "budget_tokens": 32000}
    assert r45["min_max_tokens"] == 64000
    r3 = anthropic_thinking("xhigh", "claude-haiku-3-5", "anthropic_budget")
    assert r3["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert r3["min_max_tokens"] == 32000


# --- gemini ------------------------------------------------------------------


def test_gemini_budgets_increase_then_go_dynamic():
    assert gemini_thinking_budget("low") == 4096
    assert gemini_thinking_budget("medium") == 8192
    assert gemini_thinking_budget("high") == 16384
    # The top rung resolves to `xhigh` for Gemini, i.e. dynamic self-budgeting.
    assert gemini_thinking_budget("xhigh") == -1


# --- supports_reasoning is now ONE question ----------------------------------


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", "claude-opus-5", True),
        ("azure_anthropic", "claude-opus-4-8-dev", True),
        ("gemini", "gemini-3.1-pro-preview", True),
        ("openai", "gpt-5.1", True),
        ("openai", "o3", True),
        ("openai", "gpt-4o", False),
        ("azure_openai", "gpt-4o-dev", False),
        ("grok", "grok-4.5", True),
        # Reasons, but cannot be graded — 400s on the parameter.
        ("grok", "grok-build-0.1", False),
        ("grok", "grok-4.20-0309-reasoning", False),
        # DeepSeek V4 DOES have an effort knob; gllm used to deny it.
        ("deepseek", "deepseek-v4-pro", True),
        ("zai", "glm-5.2", True),
        ("zai", "glm-4.6", True),
        ("zai", "glm-ocr", False),
        ("zai", "glm-4-32b-0414-128k", False),
        ("kimi", "kimi-k3", True),
        ("kimi", "kimi-k2.6", True),
        ("kimi", "kimi-k2.7-code", False),
    ],
)
def test_supports_reasoning_truth_table(provider, model, expected):
    assert supports_reasoning(provider, model) is expected
    assert bool(native_efforts(provider, model)) is expected
