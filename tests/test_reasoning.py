"""Unit coverage for the reasoning ladder and its per-provider mappings.

Pure functions only — no SDK, no network, same style as
test_work_mode_thinking.py.
"""

import pytest

from gllm.adapters._capabilities import supports_reasoning
from gllm.reasoning import (
    LEVELS,
    anthropic_thinking,
    gemini_thinking_budget,
    openai_effort,
)


# --- openai_effort -----------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_openai_effort_roundtrips(level):
    assert openai_effort(level) == level


def test_openai_effort_rejects_unknown():
    with pytest.raises(ValueError):
        openai_effort("max")


# --- anthropic_thinking ------------------------------------------------------


def test_anthropic_adaptive_family_uses_adaptive_plus_effort():
    # 4.6/4.7/4.8 reject enabled+budget; every rung is adaptive + an effort
    # string (graded via output_config.effort on the direct API).
    for level in ("low", "medium", "high", "xhigh"):
        for model in ("claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"):
            r = anthropic_thinking(level, model)
            assert r["thinking"] == {"type": "adaptive", "display": "summarized"}
            assert r["effort"] == level
            assert r["min_max_tokens"] == 64000


def test_anthropic_old_family_uses_enabled_budget_no_effort():
    # 4.5 / older keep the original enabled + budget_tokens interface, no effort.
    expected = {"low": 8000, "medium": 16000, "high": 32000}
    for level, budget in expected.items():
        r = anthropic_thinking(level, "claude-opus-4-5")
        assert r["thinking"] == {"type": "enabled", "budget_tokens": budget}
        assert "effort" not in r
        assert "display" not in r["thinking"]
        assert r["min_max_tokens"] > budget


def test_anthropic_xhigh_per_family():
    # 4.6/4.7/4.8 -> adaptive + summarized, max 64000.
    r = anthropic_thinking("xhigh", "claude-opus-4-8")
    assert r["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert r["min_max_tokens"] == 64000
    # 4.5 -> enabled budget 32000, max 64000.
    r45 = anthropic_thinking("xhigh", "claude-opus-4-5")
    assert r45["thinking"] == {"type": "enabled", "budget_tokens": 32000}
    assert r45["min_max_tokens"] == 64000
    # older -> enabled budget 16000, max 32000.
    r3 = anthropic_thinking("xhigh", "claude-haiku-3-5")
    assert r3["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert r3["min_max_tokens"] == 32000


# --- gemini_thinking_budget --------------------------------------------------


def test_gemini_budgets_increase_then_dynamic():
    assert gemini_thinking_budget("low", "gemini-3.1-pro-preview") == 4096
    assert gemini_thinking_budget("medium", "gemini-3.1-pro-preview") == 8192
    assert gemini_thinking_budget("high", "gemini-3.1-pro-preview") == 16384
    assert gemini_thinking_budget("xhigh", "gemini-3.1-pro-preview") == -1


# --- supports_reasoning ------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", "claude-opus-4-8", True),
        ("azure_anthropic", "claude-opus-4-8-dev", True),
        ("gemini", "gemini-3.1-pro-preview", True),
        ("openai", "gpt-5.1", True),
        ("openai", "o3", True),
        ("openai", "gpt-4o", False),
        ("azure_openai", "gpt-4o-dev", False),
        ("grok", "grok-4", True),
        ("deepseek", "deepseek-v4-flash", False),
    ],
)
def test_supports_reasoning_truth_table(provider, model, expected):
    assert supports_reasoning(provider, model) is expected
