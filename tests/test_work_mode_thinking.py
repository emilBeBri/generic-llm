"""WORK-mode adaptive-thinking branch coverage for Azure Anthropic.

We instantiate the provider without going through __init__ (which requires
credentials) and call the static-ish branch directly.
"""

from gllm.adapters.azure_anthropic import AzureAnthropicProvider


def _force(model: str) -> dict:
    kwargs: dict = {}
    AzureAnthropicProvider._force_work_env_thinking(kwargs, model)
    return kwargs


def test_4_8_uses_adaptive_thinking():
    kw = _force("claude-opus-4-8-dev")
    assert kw["thinking"]["type"] == "adaptive"
    assert kw["thinking"]["display"] == "summarized"
    assert kw["max_tokens"] == 64000


def test_4_7_and_4_6_unchanged():
    for m in ["claude-opus-4-7-dev", "claude-opus-4-6-dev"]:
        kw = _force(m)
        assert kw["thinking"]["type"] == "adaptive"


def test_4_5_still_fixed_budget():
    kw = _force("claude-opus-4-5-dev")
    assert kw["thinking"]["type"] == "enabled"
    assert "budget_tokens" in kw["thinking"]
