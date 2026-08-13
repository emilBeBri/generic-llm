"""The one place the output budget is decided: `cli._resolve_max_tokens`.

Nine adapters used to each apply their own `max(request.max_tokens, 16000)`,
which meant an explicit `--max-tokens` was silently overridden in nine places
and `--usage` reported the number the user asked for rather than the one sent.
These tests pin the three cases that replaced it: gllm's own default, an
explicit adequate value, and an explicit value too small for reasoning.
"""

from __future__ import annotations

import pytest

from gllm import cli
from gllm.reasoning import REASONING_MIN_OUTPUT


def resolve(explicit, provider="deepseek", model="deepseek-v4-pro", effort="", quiet=True):
    return cli._resolve_max_tokens(explicit, provider, model, effort, quiet=quiet)


# --- no flag: gllm chooses -------------------------------------------------

def test_default_uses_the_registry_ceiling_when_known():
    # claude-opus-5's documented max output, per platform.claude.com.
    assert resolve(None, "anthropic", "claude-opus-5") == 128_000


def test_default_falls_back_when_the_ceiling_is_unsourced():
    # deepseek rows carry no max_output — a guess there is a 400 or a truncation.
    assert resolve(None) == cli.DEFAULT_MAX_OUTPUT


def test_default_is_raised_to_the_reasoning_floor():
    assert resolve(None, effort="high") == REASONING_MIN_OUTPUT


def test_default_keeps_the_larger_of_ceiling_and_floor():
    # 128000 beats the 64000 anthropic adaptive floor; the ceiling wins.
    assert resolve(None, "anthropic", "claude-opus-5", effort="high") == 128_000


# --- explicit values ------------------------------------------------------

def test_explicit_value_is_sent_verbatim_without_reasoning():
    assert resolve(500) == 500


def test_explicit_value_above_the_floor_is_sent_verbatim():
    assert resolve(50_000, effort="high") == 50_000


def test_explicit_value_below_the_floor_is_honoured_not_overridden():
    assert resolve(500, effort="high") == 500


def test_honouring_a_low_explicit_value_warns(capsys):
    resolve(500, effort="high", quiet=False)
    err = capsys.readouterr().err
    assert "--max-tokens 500" in err
    assert str(REASONING_MIN_OUTPUT) in err
    assert "truncated" in err


def test_quiet_effort_silences_the_warning(capsys):
    resolve(500, effort="high", quiet=True)
    assert capsys.readouterr().err == ""


# --- the one hard API constraint ------------------------------------------

def test_anthropic_budget_dialect_refuses_an_illegal_value():
    """claude-opus-4-5 uses enabled+budget_tokens, where budget must be
    strictly less than max_tokens — a smaller value is a guaranteed 400."""
    with pytest.raises(ValueError, match="strictly less than max_tokens"):
        resolve(500, "anthropic", "claude-opus-4-5", effort="high")


def test_the_refusal_names_the_minimum_that_would_work():
    with pytest.raises(ValueError) as exc:
        resolve(500, "anthropic", "claude-opus-4-5", effort="high")
    # -r high on the budget dialect is a 32000-token thinking budget.
    assert "32001" in str(exc.value)


def test_adaptive_dialect_has_no_hard_minimum_so_it_only_warns(capsys):
    # Claude 5 sends no budget_tokens, so its 64000 is gllm's headroom
    # preference, not something the API enforces.
    assert resolve(500, "anthropic", "claude-opus-5", effort="high", quiet=False) == 500
    assert "may be truncated" in capsys.readouterr().err


def test_a_value_above_the_hard_minimum_but_below_headroom_is_allowed(capsys):
    """-r high on the budget dialect is a 32000 budget + 8000 answer headroom,
    so 32001..39999 is legal for the API but cramped by gllm's reckoning:
    honoured, and said out loud."""
    got = resolve(35_000, "anthropic", "claude-opus-4-5", effort="high", quiet=False)
    assert got == 35_000
    assert "40000" in capsys.readouterr().err


# --- no reasoning, no floor ----------------------------------------------

def test_without_reasoning_there_is_no_floor_at_all():
    assert resolve(1, "anthropic", "claude-opus-4-5") == 1
