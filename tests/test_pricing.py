"""Pricing: model->feed matching, provider-aware cost, and --usage emission.

All network-free: the matching/cost functions are pure (fed a fixture price
list), and the CLI test monkeypatches load_prices so no fetch happens.
"""

from __future__ import annotations

import json

import gllm.cli as cli
import gllm.pricing as pricing
from gllm.domain import Response

# A trimmed fixture mirroring the real feed shape (USD per 1M tokens).
PRICES = [
    {"id": "claude-opus-4-8", "vendor": "anthropic", "input": 5, "output": 25, "input_cached": None},
    {"id": "claude-4.5-haiku", "vendor": "anthropic", "input": 1, "output": 5, "input_cached": None},
    {"id": "gemini-3-1-pro-preview", "vendor": "google", "input": 2, "output": 12, "input_cached": None},
    {"id": "gpt-5.1", "vendor": "openai", "input": 1.25, "output": 10, "input_cached": 0.125},
    {"id": "deepseek-v4-pro", "vendor": "deepseek", "input": 1.74, "output": 3.48, "input_cached": 0.145},
]


# --- matching ---------------------------------------------------------------

def test_match_exact_id():
    assert pricing.match_price("claude-opus-4-8", PRICES)["id"] == "claude-opus-4-8"


def test_match_dot_vs_dash_normalised():
    # gllm `gemini-3.1-pro-preview` -> feed `gemini-3-1-pro-preview`.
    assert pricing.match_price("gemini-3.1-pro-preview", PRICES)["id"] == "gemini-3-1-pro-preview"


def test_match_reordered_tokens():
    # gllm `claude-haiku-4-5` -> feed `claude-4.5-haiku` (token-set).
    assert pricing.match_price("claude-haiku-4-5", PRICES)["id"] == "claude-4.5-haiku"


def test_unknown_model_is_none():
    # GLM is absent from the llm-prices feed -> no price, honestly null.
    assert pricing.match_price("glm-5.2", PRICES) is None


# --- cost (provider-aware) --------------------------------------------------

def test_cost_simple_anthropic():
    # 1M in @ $5, 1M out @ $25 -> $30.
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    entry = pricing.match_price("claude-opus-4-8", PRICES)
    assert pricing.compute_cost("anthropic", entry, usage) == 30.0


def test_cost_openai_subtracts_cached_input():
    # prompt_tokens INCLUDES cached: 1000 in (800 cached), 0 out.
    # (200 @ 1.25/1M) + (800 @ 0.125/1M) = 0.00025 + 0.0001 = 0.00035.
    usage = {"input_tokens": 1000, "output_tokens": 0, "cache_read_tokens": 800}
    entry = pricing.match_price("gpt-5.1", PRICES)
    assert pricing.compute_cost("openai", entry, usage) == round(0.00025 + 0.0001, 6)


def test_cost_anthropic_does_not_subtract_cache_read():
    # Anthropic input_tokens EXCLUDES cache; cached read billed on top.
    # opus input_cached is null -> cache read billed at full input rate.
    usage = {"input_tokens": 1000, "output_tokens": 0, "cache_read_tokens": 500}
    entry = pricing.match_price("claude-opus-4-8", PRICES)
    # (1000 + 500) @ 5/1M = 0.0075
    assert pricing.compute_cost("anthropic", entry, usage) == round(1500 * 5 / 1_000_000, 6)


def test_cost_gemini_bills_reasoning_on_top():
    # Gemini output excludes thoughts: (100 + 50) @ 12/1M.
    usage = {"input_tokens": 0, "output_tokens": 100, "reasoning_tokens": 50}
    entry = pricing.match_price("gemini-3.1-pro-preview", PRICES)
    assert pricing.compute_cost("gemini", entry, usage) == round(150 * 12 / 1_000_000, 6)


def test_cost_unpriced_is_none():
    assert pricing.compute_cost("zai", None, {"input_tokens": 10}) is None


def test_price_report_picks_first_matching_candidate(monkeypatch):
    monkeypatch.setattr(pricing, "load_prices", lambda *a, **k: (PRICES, "cache", None))
    out = pricing.price_report("anthropic", ["nope-not-real", "claude-opus-4-8"],
                               {"input_tokens": 1_000_000, "output_tokens": 0})
    assert out["priced_as"] == "claude-opus-4-8"
    assert out["price_source"] == "cache"
    assert out["cost_usd"] == 5.0


# --- CLI emission -----------------------------------------------------------

class _FakeProvider:
    def generate(self, request):
        return Response(
            text="ok", model="claude-haiku-4-5", provider="anthropic",
            input_tokens=1_000_000, output_tokens=0,
        )


def _wire(monkeypatch):
    monkeypatch.setattr(cli, "_load_user_env_file", lambda *_: None)
    monkeypatch.setattr(cli, "_build_provider", lambda _name: _FakeProvider())
    monkeypatch.setattr(cli, "_read_stdin_if_piped", lambda: "hej")
    monkeypatch.setattr(pricing, "load_prices", lambda *a, **k: (PRICES, "cache", None))
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_EFFORT", raising=False)
    monkeypatch.delenv("WORK", raising=False)
    monkeypatch.delenv("WORK_ENV", raising=False)


def test_usage_record_includes_cost(monkeypatch, capsys):
    _wire(monkeypatch)
    rc = cli.main(["--usage", "-m", "claude-haiku-4-5", "prompt"])
    assert rc == 0
    line = next(ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("gllm-usage "))
    rec = json.loads(line[len("gllm-usage "):])
    # haiku-4-5 in gllm -> claude-4.5-haiku in feed (1M in @ $1).
    assert rec["priced_as"] == "claude-4.5-haiku"
    assert rec["cost_usd"] == 1.0
    assert rec["price_source"] == "cache"
