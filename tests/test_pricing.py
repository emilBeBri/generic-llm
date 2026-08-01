"""Pricing: model->book matching, provider-aware cost, and --usage emission.

All network-free by construction now: the book is a committed offline artifact
(llm-price-tracker), and every test injects a hand-built PriceBook through the
`_load_book` seam.
"""

from __future__ import annotations

import json

import gllm.cli as cli
import gllm.pricing as pricing
from gllm.domain import Response
from llm_price_tracker.models import STANDARD, ModelEntry, Price, PriceBook


def _book(models: dict[str, tuple]) -> PriceBook:
    """PriceBook from {id: (input, output, cache_read, cache_write)}."""
    return PriceBook(
        updated_at="2026-08-01",
        models={
            mid: ModelEntry(
                id=mid,
                vendor="test",
                tiers={STANDARD: Price(input=i, output=o, cache_read=cr, cache_write=cw)},
            )
            for mid, (i, o, cr, cw) in models.items()
        },
    )


# A trimmed fixture mirroring the real book (USD per 1M tokens). Note the ids:
# the book keys by what vendors PUBLISH — Anthropic page slugs use dots
# (`claude-haiku-4.5`) where gllm registry keys use dashes.
BOOK = _book({
    "claude-opus-4-8": (5, 25, None, None),
    "claude-haiku-4.5": (1, 5, 0.1, 1.25),
    "gemini-3-1-pro-preview": (2, 12, None, None),
    "gpt-5.1": (1.25, 10, 0.125, None),
})


def _use_book(monkeypatch, book=BOOK):
    monkeypatch.setattr(pricing, "_load_book", lambda: book)


# --- matching ---------------------------------------------------------------

def test_book_exact_id(monkeypatch):
    _use_book(monkeypatch)
    key, entry = pricing._book_entry("claude-opus-4-8")
    assert key == "claude-opus-4-8"
    assert entry["input"] == 5


def test_book_dot_vs_dash_folded_both_directions(monkeypatch):
    # gllm `gemini-3.1-pro-preview` -> book `gemini-3-1-pro-preview`, and
    # gllm `claude-haiku-4-5` -> book page-slug `claude-haiku-4.5`.
    _use_book(monkeypatch)
    assert pricing._book_entry("gemini-3.1-pro-preview")[0] == "gemini-3-1-pro-preview"
    assert pricing._book_entry("claude-haiku-4-5")[0] == "claude-haiku-4.5"


def test_unknown_model_is_none(monkeypatch):
    # GLM has no tracker source yet -> no book price, honestly null.
    _use_book(monkeypatch)
    assert pricing._book_entry("glm-5.2") is None


# --- cost (provider-aware) --------------------------------------------------

_OPUS = {"input": 5, "output": 25, "input_cached": None}
_GPT51 = {"input": 1.25, "output": 10, "input_cached": 0.125}
_GEMINI = {"input": 2, "output": 12, "input_cached": None}


def test_cost_simple_anthropic():
    # 1M in @ $5, 1M out @ $25 -> $30.
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert pricing.compute_cost("anthropic", _OPUS, usage) == 30.0


def test_cost_openai_subtracts_cached_input():
    # prompt_tokens INCLUDES cached: 1000 in (800 cached), 0 out.
    # (200 @ 1.25/1M) + (800 @ 0.125/1M) = 0.00025 + 0.0001 = 0.00035.
    usage = {"input_tokens": 1000, "output_tokens": 0, "cache_read_tokens": 800}
    assert pricing.compute_cost("openai", _GPT51, usage) == round(0.00025 + 0.0001, 6)


def test_cost_anthropic_does_not_subtract_cache_read():
    # Anthropic input_tokens EXCLUDES cache; cached read billed on top.
    # opus input_cached is null -> cache read billed at full input rate.
    usage = {"input_tokens": 1000, "output_tokens": 0, "cache_read_tokens": 500}
    # (1000 + 500) @ 5/1M = 0.0075
    assert pricing.compute_cost("anthropic", _OPUS, usage) == round(1500 * 5 / 1_000_000, 6)


def test_cost_anthropic_uses_published_cache_write_rate():
    # The book carries the vendor's real write premium; it must beat the 1.25x
    # guess. Here the published rate (10/M) is 2x the guessed 6.25/M.
    entry = dict(_OPUS, cache_write=10.0)
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 1_000_000}
    assert pricing.compute_cost("anthropic", entry, usage) == 10.0


def test_cost_anthropic_write_guess_backstops_overrides():
    # Override entries carry no cache_write -> the 1.25x multiplier applies.
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 1_000_000}
    assert pricing.compute_cost("anthropic", _OPUS, usage) == round(5 * 1.25, 6)


def test_cost_gemini_bills_reasoning_on_top():
    # Gemini output excludes thoughts: (100 + 50) @ 12/1M.
    usage = {"input_tokens": 0, "output_tokens": 100, "reasoning_tokens": 50}
    assert pricing.compute_cost("gemini", _GEMINI, usage) == round(150 * 12 / 1_000_000, 6)


def test_cost_unpriced_is_none():
    assert pricing.compute_cost("zai", None, {"input_tokens": 10}) is None


def test_price_report_picks_first_matching_candidate(monkeypatch):
    _use_book(monkeypatch)
    monkeypatch.setattr(pricing, "load_overrides", lambda: {})
    out = pricing.price_report("anthropic", ["nope-not-real", "claude-opus-4-8"],
                               {"input_tokens": 1_000_000, "output_tokens": 0})
    assert out["priced_as"] == "claude-opus-4-8"
    assert out["price_source"] == "book"
    assert out["cost_usd"] == 5.0


def test_price_report_never_raises(monkeypatch):
    def _boom():
        raise RuntimeError("sibling checkout missing")

    monkeypatch.setattr(pricing, "_load_book", _boom)
    monkeypatch.setattr(pricing, "load_overrides", lambda: {})
    out = pricing.price_report("openai", ["gpt-5.1"], {"input_tokens": 10})
    assert out == {"cost_usd": None, "priced_as": None, "price_source": "none"}


# --- local overrides --------------------------------------------------------

def _write(path, obj):
    import json as _j
    path.write_text(_j.dumps(obj), encoding="utf-8")


def test_overrides_overlay_wins_and_stub_skipped(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled.json"
    overlay = tmp_path / "overlay.json"
    _write(bundled, {
        "_comment": "ignored",
        "glm-5.2": {"input": None, "output": None},          # stub -> skipped
        "glm-old": {"input": 1.0, "output": 2.0},            # active
    })
    _write(overlay, {
        "glm-5.2": {"input": 0.6, "output": 2.2, "input_cached": 0.11},  # fills the stub
        "glm-old": {"input": 9.0, "output": 9.0},            # overlay wins
    })
    monkeypatch.setattr(pricing, "_bundled_overrides_path", lambda: bundled)
    monkeypatch.setattr(pricing, "_overlay_overrides_path", lambda: overlay)

    ov = pricing.load_overrides()
    assert "_comment" not in ov
    assert ov["glm-5.2"]["input"] == 0.6          # overlay activated it
    assert ov["glm-old"]["input"] == 9.0          # overlay overrode bundled


def test_override_beats_book_in_price_report(monkeypatch):
    _use_book(monkeypatch, _book({"glm-5.2": (99, 99, None, None)}))
    monkeypatch.setattr(pricing, "load_overrides",
                        lambda: {"glm-5.2": {"input": 0.6, "output": 2.2, "input_cached": 0.11}})
    out = pricing.price_report("zai", ["glm-5.2"],
                               {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert out["price_source"] == "override"
    assert out["priced_as"] == "glm-5.2"
    assert out["cost_usd"] == round(0.6 + 2.2, 6)


def test_bundled_prices_file_is_valid_json():
    # The shipped stub must always parse; filling it must not break loading.
    data = pricing._read_override_file(pricing._bundled_overrides_path())
    assert isinstance(data, dict)


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
    monkeypatch.setattr(pricing, "_load_book", lambda: BOOK)
    monkeypatch.setattr(pricing, "load_overrides", lambda: {})  # isolate from real override files
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
    # haiku-4-5 in gllm -> the book's page-slug claude-haiku-4.5 (1M in @ $1).
    assert rec["priced_as"] == "claude-haiku-4.5"
    assert rec["cost_usd"] == 1.0
    assert rec["price_source"] == "book"
