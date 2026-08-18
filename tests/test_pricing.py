"""Pricing: model->book matching, provider-aware cost, and --usage emission.

All network-free by construction now: the book is a committed offline artifact
(llm-price-tracker), and every test injects a hand-built PriceBook through the
`_load_book` seam.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

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
    assert out == {
        "cost_usd": None,
        "priced_as": None,
        "price_source": "none",
        "price_window": None,
    }


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


# --- time-of-day (peak) pricing ---------------------------------------------
# DeepSeek bills 2x inside published UTC windows. The book models that
# structurally, so gllm resolves it per call rather than storing an hour.

_PEAK_WINDOWS = (("01:00", "04:00"), ("06:00", "10:00"))


def _peak_book():
    """A DeepSeek-shaped row: off-peak scalars plus a 2x peak variant, next to
    a flat-rate neighbour that the moment must not touch."""
    from llm_price_tracker.models import TimeWindow

    return PriceBook(
        updated_at="2026-08-18",
        models={
            "deepseek-v4-flash": ModelEntry(
                id="deepseek-v4-flash",
                vendor="deepseek",
                tiers={STANDARD: Price(
                    input=0.22, output=0.66, cache_read=0.007,
                    peak=Price(input=0.44, output=1.32, cache_read=0.014),
                    peak_windows=[TimeWindow(start=s, end=e) for s, e in _PEAK_WINDOWS],
                )},
            ),
            "gpt-5.1": ModelEntry(
                id="gpt-5.1", vendor="openai",
                tiers={STANDARD: Price(input=1.25, output=10)},
            ),
        },
    )


_1M = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
# Instants named against the fixture's OWN windows above, not a vendor's.
_IN_WINDOW = datetime(2026, 8, 18, 2, 0, tzinfo=UTC)   # inside 01:00-04:00
_BETWEEN_WINDOWS = datetime(2026, 8, 18, 5, 0, tzinfo=UTC)


def test_peak_moment_bills_the_peak_rate(monkeypatch):
    _use_book(monkeypatch, _peak_book())
    key, entry = pricing._book_entry("deepseek-v4-flash", _IN_WINDOW)
    assert entry["input"] == 0.44
    assert entry["output"] == 1.32
    assert entry["input_cached"] == 0.014   # cache hits double too
    assert entry["price_window"] == "peak"


def test_off_peak_moment_bills_the_scalar(monkeypatch):
    _use_book(monkeypatch, _peak_book())
    _, entry = pricing._book_entry("deepseek-v4-flash", _BETWEEN_WINDOWS)
    assert entry["input"] == 0.22
    assert entry["price_window"] == "off_peak"


def test_no_moment_prices_off_peak(monkeypatch):
    # The tracker's own default, and the only deterministic answer for a
    # caller that cannot name the moment.
    _use_book(monkeypatch, _peak_book())
    _, entry = pricing._book_entry("deepseek-v4-flash")
    assert entry["input"] == 0.22
    assert entry["price_window"] is None


def test_non_utc_moment_is_converted_not_read_hour_for_hour(monkeypatch):
    # 03:00+02:00 IS 01:00 UTC — inside the window. The book's contains()
    # compares .hour against UTC without converting, so failing to normalise
    # here reads hour 3 and silently halves the bill.
    _use_book(monkeypatch, _peak_book())
    shifted = datetime(2026, 8, 18, 3, 0, tzinfo=timezone(timedelta(hours=2)))
    _, entry = pricing._book_entry("deepseek-v4-flash", shifted)
    assert entry["price_window"] == "peak"

    # And the reverse: 03:00 UTC read as +02:00 would land outside.
    _, entry = pricing._book_entry(
        "deepseek-v4-flash",
        datetime(2026, 8, 18, 5, 0, tzinfo=timezone(timedelta(hours=2))),  # 03:00Z
    )
    assert entry["price_window"] == "peak"


def test_a_flat_rate_row_ignores_the_moment(monkeypatch):
    # Most vendors have no time-of-day rate; those rows must be untouched, and
    # must not claim a window they do not have.
    _use_book(monkeypatch, _peak_book())
    for at in (None, _IN_WINDOW, _BETWEEN_WINDOWS):
        _, entry = pricing._book_entry("gpt-5.1", at)
        assert entry["input"] == 1.25
        assert entry["price_window"] is None


def test_price_report_doubles_and_names_the_window(monkeypatch):
    _use_book(monkeypatch, _peak_book())
    monkeypatch.setattr(pricing, "load_overrides", lambda: {})

    off = pricing.price_report("deepseek", ["deepseek-v4-flash"], _1M, _BETWEEN_WINDOWS)
    peak = pricing.price_report("deepseek", ["deepseek-v4-flash"], _1M, _IN_WINDOW)

    assert off["cost_usd"] == round(0.22 + 0.66, 6)
    assert off["price_window"] == "off_peak"
    assert peak["cost_usd"] == round(0.44 + 1.32, 6)
    assert peak["price_window"] == "peak"
    assert peak["cost_usd"] == round(2 * off["cost_usd"], 6)


def test_an_override_flattens_the_hourly_rate_and_says_so(monkeypatch):
    # Overrides are consulted BEFORE the book and their schema has no time
    # dimension, so overriding an hourly-priced model pins it to one rate for
    # good. That is allowed — it is the escape hatch — but it must be visible:
    # price_source "override" beside a null window is the whole tell.
    _use_book(monkeypatch, _peak_book())
    monkeypatch.setattr(
        pricing, "load_overrides",
        lambda: {"deepseek-v4-flash": {"input": 0.22, "output": 0.66}},
    )
    out = pricing.price_report("deepseek", ["deepseek-v4-flash"], _1M, _IN_WINDOW)

    assert out["price_source"] == "override"
    assert out["price_window"] is None
    assert out["cost_usd"] == round(0.22 + 0.66, 6)   # NOT the peak rate


def test_the_real_book_deepseek_row_reaches_gllm_with_its_windows():
    # Against the committed book, not a fixture: proves the peak data actually
    # survives the flattening in _book_entry. The probe instant is derived from
    # the book's own windows — gllm never writes a vendor's clock time down.
    from llm_price_tracker.book import get_entry, load_book

    price = get_entry("deepseek-v4-flash", load_book()).standard
    if price.peak is None or not price.peak_windows:
        pytest.skip("the committed book publishes no peak window for deepseek")

    hour, minute = (int(x) for x in price.peak_windows[0].start.split(":"))
    at = datetime(2026, 8, 18, hour, minute, tzinfo=UTC) + timedelta(minutes=1)
    _, entry = pricing._book_entry("deepseek-v4-flash", at)
    _, off = pricing._book_entry("deepseek-v4-flash")

    assert entry["price_window"] == "peak"
    assert entry["input"] > off["input"]


# --- CLI emission of the window ---------------------------------------------

class _FakeDeepSeek:
    def generate(self, request):
        return Response(
            text="ok", model="deepseek-v4-flash", provider="deepseek",
            input_tokens=1_000_000, output_tokens=1_000_000,
        )


class _FrozenClock:
    """Stands in for cli.datetime so the dispatch stamp is deterministic."""

    fixed = _IN_WINDOW

    @classmethod
    def now(cls, tz=None):
        return cls.fixed if tz is None else cls.fixed.astimezone(tz)


def test_usage_record_reports_the_price_window(monkeypatch, capsys):
    _wire(monkeypatch)
    monkeypatch.setattr(cli, "_build_provider", lambda _name: _FakeDeepSeek())
    monkeypatch.setattr(pricing, "_load_book", lambda: _peak_book())
    monkeypatch.setattr(cli, "datetime", _FrozenClock)

    rc = cli.main(["--usage", "-m", "deepseek-v4-flash", "prompt"])
    assert rc == 0
    line = next(
        ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("gllm-usage ")
    )
    rec = json.loads(line[len("gllm-usage "):])

    assert rec["price_window"] == "peak"
    assert rec["cost_usd"] == round(0.44 + 1.32, 6)
