"""Per-call USD cost from the llm-price-tracker book.

Prices come from the llm-price-tracker package (an editable path dependency on
~/prog/prj/llm-price-tracker): a committed, daily cross-checked book of
first-party vendor prices, keyed by vendor model id, USD per 1M tokens. Its
read path is pure and offline — no network, ever, so a cold-cache offline run
can no longer stall 15s the way the old llm-prices.com feed fetch could. The
book is as fresh as the sibling checkout; its systemd timer raises an alarm
when a vendor page moves.

gllm already owns the token counts (gllm.usage), so it converts to dollars
here rather than pushing the job onto every caller.

Three pieces, separable for testing:
- _book_entry()  — gllm model name -> book entry (exact, then the tracker's
                   dot/dash-insensitive fallback). `_load_book` is the seam.
- compute_cost() — pure: provider-aware $ from an entry + token counts.
- load_overrides()/match_override() — the bundled + user-overlay gap layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Providers whose `input_tokens` EXCLUDES cached/again-billed tokens, and which
# bill cache *writes* (Anthropic 5-min cache ≈ 1.25× base input). Everywhere
# else, `input_tokens` already INCLUDES the cached read, so we subtract it.
_ANTHROPIC_PROVIDERS = {"anthropic", "azure_anthropic"}
_ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25


# --------------------------------------------------------------------------- #
# Book layer: the tracker's committed price book (offline, pydantic-only).
# --------------------------------------------------------------------------- #
def _load_book():
    """Test seam: monkeypatch me with a hand-built PriceBook. Lazy import so
    plain (non ``--usage``) runs never pay for pydantic."""
    from llm_price_tracker import load_book

    return load_book()


def _book_entry(model: str) -> tuple[str, dict] | None:
    """(book id, entry dict) for a model, or None.

    The tracker's get_entry does exact-id first, then a UNIQUE dot/dash-folded
    match (book Anthropic ids are page slugs like `claude-opus-4.6`; gllm keys
    are wire-style `claude-opus-4-6`). Namespaced keys (`groq:`, `regolo:`)
    can never resolve here — they are priced by the override layer.
    """
    if not model:
        return None
    from llm_price_tracker.book import get_entry

    entry = get_entry(model.strip(), _load_book())
    if entry is None or entry.standard is None:
        return None
    p = entry.standard
    return entry.id, {
        "input": p.input,
        "output": p.output,
        "input_cached": p.cache_read,
        "cache_write": p.cache_write,
    }


def _norm(s: str) -> str:
    # Sources mix separators: `gemini-3-1-pro-preview` vs gllm's `gemini-3.1-...`.
    return s.strip().lower().replace(".", "-")


# --------------------------------------------------------------------------- #
# Cost (pure): provider-aware $ from a feed entry + token counts.
# --------------------------------------------------------------------------- #
def _rate(value: Any) -> float:
    """USD per single token from a per-1M-token feed price (None -> 0)."""
    try:
        return float(value) / 1_000_000 if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def compute_cost(provider: str, entry: dict | None, usage: dict) -> float | None:
    """USD for one call. `usage` keys: input_tokens, output_tokens,
    cache_read_tokens, cache_write_tokens, reasoning_tokens. None if unpriced."""
    if not entry:
        return None
    in_rate = _rate(entry.get("input"))
    out_rate = _rate(entry.get("output"))
    cached = entry.get("input_cached")
    cached_rate = _rate(cached) if cached is not None else in_rate

    it = usage.get("input_tokens", 0) or 0
    ot = usage.get("output_tokens", 0) or 0
    cr = usage.get("cache_read_tokens", 0) or 0
    cw = usage.get("cache_write_tokens", 0) or 0
    rt = usage.get("reasoning_tokens", 0) or 0

    if provider in _ANTHROPIC_PROVIDERS:
        # input_tokens already excludes cache reads/writes; writes cost a
        # premium. The book carries the vendor's real cache-write rate; the
        # 1.25x guess only backstops entries without one (overrides).
        cache_write = entry.get("cache_write")
        write_rate = (
            _rate(cache_write)
            if cache_write is not None
            else _ANTHROPIC_CACHE_WRITE_MULTIPLIER * in_rate
        )
        input_cost = it * in_rate + cr * cached_rate + cw * write_rate
        output_cost = ot * out_rate
    elif provider == "gemini":
        # prompt_token_count includes cache; thoughts are billed ON TOP of output.
        input_cost = max(it - cr, 0) * in_rate + cr * cached_rate
        output_cost = (ot + rt) * out_rate
    else:
        # OpenAI-family / DeepSeek / GLM: prompt tokens include cache; reasoning
        # is already part of completion/output tokens.
        input_cost = max(it - cr, 0) * in_rate + cr * cached_rate
        output_cost = ot * out_rate

    return round(input_cost + output_cost, 6)


# --------------------------------------------------------------------------- #
# Local overrides: two-tier (bundled data/ + ~/.config/gllm/ overlay), matching
# gllm's schema/instruction layout. Overrides WIN over the book. The bundled
# file holds only gaps the book cannot price (GLM/Zhipu, groq:/regolo: rentals,
# Azure -dev deployments) — a bundled row shadowing a book-covered model is the
# stale-copy trap, and a registry test forbids it. The user overlay is the
# deliberate escape hatch and may override anything.
# --------------------------------------------------------------------------- #
def _bundled_overrides_path() -> Path:
    # pricing.py is at <repo>/src/gllm/pricing.py -> parents[2] is the repo root.
    return Path(__file__).resolve().parents[2] / "data" / "prices.json"


def _overlay_overrides_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".config"
    return root / "gllm" / "prices.json"


def _read_override_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_overrides() -> dict:
    """Merge bundled + user-overlay price overrides into {model_lower: entry}.

    Overlay wins per model. Keys starting with '_' are ignored (comments). An
    entry is only kept once it has numeric input AND output — an unfilled stub
    (null values) is skipped, so shipping a stub never fabricates a $0 price.
    """
    merged: dict = {}
    for path in (_bundled_overrides_path(), _overlay_overrides_path()):  # overlay last = wins
        for k, v in _read_override_file(path).items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            merged[k.strip().lower()] = v
    return {
        k: v for k, v in merged.items()
        if isinstance(v.get("input"), (int, float)) and isinstance(v.get("output"), (int, float))
    }


def match_override(model: str, overrides: dict) -> tuple[str, dict] | None:
    """(key, entry) for a model in the overrides, by exact then dot/dash-
    normalised name. None if absent."""
    if not model or not overrides:
        return None
    ml = model.strip().lower()
    if ml in overrides:
        return ml, overrides[ml]
    nm = _norm(model)
    for k, v in overrides.items():
        if _norm(k) == nm:
            return k, v
    return None


def price_report(provider: str, models: list[str], usage: dict) -> dict:
    """Convenience for the CLI: try local overrides first, then the price book;
    match the first model name that hits; compute cost. Never raises — pricing
    must not break the main output. Returns {cost_usd, priced_as, price_source}
    with price_source in {override, book, none}."""
    try:
        overrides = load_overrides()
        for m in models:
            hit = match_override(m, overrides)
            if hit:
                key, entry = hit
                return {
                    "cost_usd": compute_cost(provider, entry, usage),
                    "priced_as": key,
                    "price_source": "override",
                }
        for m in models:
            book_hit = _book_entry(m)
            if book_hit:
                key, entry = book_hit
                return {
                    "cost_usd": compute_cost(provider, entry, usage),
                    "priced_as": key,
                    "price_source": "book",
                }
        return {"cost_usd": None, "priced_as": None, "price_source": "none"}
    except Exception:
        return {"cost_usd": None, "priced_as": None, "price_source": "none"}
