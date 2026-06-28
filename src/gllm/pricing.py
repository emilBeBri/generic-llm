"""Per-call USD cost from the llm-prices.com feed.

Prices come from Simon Willison's llm-prices project, the aggregated feed served
at https://www.llm-prices.com/current-v1.json (the same source bebri-chat uses).
Shape:

    {"updated_at": "YYYY-MM-DD",
     "prices": [{"id", "vendor", "name", "input", "output", "input_cached"}, ...]}

All prices are USD per 1 million tokens; `input_cached` may be null. We fetch
live (stdlib urllib — no extra dependency) and cache to disk for 24h, falling
back to a stale cache when the network is down. gllm already owns the token
counts (gllm.usage), so it converts to dollars here rather than pushing the job
onto every caller.

Three pieces, separable for testing:
- load_prices()  — fetch + 24h cache + stale fallback. Touches the network.
- match_price()  — pure: gllm model name -> feed entry (exact / dot-normalised /
                   unique token-set, in that order).
- compute_cost() — pure: provider-aware $ from a feed entry + token counts.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

FEED_URL = "https://www.llm-prices.com/current-v1.json"
USER_AGENT = "gllm/llm-prices"
TIMEOUT_S = 15
MAX_RESPONSE_BYTES = 2_000_000
CACHE_TTL_S = 24 * 60 * 60

# Providers whose `input_tokens` EXCLUDES cached/again-billed tokens, and which
# bill cache *writes* (Anthropic 5-min cache ≈ 1.25× base input). Everywhere
# else, `input_tokens` already INCLUDES the cached read, so we subtract it.
_ANTHROPIC_PROVIDERS = {"anthropic", "azure_anthropic"}
_ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25


# --------------------------------------------------------------------------- #
# Data layer: fetch + cache + stale fallback (ported from bebri-chat).
# --------------------------------------------------------------------------- #
def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".cache"
    return root / "gllm" / "llm-prices-v1.json"


def _read_cache() -> dict | None:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_age_s() -> float | None:
    try:
        return time.time() - _cache_path().stat().st_mtime
    except OSError:
        return None


def _write_cache(payload: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # a cache we can't write is a perf hit, not an error


def _fetch_feed() -> dict:
    req = urllib.request.Request(FEED_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310 - fixed https URL
        data = r.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError(f"feed exceeds {MAX_RESPONSE_BYTES} bytes")
    payload = json.loads(data)
    if not isinstance(payload, dict) or not isinstance(payload.get("prices"), list):
        raise ValueError("unexpected feed shape")
    return payload


def load_prices(force_refresh: bool = False) -> tuple[list[dict], str, str | None]:
    """Return (prices, source, error). source ∈ {cache, network, stale-cache, none}."""
    age = _cache_age_s()
    if not force_refresh and age is not None and age < CACHE_TTL_S:
        cached = _read_cache()
        if cached is not None:
            return cached.get("prices", []), "cache", None
    try:
        payload = _fetch_feed()
    except Exception as e:  # network/parse failures all degrade the same way
        stale = _read_cache()
        if stale is not None:
            return stale.get("prices", []), "stale-cache", None
        return [], "none", f"price feed unavailable: {e}"
    _write_cache(payload)
    return payload.get("prices", []), "network", None


# --------------------------------------------------------------------------- #
# Matching (pure): gllm model name -> feed entry.
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    # Feed mixes separators: `gemini-3-1-pro-preview` vs gllm's `gemini-3.1-...`.
    return s.strip().lower().replace(".", "-")


def _token_set(s: str) -> frozenset[str]:
    # Feed sometimes reorders name vs version: `claude-4.5-haiku` vs gllm's
    # `claude-haiku-4-5`. A token set is order-insensitive.
    return frozenset(t for t in _norm(s).split("-") if t)


def match_price(model: str, prices: list[dict]) -> dict | None:
    if not model or not prices:
        return None
    ml = model.strip().lower()
    nm = _norm(model)
    mt = _token_set(model)

    by_id: dict[str, dict] = {}
    by_norm: dict[str, dict] = {}
    for p in prices:
        pid = str(p.get("id", ""))
        if not pid:
            continue
        by_id.setdefault(pid.lower(), p)
        by_norm.setdefault(_norm(pid), p)

    if ml in by_id:                       # 1. exact id
        return by_id[ml]
    if nm in by_norm:                     # 2. dot/dash-normalised
        return by_norm[nm]
    hits = [p for p in prices if _token_set(str(p.get("id", ""))) == mt]
    if len(hits) == 1:                    # 3. unique token-set (ambiguous -> skip)
        return hits[0]
    return None


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
        # input_tokens already excludes cache reads/writes; writes cost a premium.
        input_cost = it * in_rate + cr * cached_rate + cw * _ANTHROPIC_CACHE_WRITE_MULTIPLIER * in_rate
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
# gllm's schema/instruction layout. Overrides WIN over the feed — they fill gaps
# the feed lacks (GLM/Zhipu) and double as a manual fix for a mispriced model.
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
    """Convenience for the CLI: try local overrides first, then the feed; match
    the first model name that hits; compute cost. Never raises — pricing must not
    break the main output. Returns {cost_usd, priced_as, price_source}."""
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
        prices, source, _ = load_prices()
        entry = None
        for m in models:
            entry = match_price(m, prices)
            if entry:
                break
        return {
            "cost_usd": compute_cost(provider, entry, usage) if entry else None,
            "priced_as": entry.get("id") if entry else None,
            "price_source": source,
        }
    except Exception:
        return {"cost_usd": None, "priced_as": None, "price_source": "none"}
