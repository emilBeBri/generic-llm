"""Token-usage extraction, per provider.

At Response-build time each adapter has its provider's native usage object in
scope. These mappers normalise that object into gllm's common fields
(input/output/cache/reasoning tokens) while preserving the provider-native
numbers verbatim in `usage_raw`.

Two layers on purpose:
- The normalised fields are a lowest-common-denominator view. Their semantics
  follow each provider's own definitions, which do NOT fully agree: OpenAI and
  Anthropic fold reasoning tokens into `output_tokens`; Gemini reports thinking
  separately (`thoughts_token_count`) on top of `candidates_token_count`. So
  `reasoning_tokens` is "what the provider broke out separately", not a portable
  share of output.
- `usage_raw` is the ground truth — the provider's own usage dict, untouched.
  For exact per-model cost accounting, bill from `usage_raw`; use the normalised
  fields for rough cross-provider comparison.

Each mapper returns a dict whose keys match Response fields, so an adapter splats:
    Response(text=..., model=..., provider=..., raw=resp, **from_openai_chat(resp.usage))
"""

from __future__ import annotations

from typing import Any

_ZERO = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "reasoning_tokens": 0,
    "usage_raw": {},
}


def _i(obj: Any, name: str) -> int:
    """Read an int attribute, coercing None/missing/non-int to 0."""
    v = getattr(obj, name, 0)
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _to_plain(obj: Any) -> dict:
    """Best-effort convert a provider usage object to a JSON-able dict.

    The provider SDKs return pydantic models (OpenAI/Anthropic) or pydantic-ish
    proto wrappers (Gemini); `model_dump`/`to_dict`/`dict` cover them. Fall back
    to scraping public scalar attributes so we never raise on an unknown shape.
    """
    if obj is None:
        return {}
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                d = fn()
            except Exception:
                continue
            if isinstance(d, dict):
                return _jsonable(d)
    out: dict = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


def _jsonable(value: Any) -> Any:
    """Recursively drop anything that isn't a JSON scalar/list/dict."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()
                if isinstance(v, (int, float, str, bool, list, dict)) or v is None}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def from_anthropic(usage: Any) -> dict:
    """Anthropic `msg.usage`: cache split into read vs (pricier) creation.
    Thinking tokens are folded into output_tokens; not reported separately."""
    if usage is None:
        return dict(_ZERO)
    return {
        "input_tokens": _i(usage, "input_tokens"),
        "output_tokens": _i(usage, "output_tokens"),
        "cache_read_tokens": _i(usage, "cache_read_input_tokens"),
        "cache_write_tokens": _i(usage, "cache_creation_input_tokens"),
        "reasoning_tokens": 0,
        "usage_raw": _to_plain(usage),
    }


def from_openai_responses(usage: Any) -> dict:
    """OpenAI Responses API: `input_tokens` / `output_tokens`, with
    `input_tokens_details.cached_tokens` and `output_tokens_details.
    reasoning_tokens` (reasoning is a subset of output_tokens)."""
    if usage is None:
        return dict(_ZERO)
    itd = getattr(usage, "input_tokens_details", None)
    otd = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": _i(usage, "input_tokens"),
        "output_tokens": _i(usage, "output_tokens"),
        "cache_read_tokens": _i(itd, "cached_tokens"),
        "cache_write_tokens": 0,
        "reasoning_tokens": _i(otd, "reasoning_tokens"),
        "usage_raw": _to_plain(usage),
    }


def from_openai_chat(usage: Any) -> dict:
    """OpenAI Chat Completions shape (also Z.AI/GLM): `prompt_tokens` /
    `completion_tokens`, with `prompt_tokens_details.cached_tokens` and
    `completion_tokens_details.reasoning_tokens` when present."""
    if usage is None:
        return dict(_ZERO)
    ptd = getattr(usage, "prompt_tokens_details", None)
    ctd = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": _i(usage, "prompt_tokens"),
        "output_tokens": _i(usage, "completion_tokens"),
        "cache_read_tokens": _i(ptd, "cached_tokens"),
        "cache_write_tokens": 0,
        "reasoning_tokens": _i(ctd, "reasoning_tokens"),
        "usage_raw": _to_plain(usage),
    }


def from_gemini(usage: Any) -> dict:
    """Gemini `usage_metadata`: `prompt_token_count` / `candidates_token_count`,
    with `cached_content_token_count` and `thoughts_token_count`. Note thinking
    is billed on TOP of candidates here, so output_tokens excludes it — see
    module docstring."""
    if usage is None:
        return dict(_ZERO)
    return {
        "input_tokens": _i(usage, "prompt_token_count"),
        "output_tokens": _i(usage, "candidates_token_count"),
        "cache_read_tokens": _i(usage, "cached_content_token_count"),
        "cache_write_tokens": 0,
        "reasoning_tokens": _i(usage, "thoughts_token_count"),
        "usage_raw": _to_plain(usage),
    }


def from_deepseek(usage: Any) -> dict:
    """DeepSeek (OpenAI-compatible) reports prompt cache as hit/miss counts;
    a hit is input served from cache."""
    if usage is None:
        return dict(_ZERO)
    ctd = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": _i(usage, "prompt_tokens"),
        "output_tokens": _i(usage, "completion_tokens"),
        "cache_read_tokens": _i(usage, "prompt_cache_hit_tokens"),
        "cache_write_tokens": 0,
        "reasoning_tokens": _i(ctd, "reasoning_tokens"),
        "usage_raw": _to_plain(usage),
    }
