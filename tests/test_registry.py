"""Invariants for the provider/model registries.

These are the tests that stop `providers.PROVIDERS` and `models.MODELS` drifting
apart. A registry is only worth having if something enforces its shape — an
orphaned provider tag or an alt_model pointing at a deleted key is a silent
runtime error otherwise.

Pure data checks: no SDK, no network.
"""

from __future__ import annotations

import pytest

from gllm.models import MODELS, ModelCaps, caps_for, spec_for, wire_id_for
from gllm.providers import DISCOVERABLE_PROVIDERS, LISTABLE_PROVIDERS, PROVIDERS
from gllm.reasoning import _RANK

_OPENAI_FAMILY = {"openai", "azure_openai", "grok"}
_AZURE = {"azure_openai", "azure_anthropic"}
_VALID_DIALECTS = {
    "anthropic_adaptive",
    "anthropic_budget",
    "gemini_budget",
    "openai_effort",
    "zai_effort",
    "zai_thinking",
    "deepseek_effort",
    "kimi_effort",
    "kimi_thinking",
    "compat_effort",
    "compat_thinking_flag",
}


# --- provider registry -------------------------------------------------------


def test_provider_tag_matches_its_key():
    for tag, spec in PROVIDERS.items():
        assert spec.tag == tag


def test_every_model_provider_is_registered():
    for key, spec in MODELS.items():
        assert spec.provider in PROVIDERS, f"{key} -> unknown provider {spec.provider}"


def test_openai_compat_hosts_carry_a_base_url():
    for tag, spec in PROVIDERS.items():
        if spec.adapter_kind == "openai_compat":
            assert spec.base_url, f"{tag} is openai_compat but has no base_url"


def test_key_namespace_matches_its_tag():
    for tag, spec in PROVIDERS.items():
        if spec.key_namespace:
            assert spec.key_namespace == f"{tag}:"


def test_azure_is_not_listable():
    # Foundry has no live deployment-listing inference API.
    for tag in _AZURE:
        assert tag not in LISTABLE_PROVIDERS
        assert tag in DISCOVERABLE_PROVIDERS
        assert PROVIDERS[tag].registry_models


def test_every_provider_declares_a_key_env():
    for tag, spec in PROVIDERS.items():
        assert spec.api_key_env, f"{tag} declares no API key env var"
        for name in spec.api_key_env:
            assert name.isupper(), f"{tag}: {name} should be an ENV_VAR name"


# --- model registry ----------------------------------------------------------


def test_keys_are_lowercase():
    for key in MODELS:
        assert key == key.lower(), key


def test_wire_ids_are_non_empty():
    for key, spec in MODELS.items():
        assert spec.wire_id.strip(), key


def test_dev_suffix_only_on_azure_rows():
    """`-dev` is the Azure deployment marker. A public model ending in `-dev`
    would make the legacy WORK fallback and the guess ladder both lie."""
    for key, spec in MODELS.items():
        if key.endswith("-dev"):
            assert spec.provider in _AZURE, f"{key} ends -dev but is not Azure"
        if spec.provider in _AZURE:
            assert key.endswith("-dev"), f"{key} is Azure but does not end -dev"
            # Azure keys ARE the user-created deployment names, so key == wire.
            assert spec.wire_id == key


def test_namespaced_keys_strip_the_namespace_in_wire_id():
    for key, spec in MODELS.items():
        if ":" in key:
            ns, bare = key.split(":", 1)
            assert ns == spec.provider, f"{key} namespaced {ns} but routes to {spec.provider}"
            assert ":" not in spec.wire_id, f"{key} leaks its namespace onto the wire"
            # Case may differ (regolo serves 'Llama-3.3-70B-Instruct').
            assert spec.wire_id.lower() == bare, f"{key} -> {spec.wire_id}"


def test_host_providers_namespace_all_their_rows():
    for key, spec in MODELS.items():
        ns = PROVIDERS[spec.provider].key_namespace
        if ns:
            assert key.startswith(ns), f"{key} belongs to {spec.provider} but is not namespaced"


def test_alt_model_targets_resolve():
    for key, spec in MODELS.items():
        if spec.alt_model:
            assert spec.alt_model in MODELS, f"{key}.alt_model -> missing {spec.alt_model}"
            assert spec.alt_model != key, f"{key}.alt_model points at itself"


def test_azure_alias_targets_resolve_to_azure_rows():
    for key, spec in MODELS.items():
        if spec.azure_alias:
            target = MODELS.get(spec.azure_alias)
            assert target is not None, f"{key}.azure_alias -> missing {spec.azure_alias}"
            assert target.provider in _AZURE, f"{key}.azure_alias -> non-Azure row"
        # An Azure row cannot itself be redirected to Azure.
        if spec.provider in _AZURE:
            assert spec.azure_alias is None, key


def test_context_windows_are_plausible():
    for key, spec in MODELS.items():
        assert 1_000 <= spec.context_window <= 10_000_000, f"{key}: {spec.context_window}"


# --- capabilities ------------------------------------------------------------


def test_native_efforts_are_known_words_in_cheapest_first_order():
    """`native_efforts` is the PROVIDER's vocabulary, not gllm's ladder, so it
    may contain words gllm never exposes (`none`, `max`). Order is load-bearing:
    `resolve_effort` reads `native[-1]` as the top rung."""
    for key, spec in MODELS.items():
        efforts = spec.caps.native_efforts
        for word in efforts:
            assert word in _RANK, f"{key}: {word!r} is not a known effort word"
        ranked = sorted(efforts, key=_RANK.index)
        assert list(efforts) == ranked, f"{key}: not ordered cheapest-first"


def test_thinking_dialect_and_efforts_agree():
    """A dialect with no rungs is unreachable; rungs with no dialect are a
    translation gap. Either both or neither."""
    for key, spec in MODELS.items():
        has_dialect = spec.caps.thinking_dialect is not None
        has_efforts = bool(spec.caps.native_efforts)
        assert has_dialect == has_efforts, key
        if has_dialect:
            assert spec.caps.thinking_dialect in _VALID_DIALECTS, key


def test_dialect_matches_provider():
    prefix_by_provider = {
        "anthropic": "anthropic_",
        "azure_anthropic": "anthropic_",
        "gemini": "gemini_",
        "openai": "openai_",
        "azure_openai": "openai_",
        "grok": "openai_",
        "zai": "zai_",
        "deepseek": "deepseek_",
        "kimi": "kimi_",
        "groq": "compat_",
        "regolo": "compat_",
    }
    for key, spec in MODELS.items():
        dialect = spec.caps.thinking_dialect
        if dialect is None:
            continue
        assert dialect.startswith(prefix_by_provider[spec.provider]), f"{key}: {dialect}"


def test_api_surface_only_on_openai_family():
    for key, spec in MODELS.items():
        if spec.caps.api_surface is not None:
            assert spec.caps.api_surface in ("responses", "chat"), key
            assert spec.provider in _OPENAI_FAMILY, key


def test_chat_surface_models_have_no_pdf_or_reasoning():
    """PDF input is `input_file`, which exists only on the Responses API, and
    the classic chat line has no reasoning control."""
    for key, spec in MODELS.items():
        if spec.caps.api_surface == "chat":
            assert not spec.caps.supports_pdf, key
            assert not spec.caps.native_efforts, key


def test_pdf_capable_models_are_also_vision_capable():
    for key, spec in MODELS.items():
        if spec.caps.supports_pdf:
            assert spec.caps.supports_vision, key


# --- lookups and the legacy fallback ----------------------------------------


def test_spec_for_is_case_insensitive_and_trims():
    assert spec_for("  CLAUDE-Opus-5 ") is MODELS["claude-opus-5"]


def test_wire_id_strips_the_namespace():
    assert wire_id_for("groq:openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert wire_id_for("regolo:llama-3.3-70b-instruct") == "Llama-3.3-70B-Instruct"
    # Unregistered names pass through untouched.
    assert wire_id_for("some-model-we-never-heard-of") == "some-model-we-never-heard-of"


def test_caps_for_unknown_model_falls_back_not_raises():
    caps = caps_for("gpt-6-imaginary", "openai")
    assert isinstance(caps, ModelCaps)
    # Unknown OpenAI-family names default to Responses, the strict superset.
    assert caps.api_surface == "responses"
    assert caps.native_efforts  # guessed, not refused — the API is the one to 400


def test_bundled_price_overrides_name_real_models():
    """Every override key must be a registry key.

    Catches both directions of drift: a typo'd key (silently never matches, so
    the model looks unpriced) and a row left behind after a model is dropped.
    Pure — reads the bundled JSON, never the network feed.
    """
    import json

    from gllm.pricing import _bundled_overrides_path

    raw = json.loads(_bundled_overrides_path().read_text(encoding="utf-8"))
    for key in raw:
        if key.startswith("_"):
            continue
        assert key in MODELS, f"data/prices.json prices {key!r}, which is not a model"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-5", "anthropic_adaptive"),
        ("claude-fable-5", "anthropic_adaptive"),
        ("claude-opus-4-8", "anthropic_adaptive"),
        ("claude-sonnet-4-6", "anthropic_adaptive"),
        ("claude-opus-4-5", "anthropic_budget"),
        ("claude-haiku-4-5", "anthropic_budget"),
    ],
)
def test_claude_thinking_dialects(model, expected):
    assert MODELS[model].caps.thinking_dialect == expected


def test_bundled_price_overrides_do_not_shadow_the_book():
    """data/prices.json is the GAP layer: a row the tracker book also covers is
    a stale copy waiting to diverge. Mark deliberate disagreements with
    "override": true."""
    import gllm.pricing as pricing
    from llm_price_tracker.book import get_entry, load_book

    book = load_book()
    data = pricing._read_override_file(pricing._bundled_overrides_path())
    shadows = []
    for key, row in data.items():
        if key.startswith("_") or not isinstance(row, dict):
            continue
        if row.get("override") is True:
            continue
        entry = get_entry(key, book)
        if entry is not None and entry.standard is not None:
            shadows.append(key)
    assert shadows == [], f"book-covered rows in data/prices.json: {shadows}"
