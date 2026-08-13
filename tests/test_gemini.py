"""Gemini adapter, without network calls.

The only SDK removal that was a rewrite rather than a call-site swap, because
`google-genai` renamed things. Two distinct hazards, and they need different
defences:

1. **Case:** the wire is camelCase (`usageMetadata`, `promptTokenCount`), the SDK
   was snake_case, and `gllm.usage.from_gemini` reads snake_case. `_snake_keys`
   bridges that mechanically.
2. **Different names:** `supported_actions` (SDK) vs
   `supportedGenerationMethods` (wire) are not case variants of each other, so no
   mechanical transform can bridge them. That one was caught live — reading the
   SDK's name returned an *empty catalog* instead of erroring.
"""

from __future__ import annotations

import pytest

from gllm._http import wrap
from gllm.adapters import gemini as gm
from gllm.adapters.gemini import (
    GEMINI_API_VERSION,
    GEMINI_BASE_URL,
    GeminiProvider,
    _candidate_text,
    _snake_keys,
)
from gllm.domain import Attachment, Request

PNG = Attachment(b"\x89PNG\r\n\x1a\n", "image/png", "shot.png")


@pytest.fixture
def provider(monkeypatch):
    for name in ("GLLM_BASE_URL_GEMINI", "GOOGLE_GEMINI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return GeminiProvider(api_key="k-test")


@pytest.fixture
def posted(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers, payload, **kw):
        calls.append((url, headers, payload))
        return {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 82,
                "cachedContentTokenCount": 0,
            },
        }

    monkeypatch.setattr(gm, "post_json", fake_post)
    return calls


# --- _snake_keys -----------------------------------------------------------

def test_snake_keys_rewrites_nested_camel_case():
    out = _snake_keys({"usageMetadata": {"promptTokenCount": 8}})
    assert out == {"usage_metadata": {"prompt_token_count": 8}}


def test_snake_keys_descends_into_lists():
    out = _snake_keys({"candidates": [{"finishReason": "STOP"}]})
    assert out == {"candidates": [{"finish_reason": "STOP"}]}


def test_snake_keys_leaves_values_alone():
    """Only keys are rewritten — a camelCase VALUE must survive verbatim."""
    assert _snake_keys({"finishReason": "MAX_TOKENS"}) == {"finish_reason": "MAX_TOKENS"}
    assert _snake_keys({"name": "models/gemini-3.5-flash"})["name"] == (
        "models/gemini-3.5-flash"
    )


# --- text extraction -------------------------------------------------------

def test_candidate_text_joins_the_text_parts():
    resp = wrap({"candidates": [{"content": {"parts": [
        {"text": "a"}, {"text": "b"},
    ]}}]})
    assert _candidate_text(resp) == "ab"


def test_candidate_text_skips_thought_parts():
    resp = wrap({"candidates": [{"content": {"parts": [
        {"text": "reasoning", "thought": True},
        {"text": "answer"},
    ]}}]})
    assert _candidate_text(resp) == "answer"


def test_candidate_text_survives_a_blocked_response_with_no_candidates():
    assert _candidate_text(wrap({"usage_metadata": {}})) == ""
    assert _candidate_text(wrap({"candidates": []})) == ""


# --- request shaping -------------------------------------------------------

def test_the_version_segment_is_not_part_of_the_base_url(provider, posted):
    """Folding /v1beta into the base breaks every override in the wild — the jail
    broker's GOOGLE_GEMINI_BASE_URL points at the host and 404s otherwise."""
    provider.generate(Request(prompt="hej", model="gemini-3.5-flash"))
    assert posted[0][0] == (
        f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}"
        f"/models/gemini-3.5-flash:generateContent"
    )


def test_a_base_url_override_keeps_the_version_segment(monkeypatch, posted):
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://127.0.0.1:9/gemini/")
    provider = GeminiProvider(api_key="k-test")
    provider.generate(Request(prompt="hej", model="gemini-3.5-flash"))
    assert posted[0][0] == (
        "http://127.0.0.1:9/gemini/v1beta/models/gemini-3.5-flash:generateContent"
    )


def test_the_key_travels_as_a_header_not_a_query_parameter(provider, posted):
    provider.generate(Request(prompt="hej", model="gemini-3.5-flash"))
    assert posted[0][1] == {"x-goog-api-key": "k-test"}
    assert "key=" not in posted[0][0]


def test_system_instruction_is_top_level_not_inside_generation_config(provider, posted):
    """Nesting it in generationConfig is a 400."""
    provider.generate(
        Request(prompt="hej", model="gemini-3.5-flash", system="be terse")
    )
    body = posted[0][2]
    assert body["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert "systemInstruction" not in body["generationConfig"]


def test_budget_and_temperature_live_in_generation_config(provider, posted):
    provider.generate(
        Request(prompt="hej", model="gemini-3.5-flash", temperature=0.3, max_tokens=999)
    )
    cfg = posted[0][2]["generationConfig"]
    assert cfg["maxOutputTokens"] == 999
    assert cfg["temperature"] == 0.3


def test_reasoning_sets_a_thinking_budget(provider, posted):
    provider.generate(
        Request(prompt="hej", model="gemini-3.5-flash", reasoning="high", wire_effort="high")
    )
    assert posted[0][2]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 16384}


def test_xhigh_resolves_to_dynamic_thinking(provider, posted):
    provider.generate(
        Request(prompt="hej", model="gemini-3.5-flash", reasoning="xhigh", wire_effort="xhigh")
    )
    assert posted[0][2]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}


def test_schema_uses_response_json_schema(provider, posted):
    provider.generate(
        Request(prompt="hej", model="gemini-3.5-flash", schema={"type": "object"})
    )
    cfg = posted[0][2]["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert cfg["responseJsonSchema"] == {"type": "object"}


def test_attachments_precede_the_prompt_as_inline_data(provider, posted):
    provider.generate(
        Request(prompt="describe", model="gemini-3.5-flash", attachments=(PNG,))
    )
    parts = posted[0][2]["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    assert parts[0]["inline_data"]["data"], "bytes must be base64, not raw"
    assert parts[-1] == {"text": "describe"}


# --- response mapping ------------------------------------------------------

def test_usage_is_mapped_through_the_snake_case_rewrite(provider, posted):
    response = provider.generate(Request(prompt="hej", model="gemini-3.5-flash"))
    assert response.text == "ok"
    assert response.input_tokens == 8
    assert response.output_tokens == 3
    # Thinking is billed on TOP of candidates here, so it is reported separately.
    assert response.reasoning_tokens == 82
    assert response.stop_reason == "STOP"
    assert not response.truncated
    assert response.usage_raw["prompt_token_count"] == 8


def test_max_tokens_finish_reason_is_detected_as_truncation(provider, monkeypatch):
    monkeypatch.setattr(
        gm, "post_json",
        lambda *a, **kw: {
            "candidates": [{
                "content": {"parts": [{"text": "1"}]},
                "finishReason": "MAX_TOKENS",
            }],
        },
    )
    response = provider.generate(Request(prompt="hej", model="gemini-3.5-flash"))
    assert response.stop_reason == "MAX_TOKENS"
    assert response.truncated


# --- catalog ---------------------------------------------------------------

def test_list_models_reads_supported_generation_methods(provider, monkeypatch):
    """The wire name, NOT the SDK's `supported_actions`. Getting this wrong
    returned an empty catalog rather than raising — caught only live."""
    monkeypatch.setattr(
        gm, "get_json",
        lambda url, headers, **kw: {
            "models": [
                {"name": "models/gemini-3.5-flash",
                 "supportedGenerationMethods": ["generateContent", "countTokens"]},
                {"name": "models/text-embedding-004",
                 "supportedGenerationMethods": ["embedContent"]},
                {"name": "models/imagen-4",
                 "supportedGenerationMethods": ["generateContent"]},
            ]
        },
    )
    # imagen advertises generateContent, so the name filter is what drops it.
    assert provider.list_models() == ["gemini-3.5-flash"]


def test_a_missing_key_names_both_accepted_env_vars(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY .or GOOGLE_API_KEY."):
        GeminiProvider()
