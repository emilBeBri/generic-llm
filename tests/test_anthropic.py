"""Anthropic adapters, without network calls.

**Unit-tested only, and deliberately so:** the jail withholds Anthropic
credentials from this agent, and no Azure keys exist on this machine, so neither
`anthropic` nor `azure_anthropic` could be exercised live. See
`AZURE-FOUNDRY-SMOKE-TEST.md` for the live verification a work-box run should do.

The weight here is on `final_message_from_events`, which replaces the SDK's
`stream.get_final_message()`. Each event contributes something no other event
repeats, so a reassembly bug shows up as *quietly missing data* rather than an
error — usage reading zero output tokens is the classic symptom.
"""

from __future__ import annotations

import pytest

from gllm.adapters import anthropic as an
from gllm.adapters.anthropic import (
    ANTHROPIC_BASE_URL,
    AnthropicProvider,
    final_message_from_events,
)
from gllm.adapters.azure_anthropic import AzureAnthropicProvider, _normalize_foundry_url
from gllm.domain import Request

# A complete exchange, shaped exactly as
# platform.claude.com/docs/en/build-with-claude/streaming documents it.
STREAM = [
    {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "content": [],
        "model": "claude-opus-5", "stop_reason": None,
        "usage": {"input_tokens": 25, "output_tokens": 1},
    }},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "text", "text": ""}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "Hello"}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "!"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
     "usage": {"output_tokens": 15}},
    {"type": "message_stop"},
]


# --- SSE reassembly --------------------------------------------------------

def test_text_deltas_are_concatenated_in_order():
    msg = final_message_from_events(STREAM)
    assert msg["content"] == [{"type": "text", "text": "Hello!"}]


def test_stop_reason_comes_only_from_message_delta():
    """message_start carries stop_reason: None; message_delta is the real source."""
    assert final_message_from_events(STREAM)["stop_reason"] == "end_turn"


def test_usage_merges_both_events():
    """input_tokens exist only in message_start, output_tokens only in
    message_delta. Reading one event gives a half-empty usage record — the
    failure this test exists to catch."""
    usage = final_message_from_events(STREAM)["usage"]
    assert usage["input_tokens"] == 25
    assert usage["output_tokens"] == 15


def test_the_model_survives_from_message_start():
    assert final_message_from_events(STREAM)["model"] == "claude-opus-5"


def test_ping_and_content_block_stop_are_harmless():
    assert final_message_from_events([{"type": "ping"}] * 3)["content"] == []


def test_blocks_are_ordered_by_index_not_arrival():
    events = [
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text", "text": "second"}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": "first "}},
    ]
    msg = final_message_from_events(events)
    assert [b["text"] for b in msg["content"]] == ["first ", "second"]


def test_thinking_deltas_are_discarded():
    """gllm is one-shot and prints the answer only."""
    events = [
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "text_delta", "text": "answer"}},
    ]
    msg = final_message_from_events(events)
    text = "".join(b.get("text", "") for b in msg["content"] if b["type"] == "text")
    assert text == "answer"


def test_a_mid_stream_error_event_raises():
    """A 200 status is not the end of error handling: overloaded_error can arrive
    inside the stream."""
    events = [
        STREAM[0],
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
    ]
    with pytest.raises(RuntimeError, match="overloaded_error"):
        final_message_from_events(events)


# --- the direct adapter ----------------------------------------------------

@pytest.fixture
def provider(monkeypatch):
    for name in ("GLLM_BASE_URL_ANTHROPIC", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return AnthropicProvider(api_key="sk-ant-test")


def test_plain_calls_post_once_without_streaming(provider, monkeypatch):
    seen: list[tuple[str, dict, dict]] = []
    monkeypatch.setattr(
        an, "post_json",
        lambda url, headers, payload, **kw: seen.append((url, headers, payload)) or {
            "model": "claude-opus-5", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hej"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    )
    response = provider.generate(Request(prompt="hej", model="claude-opus-5"))

    url, headers, body = seen[0]
    assert url == f"{ANTHROPIC_BASE_URL}/v1/messages"
    assert headers == {"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"}
    assert "stream" not in body, "no reason to stream without reasoning"
    assert response.text == "hej"
    assert response.stop_reason == "end_turn"


def test_reasoning_streams_and_sets_stream_true(provider, monkeypatch):
    seen: list[dict] = []

    def fake_sse(url, headers, payload, **kw):
        seen.append(payload)
        return iter(STREAM)

    monkeypatch.setattr(an, "post_sse", fake_sse)
    response = provider.generate(
        Request(prompt="hej", model="claude-opus-5", reasoning="high", wire_effort="high")
    )
    assert seen[0]["stream"] is True
    assert seen[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    # output_config is a plain top-level field; extra_body was an SDK constraint.
    assert seen[0]["output_config"] == {"effort": "high"}
    assert "extra_body" not in seen[0]
    assert response.text == "Hello!"
    assert response.input_tokens == 25
    assert response.output_tokens == 15


def test_the_legacy_anthropic_base_url_is_honoured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/anthropic/")
    provider = AnthropicProvider(api_key="sk-ant-test")
    assert provider.base_url == "http://127.0.0.1:9/anthropic"


def test_a_missing_key_is_refused(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicProvider()


def test_list_models_asks_for_one_big_page(provider, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        an, "get_json",
        lambda url, headers, **kw: seen.append(url) or {
            "data": [{"id": "claude-opus-5"}, {"id": "claude-haiku-4-5"}],
        },
    )
    assert provider.list_models() == ["claude-haiku-4-5", "claude-opus-5"]
    assert "limit=1000" in seen[0], "the catalog paginates; one page avoids after_id"


# --- Azure Foundry ---------------------------------------------------------

def test_foundry_agents_endpoints_are_rewritten_to_the_maas_host():
    assert _normalize_foundry_url("https://res.services.ai.azure.com") == (
        "https://res.openai.azure.com/anthropic"
    )
    assert _normalize_foundry_url("https://res.cognitiveservices.azure.com") == (
        "https://res.openai.azure.com/anthropic"
    )


def test_an_already_maas_endpoint_only_gains_the_anthropic_suffix():
    assert _normalize_foundry_url("https://res.openai.azure.com") == (
        "https://res.openai.azure.com/anthropic"
    )
    assert _normalize_foundry_url("https://res.openai.azure.com/anthropic") == (
        "https://res.openai.azure.com/anthropic"
    )


def test_azure_always_streams(monkeypatch):
    seen: list[tuple[str, dict, dict]] = []

    def fake_sse(url, headers, payload, **kw):
        seen.append((url, headers, payload))
        return iter(STREAM)

    monkeypatch.setattr("gllm.adapters.azure_anthropic.post_sse", fake_sse)
    provider = AzureAnthropicProvider(
        api_key="sk-az", endpoint="https://res.openai.azure.com"
    )
    response = provider.generate(Request(prompt="hej", model="claude-opus-4-8-dev"))

    url, headers, body = seen[0]
    assert url == "https://res.openai.azure.com/anthropic/v1/messages"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["x-api-key"] == "sk-az"
    assert body["stream"] is True
    assert response.text == "Hello!"
    assert response.provider == "azure_anthropic"


def test_azure_requires_both_key_and_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_FOUNDRY_ENDPOINT is not set"):
        AzureAnthropicProvider(api_key="sk-az")
    monkeypatch.delenv("AZURE_ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_ANTHROPIC_API_KEY is not set"):
        AzureAnthropicProvider()
