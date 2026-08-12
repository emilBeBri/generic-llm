"""DeepSeek adapter request shaping and response parsing, without network calls.

The seam is `_http.post_json` — the adapter's only outbound call — so these
assert the exact JSON body that goes on the wire. That is a stricter check than
the old SDK-client stubs allowed: `extra_body` used to hide the real request
shape behind the client's kwarg handling.
"""

from __future__ import annotations

import pytest

from gllm.adapters import deepseek as ds
from gllm.adapters.deepseek import DeepSeekProvider
from gllm.domain import Attachment, Request


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("GLLM_BASE_URL_DEEPSEEK", raising=False)
    return DeepSeekProvider(api_key="sk-test")


@pytest.fixture
def posted(monkeypatch):
    """Capture (url, headers, body) and reply with a minimal valid completion."""
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers, payload, **kw):
        calls.append((url, headers, payload))
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": "hej", "role": "assistant"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 80,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        }

    monkeypatch.setattr(ds, "post_json", fake_post)
    return calls


def test_posts_to_the_chat_completions_endpoint(provider, posted):
    provider.generate(Request(prompt="hej", model="deepseek-v4-pro"))
    url, headers, body = posted[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert headers == {"Authorization": "Bearer sk-test"}
    assert body["messages"] == [{"role": "user", "content": "hej"}]
    assert "thinking" not in body
    assert "reasoning_effort" not in body


def test_base_url_override_is_honoured(monkeypatch, posted):
    monkeypatch.setenv("GLLM_BASE_URL_DEEPSEEK", "http://127.0.0.1:8899/proxy/")
    provider = DeepSeekProvider(api_key="sk-test")
    provider.generate(Request(prompt="hej", model="deepseek-v4-pro"))
    # Trailing slash stripped so the join does not produce a double slash.
    assert posted[0][0] == "http://127.0.0.1:8899/proxy/chat/completions"


def test_thinking_and_effort_go_top_level_not_in_extra_body(provider, posted):
    provider.generate(
        Request(
            prompt="hej",
            model="deepseek-v4-pro",
            reasoning="xhigh",
            wire_effort="max",
        )
    )
    body = posted[0][2]
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "max"
    assert "extra_body" not in body, "extra_body was an SDK concept, never a wire one"
    # Thinking tokens come out of the output budget; the floor protects the answer.
    assert body["max_tokens"] == 16_000


def test_temperature_is_dropped_when_thinking_is_on(provider, posted):
    provider.generate(
        Request(
            prompt="hej",
            model="deepseek-v4-pro",
            temperature=0.2,
            reasoning="high",
            wire_effort="high",
        )
    )
    assert "temperature" not in posted[0][2]


def test_temperature_is_sent_when_thinking_is_off(provider, posted):
    provider.generate(
        Request(prompt="hej", model="deepseek-v4-pro", temperature=0.2)
    )
    assert posted[0][2]["temperature"] == 0.2


def test_json_mode_sets_response_format(provider, posted):
    provider.generate(
        Request(prompt="hej", model="deepseek-v4-pro", json_mode=True)
    )
    assert posted[0][2]["response_format"] == {"type": "json_object"}


def test_response_text_and_usage_are_mapped(provider, posted):
    response = provider.generate(Request(prompt="hej", model="deepseek-v4-pro"))

    assert response.text == "hej"
    assert response.model == "deepseek-v4-pro"
    assert response.provider == "deepseek"
    assert response.input_tokens == 100
    assert response.output_tokens == 20
    assert response.cache_read_tokens == 80
    assert response.reasoning_tokens == 7
    # usage_raw is the provider's own object, verbatim — including the nested
    # detail dict that the old attribute-scraping fallback would have dropped.
    assert response.usage_raw["prompt_cache_hit_tokens"] == 80
    assert response.usage_raw["completion_tokens_details"] == {"reasoning_tokens": 7}


def test_null_content_becomes_empty_string(provider, monkeypatch):
    monkeypatch.setattr(
        ds,
        "post_json",
        lambda *a, **kw: {
            "model": "deepseek-v4-pro",
            "choices": [{"message": {"role": "assistant", "reasoning_content": "..."}}],
        },
    )
    assert provider.generate(Request(prompt="hej", model="deepseek-v4-pro")).text == ""


def test_list_models_filters_to_text_generation(provider, monkeypatch):
    monkeypatch.setattr(
        ds,
        "get_json",
        lambda url, headers, **kw: {
            "data": [
                {"id": "deepseek-v4-pro"},
                {"id": "deepseek-v4-flash"},
                {"id": "text-embedding-3-large"},
            ]
        },
    )
    assert provider.list_models() == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_attachments_are_refused(provider, posted):
    with pytest.raises(RuntimeError, match="does not accept file attachments"):
        provider.generate(
            Request(
                prompt="hej",
                model="deepseek-v4-pro",
                attachments=(Attachment(b"x", "image/png", "a.png"),),
            )
        )


def test_schema_is_refused(provider, posted):
    with pytest.raises(RuntimeError, match="no native JSON-schema enforcement"):
        provider.generate(
            Request(prompt="hej", model="deepseek-v4-pro", schema={"type": "object"})
        )
