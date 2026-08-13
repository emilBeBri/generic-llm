"""Moonshot Kimi adapter request shaping, without network calls.

The seam is `_http.post_json`, so these assert the exact JSON body that goes on
the wire. That is stricter than the old SDK-client stub allowed: `extra_body`
used to hide the real request shape behind the client's kwarg handling, and
k2.6's thinking block was asserted in a place the wire never sees.
"""

from __future__ import annotations

import pytest

from gllm.adapters import kimi as km
from gllm.adapters.kimi import KimiProvider
from gllm.domain import Attachment, Request


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("GLLM_BASE_URL_KIMI", raising=False)
    return KimiProvider(api_key="sk-test")


@pytest.fixture
def posted(monkeypatch):
    """Capture (url, headers, body); reply with a minimal valid completion."""
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers, payload, **kw):
        calls.append((url, headers, payload))
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": "ok", "role": "assistant"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 80,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        }

    monkeypatch.setattr(km, "post_json", fake_post)
    return calls


def test_posts_to_the_chat_completions_endpoint(provider, posted):
    provider.generate(Request(prompt="hej", model="kimi-k3"))
    url, headers, body = posted[0]
    assert url == "https://api.moonshot.ai/v1/chat/completions"
    assert headers == {"Authorization": "Bearer sk-test"}
    assert body["messages"] == [{"role": "user", "content": "hej"}]


def test_k3_sends_effort_without_a_thinking_block(provider, posted):
    provider.generate(
        Request(prompt="hej", model="kimi-k3", reasoning="xhigh", wire_effort="max")
    )
    body = posted[0][2]
    assert body["reasoning_effort"] == "max"
    assert "thinking" not in body
    assert "extra_body" not in body
    # The adapter forwards the CLI-resolved budget; it applies no floor itself.
    assert body["max_completion_tokens"] == 4096


def test_k26_sends_binary_thinking_top_level_without_effort(provider, posted):
    provider.generate(
        Request(prompt="hej", model="kimi-k2.6", reasoning="low", wire_effort="high")
    )
    body = posted[0][2]
    assert body["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in body
    assert "extra_body" not in body, "extra_body was an SDK concept, never a wire one"


def test_k27_rejects_explicit_reasoning(provider, posted):
    with pytest.raises(RuntimeError, match="exposes no reasoning control"):
        provider.generate(
            Request(
                prompt="hej", model="kimi-k2.7-code", reasoning="high", wire_effort="high"
            )
        )


def test_temperature_is_refused_not_ignored(provider, posted):
    with pytest.raises(RuntimeError, match="drop --temperature"):
        provider.generate(Request(prompt="hej", model="kimi-k3", temperature=0.2))


def test_images_use_data_uri_content_blocks(provider, posted):
    provider.generate(
        Request(
            prompt="describe",
            model="kimi-k3",
            attachments=(
                Attachment(
                    data=b"\x89PNG\r\n\x1a\n",
                    mime_type="image/png",
                    source_label="shot.png",
                ),
            ),
        )
    )
    content = posted[0][2]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_pdfs_are_refused(provider, posted):
    with pytest.raises(RuntimeError, match="does not accept inline PDF"):
        provider.generate(
            Request(
                prompt="read",
                model="kimi-k3",
                attachments=(Attachment(b"%PDF-", "application/pdf", "a.pdf"),),
            )
        )


def test_response_and_usage_are_mapped(provider, posted):
    response = provider.generate(Request(prompt="hej", model="kimi-k3"))
    assert response.text == "ok"
    assert response.provider == "kimi"
    assert response.input_tokens == 100
    assert response.output_tokens == 20
    # Kimi reports cached prefix tokens top-level, unlike OpenAI's nested shape.
    assert response.cache_read_tokens == 80
    assert response.reasoning_tokens == 5
    assert response.usage_raw["cached_tokens"] == 80


def test_base_url_override_is_honoured(monkeypatch, posted):
    monkeypatch.setenv("GLLM_BASE_URL_KIMI", "http://127.0.0.1:9/proxy/")
    provider = KimiProvider(api_key="sk-test")
    provider.generate(Request(prompt="hej", model="kimi-k3"))
    assert posted[0][0] == "http://127.0.0.1:9/proxy/chat/completions"


def test_list_models_filters_to_text_generation(provider, monkeypatch):
    monkeypatch.setattr(
        km,
        "get_json",
        lambda url, headers, **kw: {
            # `moonshot-embed` alone would NOT be filtered — the markers are
            # "embedding" and "embed-", so a bare "embed" suffix slips through.
            "data": [
                {"id": "kimi-k3"},
                {"id": "kimi-k2.6"},
                {"id": "moonshot-embedding-v1"},
            ],
        },
    )
    assert provider.list_models() == ["kimi-k2.6", "kimi-k3"]
