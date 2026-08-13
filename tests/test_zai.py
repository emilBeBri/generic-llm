"""Z.AI / GLM adapter request shaping, without network calls.

GLM is standalone rather than an `OpenAIProvider` subclass — that one routes
`glm-*` to the Responses API, which Z.AI does not speak. These assert the wire
body directly through the `_http.post_json` seam.
"""

from __future__ import annotations

import pytest

from gllm.adapters import zai as za
from gllm.adapters.zai import ZaiProvider
from gllm.domain import Attachment, Request

PNG = Attachment(b"\x89PNG\r\n\x1a\n", "image/png", "shot.png")


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("GLLM_BASE_URL_ZAI", raising=False)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    return ZaiProvider(api_key="sk-test")


@pytest.fixture
def posted(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers, payload, **kw):
        calls.append((url, headers, payload))
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": "ok", "role": "assistant"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }

    monkeypatch.setattr(za, "post_json", fake_post)
    return calls


def test_the_trailing_slash_in_the_default_base_url_is_not_doubled(provider, posted):
    """ZAI_DEFAULT_BASE_URL ends in `/`; a naive join gives `//chat/completions`."""
    provider.generate(Request(prompt="hej", model="glm-4.7"))
    assert posted[0][0] == "https://api.z.ai/api/paas/v4/chat/completions"


def test_headers_carry_the_bearer_token(provider, posted):
    provider.generate(Request(prompt="hej", model="glm-4.7"))
    assert posted[0][1] == {"Authorization": "Bearer sk-test"}


def test_thinking_goes_top_level_not_in_extra_body(provider, posted):
    provider.generate(
        Request(prompt="hej", model="glm-4.7", reasoning="high", wire_effort="high")
    )
    body = posted[0][2]
    assert body["thinking"] == {"type": "enabled"}
    assert "extra_body" not in body, "extra_body was an SDK concept, never a wire one"


def test_reasoning_effort_is_gated_to_models_that_honour_it(provider, posted):
    """Only glm-5.2+ takes reasoning_effort; 4.7 gets the thinking block alone."""
    provider.generate(
        Request(prompt="hej", model="glm-4.7", reasoning="high", wire_effort="high")
    )
    assert "reasoning_effort" not in posted[0][2]

    provider.generate(
        Request(prompt="hej", model="glm-5.2", reasoning="high", wire_effort="high")
    )
    assert posted[1][2]["reasoning_effort"] == "high"


def test_json_mode_sets_response_format(provider, posted):
    provider.generate(Request(prompt="hej", model="glm-4.7", json_mode=True))
    assert posted[0][2]["response_format"] == {"type": "json_object"}


def test_schema_is_refused(provider, posted):
    with pytest.raises(RuntimeError, match="no native JSON-schema enforcement"):
        provider.generate(
            Request(prompt="hej", model="glm-4.7", schema={"type": "object"})
        )


def test_images_are_refused_on_a_text_model(provider, posted):
    with pytest.raises(RuntimeError, match="is not a vision model"):
        provider.generate(
            Request(prompt="describe", model="glm-4.7", attachments=(PNG,))
        )


def test_images_are_accepted_on_a_vision_model(provider, posted):
    provider.generate(
        Request(prompt="describe", model="glm-4.6v", attachments=(PNG,))
    )
    content = posted[0][2]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_pdfs_are_refused(provider, posted):
    with pytest.raises(RuntimeError, match="does not accept PDF"):
        provider.generate(
            Request(
                prompt="read",
                model="glm-4.6v",
                attachments=(Attachment(b"%PDF-", "application/pdf", "a.pdf"),),
            )
        )


def test_response_and_usage_are_mapped(provider, posted):
    response = provider.generate(Request(prompt="hej", model="glm-4.7"))
    assert response.text == "ok"
    assert response.provider == "zai"
    assert response.input_tokens == 12
    assert response.output_tokens == 3


def test_base_url_override_beats_the_default(monkeypatch, posted):
    monkeypatch.setenv("GLLM_BASE_URL_ZAI", "http://127.0.0.1:9/zai/")
    provider = ZaiProvider(api_key="sk-test")
    provider.generate(Request(prompt="hej", model="glm-4.7"))
    assert posted[0][0] == "http://127.0.0.1:9/zai/chat/completions"


def test_the_legacy_zai_base_url_still_works(monkeypatch, posted):
    """Predates the generic name; kept because a coding-plan key needs it."""
    monkeypatch.delenv("GLLM_BASE_URL_ZAI", raising=False)
    monkeypatch.setenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/")
    provider = ZaiProvider(api_key="sk-test")
    provider.generate(Request(prompt="hej", model="glm-4.7"))
    assert posted[0][0] == "https://api.z.ai/api/coding/paas/v4/chat/completions"


def test_list_models_filters_to_text_generation(provider, monkeypatch):
    monkeypatch.setattr(
        za,
        "get_json",
        lambda url, headers, **kw: {
            "data": [{"id": "glm-4.7"}, {"id": "glm-5.2"}, {"id": "embedding-3"}],
        },
    )
    assert provider.list_models() == ["glm-4.7", "glm-5.2"]
