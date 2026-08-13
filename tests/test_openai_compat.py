"""The generic OpenAI-compatible host adapter, without network calls.

One adapter parameterised by a `ProviderSpec`, so the thing worth testing is
that host quirks stay *config* rather than branches — and, since the stdlib
transport landed, that `ProviderSpec.extra_body` reaches the request body at all.
Regolo's `disable_fallbacks` is the load-bearing case: without it the host may
silently answer with a different model than the one asked for, which makes model
identity, cost accounting and every capability gate a lie.
"""

from __future__ import annotations

import pytest

from gllm.adapters import openai_compat as oc
from gllm.adapters.openai_compat import OpenAICompatProvider
from gllm.domain import Attachment, Request
from gllm.providers import PROVIDERS

PNG = Attachment(b"\x89PNG\r\n\x1a\n", "image/png", "shot.png")


@pytest.fixture
def posted(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers, payload, **kw):
        calls.append((url, headers, payload))
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": "ok", "role": "assistant"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }

    monkeypatch.setattr(oc, "post_json", fake_post)
    return calls


def _provider(tag, monkeypatch):
    monkeypatch.delenv(f"GLLM_BASE_URL_{tag.upper()}", raising=False)
    return OpenAICompatProvider(PROVIDERS[tag], api_key="sk-test")


def test_groq_posts_to_its_own_endpoint(monkeypatch, posted):
    provider = _provider("groq", monkeypatch)
    provider.generate(Request(prompt="hej", model="groq:openai/gpt-oss-120b"))
    url, headers, body = posted[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers == {"Authorization": "Bearer sk-test"}
    # The host knows the bare id, not gllm's namespaced registry key.
    assert body["model"] == "groq:openai/gpt-oss-120b"


def test_the_wire_model_strips_the_host_namespace(monkeypatch, posted):
    provider = _provider("groq", monkeypatch)
    provider.generate(
        Request(
            prompt="hej",
            model="groq:openai/gpt-oss-120b",
            wire_model="openai/gpt-oss-120b",
        )
    )
    assert posted[0][2]["model"] == "openai/gpt-oss-120b"


def test_groq_sends_no_extra_body_keys(monkeypatch, posted):
    provider = _provider("groq", monkeypatch)
    provider.generate(Request(prompt="hej", model="groq:openai/gpt-oss-120b"))
    assert "disable_fallbacks" not in posted[0][2]
    assert "extra_body" not in posted[0][2]


def test_regolo_merges_disable_fallbacks_into_the_body(monkeypatch, posted):
    """The quirk that stops the host answering with a different model."""
    provider = _provider("regolo", monkeypatch)
    provider.generate(Request(prompt="hej", model="regolo:qwen3.5-122b"))
    body = posted[0][2]
    assert body["disable_fallbacks"] is True
    assert "extra_body" not in body, "extra_body was an SDK concept, never a wire one"


def test_compat_effort_sends_a_bare_reasoning_effort(monkeypatch, posted):
    provider = _provider("groq", monkeypatch)
    provider.generate(
        Request(
            prompt="hej",
            model="groq:openai/gpt-oss-120b",
            reasoning="high",
            wire_effort="high",
        )
    )
    body = posted[0][2]
    assert body["reasoning_effort"] == "high"
    assert "thinking" not in body


def test_compat_thinking_flag_pairs_effort_with_a_top_level_flag(monkeypatch, posted):
    provider = _provider("regolo", monkeypatch)
    provider.generate(
        Request(
            prompt="hej",
            model="regolo:gpt-oss-120b",
            reasoning="high",
            wire_effort="high",
        )
    )
    body = posted[0][2]
    assert body["reasoning_effort"] == "high"
    assert body["thinking"] is True
    assert body["disable_fallbacks"] is True


def test_regolo_adds_its_format_field_inside_image_url(monkeypatch):
    """Exercised at the unit, not through `generate`: NO registered regolo row
    currently has `supports_vision`, so `_user_content` would refuse every one of
    them before this code could run. The spec flag is live config for a model
    row that does not exist yet."""
    provider = _provider("regolo", monkeypatch)
    part = provider._image_part(PNG)
    assert part["image_url"]["format"] == "image/png"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_groq_omits_the_format_field(monkeypatch):
    provider = _provider("groq", monkeypatch)
    assert "format" not in provider._image_part(PNG)["image_url"]


def test_schema_is_refused(monkeypatch, posted):
    provider = _provider("groq", monkeypatch)
    with pytest.raises(RuntimeError, match="no native JSON-schema enforcement"):
        provider.generate(
            Request(
                prompt="hej",
                model="groq:openai/gpt-oss-120b",
                schema={"type": "object"},
            )
        )


def test_pdfs_are_refused(monkeypatch, posted):
    provider = _provider("groq", monkeypatch)
    with pytest.raises(RuntimeError, match="does not accept PDF"):
        provider.generate(
            Request(
                prompt="read",
                model="groq:openai/gpt-oss-120b",
                attachments=(Attachment(b"%PDF-", "application/pdf", "a.pdf"),),
            )
        )


def test_base_url_override_is_honoured(monkeypatch, posted):
    monkeypatch.setenv("GLLM_BASE_URL_GROQ", "http://127.0.0.1:9/groq/")
    provider = OpenAICompatProvider(PROVIDERS["groq"], api_key="sk-test")
    provider.generate(Request(prompt="hej", model="groq:openai/gpt-oss-120b"))
    assert posted[0][0] == "http://127.0.0.1:9/groq/chat/completions"


def test_a_missing_key_names_every_accepted_env_var(monkeypatch):
    for name in PROVIDERS["groq"].api_key_env:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="is not set"):
        OpenAICompatProvider(PROVIDERS["groq"])


def test_list_models_filters_to_text_generation(monkeypatch):
    provider = _provider("groq", monkeypatch)
    monkeypatch.setattr(
        oc,
        "get_json",
        lambda url, headers, **kw: {
            "data": [{"id": "qwen/qwen3-32b"}, {"id": "whisper-large-v3"}],
        },
    )
    assert provider.list_models() == ["qwen/qwen3-32b"]
