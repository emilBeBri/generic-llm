"""OpenAI adapter (both surfaces) plus its two subclasses, without network calls.

The Responses API is the only conversion in the tree that had to rebuild
something the SDK provided: `resp.output_text` is an SDK convenience, not a wire
field, so `_output_text` reassembles it from the raw typed `output` list. That
reconstruction is most of what these tests are for.

Azure is unit-tested only, deliberately: no Azure keys exist on this machine, so
its endpoint derivation is verified statically rather than live.
"""

from __future__ import annotations

import pytest

from gllm._http import wrap
from gllm.adapters import openai as oa
from gllm.adapters.azure_openai import AzureOpenAIProvider
from gllm.adapters.grok import GrokProvider
from gllm.adapters.openai import OPENAI_BASE_URL, OpenAIProvider, _output_text
from gllm.domain import Attachment, Request

RESPONSES_MODEL = "gpt-5.6"      # Responses surface
CHAT_MODEL = "gpt-4o"            # Chat Completions surface


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENAI_BASE_URL", "GLLM_BASE_URL_OPENAI", "GLLM_BASE_URL_GROK"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def posted(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers, payload, **kw):
        calls.append((url, headers, payload))
        if url.endswith("/responses"):
            return {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    },
                ],
                "usage": {"input_tokens": 9, "output_tokens": 2},
            }
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2},
        }

    monkeypatch.setattr(oa, "post_json", fake_post)
    return calls


# --- _output_text: what the SDK used to hand us for free -------------------

def test_output_text_skips_non_message_items():
    """`reasoning` and tool-call items carry no answer text."""
    resp = wrap({
        "output": [
            {"type": "reasoning", "summary": ["thinking..."]},
            {"type": "web_search_call", "status": "completed"},
            {"type": "message", "content": [{"type": "output_text", "text": "hej"}]},
        ]
    })
    assert _output_text(resp) == "hej"


def test_output_text_concatenates_every_text_part_in_order():
    resp = wrap({
        "output": [
            {"type": "message", "content": [
                {"type": "output_text", "text": "a"},
                {"type": "output_text", "text": "b"},
            ]},
            {"type": "message", "content": [{"type": "output_text", "text": "c"}]},
        ]
    })
    assert _output_text(resp) == "abc"


def test_output_text_is_empty_when_there_is_no_output():
    assert _output_text(wrap({"output": []})) == ""
    assert _output_text(wrap({"status": "completed"})) == ""


def test_a_refusal_with_no_text_raises_instead_of_printing_a_blank_line():
    """The SDK's output_text would have been "", so gllm would have exited 0
    having printed nothing — indistinguishable from an empty answer."""
    resp = wrap({
        "output": [{"type": "message", "content": [
            {"type": "refusal", "refusal": "I can't help with that."},
        ]}]
    })
    with pytest.raises(RuntimeError, match="refused this request"):
        _output_text(resp)


def test_a_refusal_alongside_text_does_not_raise():
    resp = wrap({
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "here is the safe part"},
            {"type": "refusal", "refusal": "but not that bit"},
        ]}]
    })
    assert _output_text(resp) == "here is the safe part"


# --- the two surfaces ------------------------------------------------------

def test_responses_surface_posts_to_responses(posted):
    provider = OpenAIProvider(api_key="sk-test")
    response = provider.generate(Request(prompt="hej", model=RESPONSES_MODEL))
    url, headers, body = posted[0]
    assert url == f"{OPENAI_BASE_URL}/responses"
    assert headers == {"Authorization": "Bearer sk-test"}
    assert body["input"] == "hej"
    assert body["store"] is False
    assert response.text == "ok"
    assert response.input_tokens == 9


def test_chat_surface_posts_to_chat_completions(posted):
    provider = OpenAIProvider(api_key="sk-test")
    response = provider.generate(Request(prompt="hej", model=CHAT_MODEL))
    url, _, body = posted[0]
    assert url == f"{OPENAI_BASE_URL}/chat/completions"
    assert body["messages"] == [{"role": "user", "content": "hej"}]
    assert response.text == "ok"
    assert response.stop_reason == "stop"


def test_reasoning_effort_goes_in_the_reasoning_object(posted):
    provider = OpenAIProvider(api_key="sk-test")
    provider.generate(
        Request(prompt="hej", model=RESPONSES_MODEL, reasoning="high", wire_effort="high")
    )
    assert posted[0][2]["reasoning"] == {"effort": "high"}
    assert "temperature" not in posted[0][2]


def test_schema_uses_the_responses_text_format_shape(posted):
    provider = OpenAIProvider(api_key="sk-test")
    provider.generate(
        Request(prompt="hej", model=RESPONSES_MODEL, schema={"type": "object"})
    )
    fmt = posted[0][2]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True


def test_truncation_is_read_from_incomplete_details(monkeypatch):
    monkeypatch.setattr(
        oa,
        "post_json",
        lambda *a, **kw: {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "cut off"},
            ]}],
        },
    )
    provider = OpenAIProvider(api_key="sk-test")
    response = provider.generate(Request(prompt="hej", model=RESPONSES_MODEL))
    assert response.stop_reason == "max_output_tokens"
    assert response.truncated


# --- base URL resolution ---------------------------------------------------

def test_the_legacy_openai_base_url_is_honoured(monkeypatch, posted):
    """The SDK used to read this itself; the jail's key broker sets it, so
    dropping the SDK without wiring it would have broken every jailed call."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/openai/")
    provider = OpenAIProvider(api_key="sk-test")
    provider.generate(Request(prompt="hej", model=CHAT_MODEL))
    assert posted[0][0] == "http://127.0.0.1:9/openai/chat/completions"


def test_the_generic_name_beats_the_legacy_one(monkeypatch, posted):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/legacy/")
    monkeypatch.setenv("GLLM_BASE_URL_OPENAI", "http://127.0.0.1:9/generic/")
    provider = OpenAIProvider(api_key="sk-test")
    provider.generate(Request(prompt="hej", model=CHAT_MODEL))
    assert posted[0][0] == "http://127.0.0.1:9/generic/chat/completions"


def test_a_subclass_base_url_beats_the_environment(monkeypatch, posted):
    monkeypatch.setenv("GLLM_BASE_URL_OPENAI", "http://127.0.0.1:9/generic/")
    provider = OpenAIProvider(api_key="sk-test", base_url="https://sub.example/v1")
    provider.generate(Request(prompt="hej", model=CHAT_MODEL))
    assert posted[0][0] == "https://sub.example/v1/chat/completions"


def test_a_missing_key_is_refused(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OpenAIProvider(api_key=None)


def test_list_models_filters_to_text_generation(monkeypatch):
    monkeypatch.setattr(
        oa,
        "get_json",
        lambda url, headers, **kw: {
            "data": [{"id": "gpt-5.6"}, {"id": "text-embedding-3-large"}],
        },
    )
    assert OpenAIProvider(api_key="sk-test").list_models() == ["gpt-5.6"]


# --- the subclasses --------------------------------------------------------

def test_grok_points_at_xai_and_renames_the_provider(posted):
    provider = GrokProvider(api_key="sk-xai")
    assert provider.name == "grok"
    provider.generate(Request(prompt="hej", model="grok-4.5"))
    assert posted[0][0] == "https://api.x.ai/v1/responses"
    assert posted[0][1] == {"Authorization": "Bearer sk-xai"}


def test_grok_refuses_pdfs_before_the_api_can(posted):
    provider = GrokProvider(api_key="sk-xai")
    with pytest.raises(RuntimeError, match="does not accept PDF"):
        provider.generate(
            Request(
                prompt="read",
                model="grok-4.5",
                attachments=(Attachment(b"%PDF-", "application/pdf", "a.pdf"),),
            )
        )


def test_grok_honours_its_own_base_url_override(monkeypatch, posted):
    monkeypatch.setenv("GLLM_BASE_URL_GROK", "http://127.0.0.1:9/grok/")
    provider = GrokProvider(api_key="sk-xai")
    provider.generate(Request(prompt="hej", model="grok-4.5"))
    assert posted[0][0] == "http://127.0.0.1:9/grok/responses"


def test_azure_appends_v1_to_a_bare_foundry_endpoint(posted):
    provider = AzureOpenAIProvider(
        api_key="sk-az", endpoint="https://res.openai.azure.com"
    )
    assert provider.name == "azure_openai"
    provider.generate(Request(prompt="hej", model="gpt-5.1-dev"))
    assert posted[0][0] == "https://res.openai.azure.com/v1/responses"


def test_azure_does_not_double_up_an_existing_v1(posted):
    provider = AzureOpenAIProvider(
        api_key="sk-az", endpoint="https://res.openai.azure.com/v1/"
    )
    provider.generate(Request(prompt="hej", model="gpt-5.1-dev"))
    assert posted[0][0] == "https://res.openai.azure.com/v1/responses"


def test_azure_requires_both_key_and_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_FOUNDRY_ENDPOINT is not set"):
        AzureOpenAIProvider(api_key="sk-az")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY is not set"):
        AzureOpenAIProvider()
