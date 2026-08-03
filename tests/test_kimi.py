"""Moonshot Kimi adapter request shaping, without network calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gllm.adapters.kimi import KimiProvider
from gllm.domain import Attachment, Request


class _Completions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=80,
        )
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=usage,
        )


def _provider() -> tuple[KimiProvider, _Completions]:
    completions = _Completions()
    provider = object.__new__(KimiProvider)
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return provider, completions


def test_k3_sends_effort_without_thinking_block():
    provider, completions = _provider()
    response = provider.generate(
        Request(
            prompt="hej",
            model="kimi-k3",
            reasoning="xhigh",
            wire_effort="max",
        )
    )

    assert response.text == "ok"
    assert completions.kwargs["reasoning_effort"] == "max"
    assert "extra_body" not in completions.kwargs
    assert completions.kwargs["max_completion_tokens"] == 16_000


def test_k26_sends_binary_thinking_without_effort():
    provider, completions = _provider()
    provider.generate(
        Request(
            prompt="hej",
            model="kimi-k2.6",
            reasoning="low",
            wire_effort="high",
        )
    )

    assert completions.kwargs["extra_body"] == {
        "thinking": {
            "type": "enabled",
        },
    }
    assert "reasoning_effort" not in completions.kwargs


def test_k27_rejects_explicit_reasoning():
    provider, _ = _provider()

    with pytest.raises(RuntimeError, match="exposes no reasoning control"):
        provider.generate(
            Request(
                prompt="hej",
                model="kimi-k2.7-code",
                reasoning="high",
                wire_effort="high",
            )
        )


def test_temperature_is_refused_not_ignored():
    provider, _ = _provider()

    with pytest.raises(RuntimeError, match="drop --temperature"):
        provider.generate(
            Request(prompt="hej", model="kimi-k3", temperature=0.2)
        )


def test_images_use_data_uri_content_blocks():
    provider, completions = _provider()
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

    content = completions.kwargs["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
