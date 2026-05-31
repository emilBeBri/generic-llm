from gllm.adapters._capabilities import (
    supports_image,
    supports_pdf,
    use_responses_api,
)


def test_responses_api_models():
    for m in ["gpt-5", "gpt-5.5", "o1", "o3", "o4-mini", "codex-mini-latest", "grok-4.3"]:
        assert use_responses_api(m) is True


def test_chat_completions_models():
    for m in ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
        assert use_responses_api(m) is False


def test_image_capability_matrix():
    assert supports_image("anthropic")
    assert supports_image("azure_anthropic")
    assert supports_image("openai")
    assert supports_image("azure_openai")
    assert supports_image("gemini")
    assert supports_image("grok")
    assert not supports_image("deepseek")


def test_pdf_capability_matrix():
    # Anthropic and Gemini support PDFs on every model.
    assert supports_pdf("anthropic", "claude-opus-4-8")
    assert supports_pdf("azure_anthropic", "claude-opus-4-8-dev")
    assert supports_pdf("gemini", "gemini-3-pro-preview")

    # OpenAI: PDFs only on the Responses API path.
    assert supports_pdf("openai", "gpt-5")
    assert supports_pdf("openai", "o3-mini")
    assert not supports_pdf("openai", "gpt-4o")
    assert not supports_pdf("openai", "gpt-4.1-mini")

    # Grok / DeepSeek: never.
    assert not supports_pdf("grok", "grok-4.3")
    assert not supports_pdf("deepseek", "deepseek-v4-flash")
