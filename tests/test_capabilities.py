from gllm.adapters._capabilities import (
    glm_supports_reasoning_effort,
    is_glm_vision_model,
    is_text_generation_model,
    supports_attachment,
    supports_image,
    supports_pdf,
    use_responses_api,
)
from gllm.domain import Attachment


def test_responses_api_models():
    for m in ["gpt-5", "gpt-5.5", "o1", "o3", "o4-mini", "codex-mini-latest", "grok-4.3"]:
        assert use_responses_api(m) is True


def test_chat_completions_models():
    for m in ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
        assert use_responses_api(m) is False


def test_image_capability_matrix():
    assert supports_image("anthropic", "claude-opus-5")
    assert supports_image("azure_anthropic", "claude-opus-4-8-dev")
    assert supports_image("openai", "gpt-5.6")
    assert supports_image("azure_openai", "gpt-5.1-dev")
    assert supports_image("gemini", "gemini-3.6-flash")
    assert supports_image("grok", "grok-4.3")
    assert supports_image("kimi", "kimi-k3")
    assert supports_image("kimi", "kimi-k2.6")
    assert not supports_image("deepseek", "deepseek-v4-flash")


def test_image_capability_is_per_model_on_glm():
    # Vision is a separate GLM model LINE, so 'can zai take an image' has no
    # honest provider-level answer — it is a per-model fact, and the registry
    # is where it now lives (it used to be an adapter-internal raise).
    assert supports_image("zai", "glm-4.6v")
    assert supports_image("zai", "glm-5v-turbo")
    assert not supports_image("zai", "glm-5.2")
    assert not supports_image("zai", "glm-4.7")


def test_image_capability_on_compat_hosts():
    # Groq/Regolo rows we carry are text-only today.
    assert not supports_image("groq", "groq:openai/gpt-oss-120b")
    assert not supports_image("regolo", "regolo:qwen3.5-122b")


def test_glm_vision_split():
    # Vision models take images; text GLMs do not.
    for m in ["glm-5v-turbo", "glm-4.6v", "glm-4.6v-flash", "glm-4.5v", "glm-ocr"]:
        assert is_glm_vision_model(m), m
    for m in ["glm-5.2", "glm-4.6", "glm-4.7-flash", "glm-4-32b-0414-128k"]:
        assert not is_glm_vision_model(m), m


def test_glm_reasoning_effort_gated_to_5_2():
    assert glm_supports_reasoning_effort("glm-5.2")
    # Thinking on/off works on 4.5+, but reasoning_effort is 5.2-only.
    for m in ["glm-5.1", "glm-5", "glm-4.7", "glm-4.6", "glm-4.6v"]:
        assert not glm_supports_reasoning_effort(m), m


def test_pdf_capability_matrix():
    # Anthropic and Gemini support PDFs on every model.
    assert supports_pdf("anthropic", "claude-opus-4-8")
    assert supports_pdf("azure_anthropic", "claude-opus-4-8-dev")
    assert supports_pdf("gemini", "gemini-3.1-pro-preview")

    # OpenAI supports file content parts on both API surfaces.
    assert supports_pdf("openai", "gpt-5")
    assert supports_pdf("openai", "o3-mini")
    assert supports_pdf("openai", "gpt-4o")
    assert supports_pdf("openai", "gpt-4.1-mini")
    assert supports_pdf("azure_openai", "gpt-4.1-nano-dev")

    # Grok / DeepSeek / GLM: never (no native document input).
    assert not supports_pdf("grok", "grok-4.3")
    assert not supports_pdf("deepseek", "deepseek-v4-flash")
    assert not supports_pdf("zai", "glm-4.6v")
    assert not supports_pdf("kimi", "kimi-k3")


def test_non_pdf_documents_are_public_openai_only():
    docx = Attachment(
        data=b"office bytes",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        source_label="report.docx",
    )
    assert supports_attachment("openai", "gpt-5.6", docx)
    assert supports_attachment("openai", "gpt-4o", docx)

    for provider, model in [
        ("azure_openai", "gpt-5.1-dev"),
        ("anthropic", "claude-opus-5"),
        ("azure_anthropic", "claude-opus-4-8-dev"),
        ("gemini", "gemini-3.6-flash"),
        ("grok", "grok-4.3"),
        ("zai", "glm-5v-turbo"),
        ("kimi", "kimi-k3"),
        ("deepseek", "deepseek-v4-flash"),
    ]:
        assert not supports_attachment(provider, model, docx)


def test_openai_accepts_text_file_from_stdin_with_explicit_mime():
    text = Attachment(
        data=b"hello",
        mime_type="text/plain",
        source_label="<stdin>",
    )
    assert supports_attachment("openai", "gpt-4.1", text)
    assert not supports_attachment("azure_openai", "gpt-4.1-nano-dev", text)


def test_openai_mime_override_accepts_an_unlisted_suffix():
    text = Attachment(
        data=b"hello",
        mime_type="text/plain",
        source_label="notes.dat",
    )
    assert supports_attachment("openai", "gpt-4.1", text)


def test_openai_rejects_unsupported_file_suffix():
    archive = Attachment(
        data=b"zip",
        mime_type="application/zip",
        source_label="bundle.zip",
    )
    assert not supports_attachment("openai", "gpt-5.6", archive)


def test_text_generation_filter_keeps_chat_models():
    # The `--models` filter must keep real text-generation chat models across
    # every provider family.
    for m in [
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-3.5-flash",
        "gpt-5.5",
        "o3-mini",
        "claude-opus-4-8",
        "deepseek-v4-flash",
        "grok-4.3",
        "gemma-4-31b-it",
        "gpt-4o-search-preview",  # web-search text model, must NOT be hidden
    ]:
        assert is_text_generation_model(m), m


def test_text_generation_filter_drops_media_and_embeddings():
    # Embeddings, speech, image, video, music, robotics, computer-use — all
    # advertise generateContent or appear in the raw catalog but are not usable
    # text-generation models through gllm.
    for m in [
        "text-embedding-3-large",
        "gemini-embedding-001",
        "whisper-1",
        "tts-1-hd",
        "gemini-2.5-flash-preview-tts",
        "dall-e-3",
        "gemini-2.5-flash-image",
        "gpt-4o-audio-preview",
        "gpt-4o-realtime-preview",
        "omni-moderation-latest",
        "sora-2",
        "grok-imagine-video",
        "lyria-3-pro-preview",
        "nano-banana-pro-preview",
        "gemini-robotics-er-1.6-preview",
        "gemini-2.5-computer-use-preview-10-2025",
    ]:
        assert not is_text_generation_model(m), m


def test_strict_schema_matrix():
    from gllm.adapters._capabilities import supports_strict_schema

    # Native json_schema enforcement.
    assert supports_strict_schema("anthropic", "claude-opus-4-8")
    assert supports_strict_schema("openai", "gpt-5.1")
    assert supports_strict_schema("openai", "gpt-4o")  # response_format json_schema
    assert supports_strict_schema("azure_openai", "gpt-4o-dev")
    assert supports_strict_schema("gemini", "gemini-3.1-pro-preview")
    assert supports_strict_schema("grok", "grok-4")
    # Azure Foundry exposes output_config; --schema is attempted natively
    # (verification pending — see AZURE-FOUNDRY-SMOKE-TEST.md).
    assert supports_strict_schema("azure_anthropic", "claude-opus-4-8-dev")

    # DeepSeek and GLM: json_object only (no native json_schema) — stay refused.
    assert not supports_strict_schema("deepseek", "deepseek-v4-flash")
    assert not supports_strict_schema("zai", "glm-4.6")
    assert not supports_strict_schema("kimi", "kimi-k3")
