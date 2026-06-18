from gllm.routing import effective_model, provider_for


def test_opus_4_8_routes_to_anthropic():
    assert provider_for("claude-opus-4-8") == "anthropic"


def test_opus_4_8_dev_routes_to_azure_anthropic():
    assert provider_for("claude-opus-4-8-dev") == "azure_anthropic"


def test_dev_suffix_overrides_anthropic_check():
    # The -dev suffix is checked first; without it claude-opus-4-7 is direct.
    assert provider_for("claude-opus-4-7-dev") == "azure_anthropic"
    assert provider_for("claude-opus-4-7") == "anthropic"


def test_gpt_5_5_dev_routes_to_azure_openai():
    assert provider_for("gpt-5.5-dev") == "azure_openai"


def test_full_bebri_chat_model_set():
    cases = {
        "claude-opus-4-5": "anthropic",
        "claude-opus-4-6": "anthropic",
        "claude-opus-4-7": "anthropic",
        "claude-opus-4-8": "anthropic",
        "claude-sonnet-4-5": "anthropic",
        "claude-sonnet-4-6": "anthropic",
        "claude-haiku-4-5": "anthropic",
        "claude-haiku-4-6": "anthropic",
        "gpt-5": "openai",
        "gpt-5-mini": "openai",
        "gpt-5.2": "openai",
        "gpt-5.2-pro": "openai",
        "gpt-5.1-codex": "openai",
        "o1": "openai",
        "o3-mini": "openai",
        "o4-mini": "openai",
        "codex-mini-latest": "openai",
        "gemini-3.1-pro-preview": "gemini",
        "gemini-3.5-flash": "gemini",
        "deepseek-v4-pro": "deepseek",
        "deepseek-v4-flash": "deepseek",
        "grok-4.3": "grok",
        "grok-4.20-multi-agent-0309": "grok",
        "grok-build-0.1": "grok",
        "glm-5.2": "zai",
        "glm-4.6": "zai",
        "glm-4.6v": "zai",
        "glm-ocr": "zai",
        "gpt-5.4-pro-dev": "azure_openai",
        "claude-opus-4-8-dev": "azure_anthropic",
    }
    for model, expected in cases.items():
        assert provider_for(model) == expected, f"{model} -> {provider_for(model)}"


# --- WORK-mode Azure redirect (effective_model) ------------------------------


def test_work_off_is_passthrough():
    assert effective_model("claude-opus-4-8", False) == "claude-opus-4-8"
    assert effective_model("gpt-5.1", False) == "gpt-5.1"


def test_work_appends_dev_to_anthropic_and_openai():
    assert effective_model("claude-opus-4-8", True) == "claude-opus-4-8-dev"
    assert effective_model("gpt-5.1", True) == "gpt-5.1-dev"
    assert effective_model("o3-mini", True) == "o3-mini-dev"
    # ...and the redirected name then routes to the Azure adapter.
    assert provider_for(effective_model("claude-opus-4-8", True)) == "azure_anthropic"
    assert provider_for(effective_model("gpt-5.1", True)) == "azure_openai"


def test_work_does_not_double_suffix_or_touch_explicit_dev():
    assert effective_model("claude-opus-4-8-dev", True) == "claude-opus-4-8-dev"


def test_work_leaves_non_azure_providers_alone():
    for m in ["gemini-3.5-flash", "grok-4", "deepseek-v4-flash", "glm-5.2"]:
        assert effective_model(m, True) == m
