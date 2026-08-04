from __future__ import annotations

from gllm import cli
from gllm.providers import PROVIDERS


def _clear_provider_env(monkeypatch) -> None:
    for spec in PROVIDERS.values():
        for name in (*spec.api_key_env, *spec.required_env):
            monkeypatch.delenv(name, raising=False)


def test_default_listing_uses_only_configured_providers(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    assert cli._configured_model_providers(False) == ("openai", "gemini")


def test_work_listing_replaces_direct_hosts_with_configured_azure(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-test")
    monkeypatch.setenv("OPENAI_API_KEY", "direct-test")
    monkeypatch.setenv("AZURE_ANTHROPIC_API_KEY", "azure-test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test")
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://example.invalid")

    assert cli._configured_model_providers(True) == (
        "azure_openai",
        "azure_anthropic",
    )


def test_azure_requires_endpoint_for_default_discovery(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")

    assert cli._configured_model_providers(True) == ()


def test_work_listing_prints_registered_azure_deployments(monkeypatch, capsys):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://example.invalid")

    assert cli._run_models("*", work=True) == 0
    output = capsys.readouterr()
    assert "azure_openai\tgpt-5.6-sol-dev\n" in output.out
    assert output.err == ""


def test_explicit_direct_provider_follows_work_redirect(monkeypatch, capsys):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AZURE_ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://example.invalid")

    assert cli._run_models("anthropic", work=True) == 0
    output = capsys.readouterr()
    assert "azure_anthropic\tclaude-opus-5-dev\n" in output.out
    assert output.err == ""
