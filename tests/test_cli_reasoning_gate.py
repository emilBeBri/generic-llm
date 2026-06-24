"""main()-level tests for the reasoning capability gate.

The contract: an *explicit* `--reasoning` a model can't honour is a hard error
(exit 2), but a reasoning level inherited from the ambient `$DEFAULT_EFFORT`
default is silently dropped on non-reasoning models (exit 0, no reasoning sent).
We mock the provider boundary so no network call happens; a captured Request
lets us assert what reasoning (if any) was dispatched.
"""

from __future__ import annotations

import gllm.cli as cli
from gllm.domain import Response


class _FakeProvider:
    """Records the Request it is handed and returns a canned Response."""

    last_request = None

    def generate(self, request):
        _FakeProvider.last_request = request
        return Response(text="ok", model=request.model, provider="fake")


def _wire(monkeypatch, *, default_effort=None):
    """Isolate main() from the host env and the network."""
    # Don't let the repo .env leak DEFAULT_MODEL/EFFORT/keys into the test.
    monkeypatch.setattr(cli, "_load_user_env_file", lambda *_: None)
    monkeypatch.setattr(cli, "_build_provider", lambda _name: _FakeProvider())
    monkeypatch.setattr(cli, "_read_stdin_if_piped", lambda: "hej")
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("WORK", raising=False)
    monkeypatch.delenv("WORK_ENV", raising=False)
    if default_effort is None:
        monkeypatch.delenv("DEFAULT_EFFORT", raising=False)
    else:
        monkeypatch.setenv("DEFAULT_EFFORT", default_effort)
    _FakeProvider.last_request = None


def test_ambient_effort_dropped_on_non_reasoning_model(monkeypatch, capsys):
    """$DEFAULT_EFFORT=low + gpt-4.1 (no reasoning control) -> drop, succeed."""
    _wire(monkeypatch, default_effort="low")

    rc = cli.main(["-m", "gpt-4.1-nano", "prompt"])

    assert rc == 0
    assert _FakeProvider.last_request is not None
    assert _FakeProvider.last_request.reasoning is None


def test_explicit_reasoning_fails_loud_on_non_reasoning_model(monkeypatch, capsys):
    """Explicit -r low + gpt-4.1 -> hard error, no dispatch."""
    _wire(monkeypatch)

    rc = cli.main(["-m", "gpt-4.1-nano", "-r", "low", "prompt"])

    assert rc == 2
    assert _FakeProvider.last_request is None
    assert "has no reasoning control" in capsys.readouterr().err


def test_ambient_effort_kept_on_reasoning_model(monkeypatch, capsys):
    """$DEFAULT_EFFORT=low + a reasoning-capable model -> reasoning is sent."""
    _wire(monkeypatch, default_effort="low")

    rc = cli.main(["-m", "claude-haiku-4-5", "prompt"])

    assert rc == 0
    assert _FakeProvider.last_request.reasoning == "low"
