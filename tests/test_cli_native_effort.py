"""main()-level tests for `--native-effort`.

The contract: by default gllm's four rungs are TRANSLATED onto each model's own
vocabulary — `-r xhigh` means "the most this model has", which lands on `max` for
DeepSeek, GLM, Claude-5 and GPT-5.6. That portability is the right default and
the wrong behaviour when you need to know exactly what reached the provider, so
`--native-effort` turns the translation off and passes `-r` through verbatim.

The value must then be one the model actually has: `xhigh` is a real rung on
claude-opus-5 (below `max`) and does not exist at all on deepseek-v4-flash. That
asymmetry is the confusion the flag exists to remove, so it is asserted here.
"""

from __future__ import annotations

import gllm.cli as cli
from gllm.domain import Response


class _FakeProvider:
    last_request = None

    def generate(self, request):
        _FakeProvider.last_request = request
        return Response(text="ok", model=request.model, provider="fake")


def _wire(monkeypatch, *, default_effort=None):
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


# --- the default stays translated --------------------------------------------

def test_translated_xhigh_still_becomes_max(monkeypatch, capsys):
    """Without the flag, nothing changes: xhigh -> the model's top rung."""
    _wire(monkeypatch)

    rc = cli.main(["-m", "deepseek-v4-flash", "-r", "xhigh", "prompt"])

    assert rc == 0
    # `reasoning` keeps the rung that was ASKED for; `wire_effort` is what is
    # actually sent. The split is what lets usage logging report both.
    assert _FakeProvider.last_request.reasoning == "xhigh"
    assert _FakeProvider.last_request.wire_effort == "max"
    assert "-r xhigh -> 'max'" in capsys.readouterr().err


def test_translated_mode_rejects_a_native_value_and_names_the_flag(monkeypatch, capsys):
    """`-r max` is not a gllm rung. The error should point at the way to get it."""
    _wire(monkeypatch)

    rc = cli.main(["-m", "deepseek-v4-flash", "-r", "max", "prompt"])

    assert rc == 2
    assert _FakeProvider.last_request is None
    err = capsys.readouterr().err
    assert "must be one of low, medium, high, xhigh" in err
    assert "--native-effort" in err


# --- --native-effort passes the value through --------------------------------

def test_native_effort_sends_the_value_verbatim(monkeypatch, capsys):
    _wire(monkeypatch)

    rc = cli.main(["-m", "deepseek-v4-flash", "-r", "max", "--native-effort", "prompt"])

    assert rc == 0
    assert _FakeProvider.last_request.wire_effort == "max"
    # nothing was remapped, so there is no remap notice to print
    assert "->" not in capsys.readouterr().err


def test_native_effort_rejects_a_rung_the_model_lacks(monkeypatch, capsys):
    """deepseek-v4-flash offers high|max — it has no `xhigh` at all."""
    _wire(monkeypatch)

    rc = cli.main(["-m", "deepseek-v4-flash", "-r", "xhigh", "--native-effort", "prompt"])

    assert rc == 2
    assert _FakeProvider.last_request is None
    assert "has no effort 'xhigh'" in capsys.readouterr().err


def test_native_xhigh_is_valid_where_the_model_really_has_it(monkeypatch):
    """The mirror image: claude-opus-5 has a real xhigh, distinct from max."""
    _wire(monkeypatch)

    rc = cli.main(["-m", "claude-opus-5", "-r", "xhigh", "--native-effort", "prompt"])

    assert rc == 0
    # Untranslated: xhigh is sent as xhigh, NOT silently promoted to max the way
    # the default ladder would promote it.
    assert _FakeProvider.last_request.wire_effort == "xhigh"


def test_native_effort_requires_an_explicit_r(monkeypatch, capsys):
    """It must not inherit $DEFAULT_EFFORT: that is a portable rung, and feeding
    it to a provider as a native value is the very confusion this flag removes."""
    _wire(monkeypatch, default_effort="high")

    rc = cli.main(["-m", "deepseek-v4-flash", "--native-effort", "prompt"])

    assert rc == 2
    assert _FakeProvider.last_request is None
    assert "needs an explicit -r" in capsys.readouterr().err


def test_native_effort_still_refused_on_a_knobless_model(monkeypatch, capsys):
    """The capability gate is not bypassed by asking for a native value."""
    _wire(monkeypatch)

    rc = cli.main(["-m", "gpt-4o", "-r", "high", "--native-effort", "prompt"])

    assert rc == 2
    assert _FakeProvider.last_request is None
    assert "has no reasoning control" in capsys.readouterr().err
