"""Call log: path resolution, append semantics, and the CLI wiring.

The log is opt-in because pricing a call loads the price book (~150ms of
pydantic) that plain runs deliberately skip. These tests pin both halves of
that: OFF must cost nothing and write nothing, ON must produce a complete
record including cost and latency.
"""

from __future__ import annotations

import json

import gllm.calllog as calllog
import gllm.cli as cli
import gllm.pricing as pricing
from gllm.domain import Response
from llm_price_tracker.models import STANDARD, ModelEntry, Price, PriceBook


# --- path resolution --------------------------------------------------------
def test_off_by_default(monkeypatch):
    monkeypatch.delenv("GLLM_CALL_LOG", raising=False)
    assert calllog.log_path() is None
    assert calllog.enabled() is False


def test_falsey_words_are_off(monkeypatch):
    for value in ("", "0", "off", "false", "no", "OFF", " off "):
        monkeypatch.setenv("GLLM_CALL_LOG", value)
        assert calllog.log_path() is None, value


def test_truthy_words_mean_the_default_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    for value in ("1", "on", "true", "YES"):
        monkeypatch.setenv("GLLM_CALL_LOG", value)
        assert calllog.log_path() == tmp_path / "gllm" / "calls.jsonl", value


def test_default_path_is_state_not_config(monkeypatch):
    # State, not config and not bundled data/: per-machine history that must
    # never end up in the repo.
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/nobody")
    assert calllog.default_path().parts[-4:] == (".local", "state", "gllm", "calls.jsonl")


def test_anything_else_is_a_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GLLM_CALL_LOG", str(tmp_path / "x.jsonl"))
    assert calllog.log_path() == tmp_path / "x.jsonl"


def test_path_expands_user_and_vars(monkeypatch):
    monkeypatch.setenv("HOME", "/home/nobody")
    monkeypatch.setenv("SOMEDIR", "/srv/logs")
    monkeypatch.setenv("GLLM_CALL_LOG", "~/a.jsonl")
    assert str(calllog.log_path()) == "/home/nobody/a.jsonl"
    monkeypatch.setenv("GLLM_CALL_LOG", "$SOMEDIR/b.jsonl")
    assert str(calllog.log_path()) == "/srv/logs/b.jsonl"


# --- append -----------------------------------------------------------------
def test_append_creates_parents_and_writes_one_line_per_record(monkeypatch, tmp_path):
    target = tmp_path / "deep" / "nested" / "calls.jsonl"
    monkeypatch.setenv("GLLM_CALL_LOG", str(target))

    calllog.append({"n": 1})
    calllog.append({"n": 2})

    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["n"] for x in lines] == [1, 2]


def test_append_is_a_noop_when_off(monkeypatch, tmp_path):
    monkeypatch.delenv("GLLM_CALL_LOG", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    calllog.append({"n": 1})
    assert not (tmp_path / "gllm").exists()


def test_a_broken_path_warns_but_does_not_raise(monkeypatch, tmp_path, capsys):
    # The answer is already on stdout by the time this runs; a log failure must
    # not turn a successful call into a failed command. But it must be audible.
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("GLLM_CALL_LOG", str(blocker / "calls.jsonl"))

    calllog.append({"n": 1})

    assert "call log write failed" in capsys.readouterr().err


def test_unjsonable_values_degrade_rather_than_lose_the_record(monkeypatch, tmp_path):
    target = tmp_path / "calls.jsonl"
    monkeypatch.setenv("GLLM_CALL_LOG", str(target))

    calllog.append({"ok": 1, "weird": object()})

    row = json.loads(target.read_text(encoding="utf-8"))
    assert row["ok"] == 1
    assert isinstance(row["weird"], str)


# --- CLI wiring -------------------------------------------------------------
BOOK = PriceBook(
    updated_at="2026-08-19",
    models={
        "claude-haiku-4.5": ModelEntry(
            id="claude-haiku-4.5", vendor="test",
            tiers={STANDARD: Price(input=1, output=5)},
        )
    },
)


class _FakeProvider:
    def generate(self, request):
        return Response(
            text="ok", model="claude-haiku-4-5", provider="anthropic",
            input_tokens=1_000_000, output_tokens=0,
        )


class _Boom:
    def generate(self, request):
        raise RuntimeError("upstream exploded")


def _wire(monkeypatch, provider=None):
    monkeypatch.setattr(cli, "_load_user_env_file", lambda *_: None)
    monkeypatch.setattr(cli, "_build_provider", lambda _n: provider or _FakeProvider())
    monkeypatch.setattr(cli, "_read_stdin_if_piped", lambda: "hej")
    monkeypatch.setattr(pricing, "_load_book", lambda: BOOK)
    monkeypatch.setattr(pricing, "load_overrides", lambda: {})
    for var in ("DEFAULT_MODEL", "DEFAULT_EFFORT", "WORK", "WORK_ENV"):
        monkeypatch.delenv(var, raising=False)


def test_a_call_is_logged_with_cost_and_latency(monkeypatch, tmp_path):
    target = tmp_path / "calls.jsonl"
    monkeypatch.setenv("GLLM_CALL_LOG", str(target))
    _wire(monkeypatch)

    assert cli.main(["-m", "claude-haiku-4-5", "prompt"]) == 0

    row = json.loads(target.read_text(encoding="utf-8"))
    assert row["ok"] is True
    assert row["model"] == "claude-haiku-4-5"
    assert row["cost_usd"] == 1.0          # priced even without --usage
    assert row["input_tokens"] == 1_000_000
    assert isinstance(row["elapsed_s"], float)
    assert row["ts"].endswith("+00:00")    # UTC, not naive local
    assert row["response_chars"] == 2


def test_the_log_records_lengths_never_the_text(monkeypatch, tmp_path):
    # A file that silently becomes a transcript of every prompt is a privacy
    # problem. Lengths answer the cost/latency questions without the content.
    target = tmp_path / "calls.jsonl"
    monkeypatch.setenv("GLLM_CALL_LOG", str(target))
    _wire(monkeypatch)

    cli.main(["-m", "claude-haiku-4-5", "a-very-distinctive-secret-prompt"])

    raw = target.read_text(encoding="utf-8")
    assert "a-very-distinctive-secret-prompt" not in raw
    assert json.loads(raw)["prompt_chars"] > 0


def test_a_failed_call_is_logged_too(monkeypatch, tmp_path):
    target = tmp_path / "calls.jsonl"
    monkeypatch.setenv("GLLM_CALL_LOG", str(target))
    _wire(monkeypatch, provider=_Boom())

    assert cli.main(["-m", "claude-haiku-4-5", "prompt"]) == 1

    row = json.loads(target.read_text(encoding="utf-8"))
    assert row["ok"] is False
    assert "upstream exploded" in row["error"]
    assert isinstance(row["elapsed_s"], float)


def test_nothing_is_written_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("GLLM_CALL_LOG", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    _wire(monkeypatch)

    assert cli.main(["-m", "claude-haiku-4-5", "prompt"]) == 0
    assert not (tmp_path / "gllm").exists()


def test_usage_flag_still_prints_and_also_logs(monkeypatch, tmp_path, capsys):
    target = tmp_path / "calls.jsonl"
    monkeypatch.setenv("GLLM_CALL_LOG", str(target))
    _wire(monkeypatch)

    cli.main(["--usage", "-m", "claude-haiku-4-5", "prompt"])

    err = capsys.readouterr().err
    assert any(ln.startswith("gllm-usage ") for ln in err.splitlines())
    assert json.loads(target.read_text(encoding="utf-8"))["ok"] is True
