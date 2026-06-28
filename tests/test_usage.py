"""Usage extraction (gllm.usage) + the --usage CLI emission.

The mappers run against lightweight fakes shaped like each provider's SDK usage
object — no network. The CLI test mocks the provider boundary and asserts the
machine-readable `gllm-usage ` line parses with the normalised fields.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import gllm.cli as cli
from gllm.domain import Response
from gllm.usage import (
    from_anthropic,
    from_deepseek,
    from_gemini,
    from_openai_chat,
    from_openai_responses,
)


def test_anthropic_maps_cache_read_and_write():
    u = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=120,
    )
    out = from_anthropic(u)
    assert out["input_tokens"] == 1000
    assert out["output_tokens"] == 200
    assert out["cache_read_tokens"] == 800
    assert out["cache_write_tokens"] == 120
    assert out["reasoning_tokens"] == 0  # Anthropic folds thinking into output
    assert out["usage_raw"]["cache_read_input_tokens"] == 800


def test_openai_chat_maps_cached_and_reasoning_details():
    u = SimpleNamespace(
        prompt_tokens=500,
        completion_tokens=300,
        prompt_tokens_details=SimpleNamespace(cached_tokens=450),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=210),
    )
    out = from_openai_chat(u)
    assert out["input_tokens"] == 500
    assert out["output_tokens"] == 300
    assert out["cache_read_tokens"] == 450
    assert out["reasoning_tokens"] == 210


def test_openai_responses_maps_details():
    u = SimpleNamespace(
        input_tokens=400,
        output_tokens=600,
        input_tokens_details=SimpleNamespace(cached_tokens=100),
        output_tokens_details=SimpleNamespace(reasoning_tokens=350),
    )
    out = from_openai_responses(u)
    assert out["cache_read_tokens"] == 100
    assert out["reasoning_tokens"] == 350


def test_gemini_maps_cached_content_and_thoughts():
    u = SimpleNamespace(
        prompt_token_count=900,
        candidates_token_count=250,
        cached_content_token_count=700,
        thoughts_token_count=180,
    )
    out = from_gemini(u)
    assert out["input_tokens"] == 900
    assert out["output_tokens"] == 250
    assert out["cache_read_tokens"] == 700
    assert out["reasoning_tokens"] == 180


def test_deepseek_maps_prompt_cache_hit():
    u = SimpleNamespace(
        prompt_tokens=300,
        completion_tokens=100,
        prompt_cache_hit_tokens=256,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=40),
    )
    out = from_deepseek(u)
    assert out["cache_read_tokens"] == 256
    assert out["reasoning_tokens"] == 40


def test_mappers_are_none_safe():
    # A provider that returns no usage object must not raise — all zeros.
    for fn in (from_anthropic, from_openai_chat, from_openai_responses,
               from_gemini, from_deepseek):
        out = fn(None)
        assert out["input_tokens"] == 0 and out["usage_raw"] == {}


def test_missing_detail_fields_coerce_to_zero():
    # Chat usage with no *_details sub-objects (older/minimal responses).
    u = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    out = from_openai_chat(u)
    assert out["cache_read_tokens"] == 0 and out["reasoning_tokens"] == 0


# --- CLI emission ------------------------------------------------------------

class _FakeProvider:
    def generate(self, request):
        return Response(
            text="ok",
            model="claude-haiku-4-5",
            provider="anthropic",
            input_tokens=120,
            output_tokens=45,
            cache_read_tokens=30,
            cache_write_tokens=10,
            reasoning_tokens=12,
            usage_raw={"input_tokens": 120, "cache_read_input_tokens": 30},
        )


def _wire(monkeypatch):
    monkeypatch.setattr(cli, "_load_user_env_file", lambda *_: None)
    monkeypatch.setattr(cli, "_build_provider", lambda _name: _FakeProvider())
    monkeypatch.setattr(cli, "_read_stdin_if_piped", lambda: "hej")
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_EFFORT", raising=False)
    monkeypatch.delenv("WORK", raising=False)
    monkeypatch.delenv("WORK_ENV", raising=False)


def _usage_line(err: str) -> dict:
    for line in err.splitlines():
        if line.startswith("gllm-usage "):
            return json.loads(line[len("gllm-usage "):])
    raise AssertionError(f"no gllm-usage line in stderr:\n{err}")


def test_usage_flag_emits_parseable_json(monkeypatch, capsys):
    _wire(monkeypatch)

    rc = cli.main(["--usage", "-m", "claude-haiku-4-5", "prompt"])

    cap = capsys.readouterr()
    assert rc == 0
    assert cap.out.strip() == "ok"  # stdout stays pure model text
    rec = _usage_line(cap.err)
    assert rec["provider"] == "anthropic"
    assert rec["input_tokens"] == 120 and rec["output_tokens"] == 45
    assert rec["cache_read_tokens"] == 30 and rec["cache_write_tokens"] == 10
    assert rec["reasoning_tokens"] == 12
    assert rec["usage_raw"]["cache_read_input_tokens"] == 30


def test_no_usage_line_without_flag(monkeypatch, capsys):
    _wire(monkeypatch)

    cli.main(["-m", "claude-haiku-4-5", "prompt"])

    assert "gllm-usage " not in capsys.readouterr().err
