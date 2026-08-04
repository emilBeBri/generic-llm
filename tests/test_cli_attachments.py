"""CLI attachment gates and Request construction, without network calls."""

from __future__ import annotations

import pytest

from gllm import cli
from gllm.domain import Response


class _FakeProvider:
    last_request = None

    def generate(self, request):
        _FakeProvider.last_request = request
        return Response(text="ok", model=request.model, provider="fake")


def _wire(monkeypatch):
    monkeypatch.setattr(cli, "_load_user_env_file", lambda *_: None)
    monkeypatch.setattr(cli, "_build_provider", lambda _name: _FakeProvider())
    monkeypatch.setattr(cli, "_read_stdin_if_piped", lambda: None)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_EFFORT", raising=False)
    monkeypatch.delenv("WORK", raising=False)
    monkeypatch.delenv("WORK_ENV", raising=False)
    _FakeProvider.last_request = None


@pytest.mark.parametrize(
    ("model", "filename", "expected_mime"),
    [
        (
            "gpt-5.6",
            "report.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        ("gpt-4o", "notes.txt", "text/plain"),
        ("gpt-4.1", "data.tsv", "text/tsv"),
    ],
    ids=["responses-office", "chat-text", "chat-spreadsheet"],
)
def test_public_openai_documents_reach_provider(
    monkeypatch,
    tmp_path,
    model,
    filename,
    expected_mime,
):
    _wire(monkeypatch)
    path = tmp_path / filename
    path.write_bytes(b"file bytes")

    rc = cli.main(["-m", model, "-f", str(path), "inspect"])

    assert rc == 0
    attachment = _FakeProvider.last_request.attachments[0]
    assert attachment.source_label == str(path)
    assert attachment.mime_type == expected_mime


def test_openai_chat_pdf_reaches_provider(monkeypatch, tmp_path):
    _wire(monkeypatch)
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\n")

    rc = cli.main(["-m", "gpt-4o", "-f", str(path), "summarize"])

    assert rc == 0
    assert _FakeProvider.last_request.attachments[0].mime_type == "application/pdf"


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.1-dev",
        "claude-opus-5",
        "gemini-3.6-flash",
        "grok-4.3",
        "deepseek-v4-flash",
    ],
)
def test_non_pdf_documents_fail_before_dispatch(
    monkeypatch,
    tmp_path,
    capsys,
    model,
):
    _wire(monkeypatch)
    path = tmp_path / "report.docx"
    path.write_bytes(b"office bytes")

    rc = cli.main(["-m", model, "-f", str(path), "inspect"])

    captured = capsys.readouterr()
    assert rc == 2
    assert _FakeProvider.last_request is None
    assert "does not accept file type" in captured.err
