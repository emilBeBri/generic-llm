"""OpenAI attachment payloads, without network calls."""

from __future__ import annotations

import pytest

from gllm.adapters.openai import _chat_user_content, _responses_input
from gllm.domain import Attachment


def _attachment(
    source_label: str,
    mime_type: str,
    data: bytes = b"document",
) -> Attachment:
    return Attachment(
        data=data,
        mime_type=mime_type,
        source_label=source_label,
    )


def test_responses_encodes_office_file_and_image():
    content = _responses_input(
        "compare these",
        (
            _attachment(
                "report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            _attachment("chart.png", "image/png", b"image"),
        ),
        model="gpt-5.6",
    )[0]["content"]

    assert content[0]["type"] == "input_file"
    assert content[0]["filename"] == "report.docx"
    assert content[0]["file_data"].startswith(
        "data:application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document;base64,"
    )
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert content[2] == {"type": "input_text", "text": "compare these"}


def test_chat_encodes_pdf_and_spreadsheet_as_file_parts():
    content = _chat_user_content(
        "summarize",
        (
            _attachment("paper.pdf", "application/pdf"),
            _attachment(
                "data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        ),
        model="gpt-4o",
    )

    assert content[0] == {"type": "text", "text": "summarize"}
    assert content[1]["type"] == "file"
    assert content[1]["file"]["filename"] == "paper.pdf"
    assert content[1]["file"]["file_data"].startswith("data:application/pdf;base64,")
    assert content[2]["type"] == "file"
    assert content[2]["file"]["filename"] == "data.xlsx"


@pytest.mark.parametrize(
    ("mime_type", "expected_filename"),
    [
        ("text/plain", "file.txt"),
        ("text/tsv", "file.tsv"),
        ("application/toml", "file.toml"),
        ("application/x-yaml", "file.yml"),
        ("application/x-rust", "file.rs"),
        ("text/x-c++", "file.cpp"),
    ],
)
def test_stdin_file_gets_a_mime_derived_filename(mime_type, expected_filename):
    content = _responses_input(
        "read",
        (_attachment("<stdin>", mime_type),),
        model="gpt-5.6",
    )[0]["content"]

    assert content[0]["filename"] == expected_filename


def test_mime_override_replaces_an_unsupported_filename_suffix():
    content = _responses_input(
        "read",
        (_attachment("notes.dat", "text/plain"),),
        model="gpt-5.6",
    )[0]["content"]

    assert content[0]["filename"] == "notes.txt"


@pytest.mark.parametrize(
    ("builder", "provider", "model"),
    [
        (_responses_input, "azure_openai", "gpt-5.1-dev"),
        (_chat_user_content, "azure_openai", "gpt-4.1-nano-dev"),
        (_responses_input, "grok", "grok-4.3"),
    ],
    ids=["azure-responses", "azure-chat", "grok"],
)
def test_non_pdf_documents_are_rejected_by_compatible_hosts(
    builder,
    provider,
    model,
):
    docx = _attachment(
        "report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    with pytest.raises(RuntimeError, match="cannot encode attachment"):
        builder("read", (docx,), provider=provider, model=model)


def test_azure_chat_still_accepts_pdf():
    content = _chat_user_content(
        "read",
        (_attachment("paper.pdf", "application/pdf"),),
        provider="azure_openai",
        model="gpt-4.1-nano-dev",
    )

    assert content[1]["type"] == "file"


def test_unsupported_public_openai_file_is_rejected():
    archive = _attachment("bundle.zip", "application/zip")

    with pytest.raises(RuntimeError, match="cannot encode attachment"):
        _responses_input("inspect", (archive,), model="gpt-5.6")
