"""Truncation detection: `Response.truncated` and the per-provider vocabularies.

A cut-off answer is indistinguishable from a complete one on stdout — the bug
that motivated this was a capped Gemini call printing `1` for `23*47`, which
reads as a confident wrong answer rather than a truncated one. Each provider
spells the reason differently, so the mapping is what these tests pin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gllm._http import wrap
from gllm.adapters import gemini as gemini_mod
from gllm.adapters import openai as openai_mod
from gllm.domain import Response


def _resp(stop_reason):
    return Response(text="x", model="m", provider="p", stop_reason=stop_reason)


# --- the normalisation ----------------------------------------------------

@pytest.mark.parametrize("reason", ["max_tokens", "max_output_tokens", "length"])
def test_budget_exhaustion_reasons_are_truncation(reason):
    assert _resp(reason).truncated


def test_gemini_enum_name_is_matched_case_insensitively():
    assert _resp("MAX_TOKENS").truncated


@pytest.mark.parametrize("reason", ["stop", "end_turn", "tool_use", "STOP", None, ""])
def test_normal_completions_are_not_truncation(reason):
    assert not _resp(reason).truncated


def test_an_unknown_reason_is_not_reported_as_truncation():
    """Better a missed warning than a false one: gllm does not guess that an
    unrecognised reason means the budget ran out."""
    assert not _resp("content_filter").truncated


# --- per-provider extraction ---------------------------------------------

def test_openai_chat_reads_finish_reason():
    resp = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])
    assert openai_mod._finish_reason(resp) == "length"


def test_openai_responses_reads_incomplete_details():
    resp = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    assert openai_mod._incomplete_reason(resp) == "max_output_tokens"


def test_openai_responses_complete_has_no_reason():
    assert openai_mod._incomplete_reason(SimpleNamespace(incomplete_details=None)) is None


def test_gemini_unwraps_the_enum_to_its_name():
    """str() on the enum gives 'FinishReason.MAX_TOKENS', which matches nothing —
    .name is the comparable value."""
    enum_like = SimpleNamespace(name="MAX_TOKENS")
    resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason=enum_like)])
    assert gemini_mod._finish_reason(resp) == "MAX_TOKENS"


def test_gemini_tolerates_a_plain_string_reason():
    resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])
    assert gemini_mod._finish_reason(resp) == "STOP"


def test_gemini_with_no_candidates_reports_nothing():
    assert gemini_mod._finish_reason(SimpleNamespace(candidates=[])) is None


def test_chat_extraction_works_through_the_http_wrapper():
    """The stdlib transport hands adapters an `Obj`, not a pydantic model."""
    resp = wrap({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
    assert openai_mod._finish_reason(resp) == "length"


def test_extraction_survives_a_response_with_no_choices():
    assert openai_mod._finish_reason(SimpleNamespace(choices=[])) is None
