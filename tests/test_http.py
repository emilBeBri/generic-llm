"""The stdlib JSON transport that replaced the vendor SDKs.

`gllm._http` is the one module in the tree that touches a socket, so it is
tested against a real loopback HTTP server rather than a mock: status handling,
retry/backoff and header plumbing are exactly the parts a mock would assume
correct. Retry *delays* are asserted by intercepting `time.sleep` — the point is
the computed delay, not spending it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gllm import _http
from gllm._http import APIError, Obj, get_json, post_json, wrap


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _serve(self) -> None:
        script = self.server.script
        received = {
            "method": self.command,
            "target": self.path,
            "headers": dict(self.headers),
            "body": None,
        }
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            received["body"] = json.loads(self.rfile.read(length))
        self.server.requests.append(received)

        status, payload, headers = script.pop(0) if script else (200, {"ok": True}, {})
        raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = _serve
    do_POST = _serve

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass


@pytest.fixture
def server():
    """A loopback server. `srv.script` is a queue of (status, payload, headers)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.script = []
    srv.requests = []
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    srv.url = f"http://127.0.0.1:{srv.server_port}"
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


@pytest.fixture
def no_sleep(monkeypatch):
    """Record retry delays instead of serving them."""
    slept: list[float] = []
    monkeypatch.setattr(_http.time, "sleep", slept.append)
    return slept


# --- wrap / Obj -------------------------------------------------------------

def test_wrap_gives_sdk_shaped_attribute_access():
    resp = wrap({
        "model": "deepseek-v4-pro",
        "choices": [{"message": {"content": "hej", "role": "assistant"}}],
    })
    assert resp.model == "deepseek-v4-pro"
    assert resp.choices[0].message.content == "hej"
    assert isinstance(resp.choices, list)


def test_wrap_leaves_scalars_and_lists_of_scalars_alone():
    assert wrap(3) == 3
    assert wrap(None) is None
    assert wrap(["a", "b"]) == ["a", "b"]


def test_missing_field_names_the_available_keys():
    resp = wrap({"prompt_tokens": 10})
    with pytest.raises(AttributeError) as exc:
        _ = resp.completion_tokens
    assert "completion_tokens" in str(exc.value)
    assert "prompt_tokens" in str(exc.value)


def test_getattr_default_works_so_usage_mappers_are_unchanged():
    usage = wrap({"prompt_tokens": 10})
    assert getattr(usage, "completion_tokens_details", None) is None
    assert getattr(usage, "prompt_tokens", 0) == 10


def test_to_dict_is_what_usage_to_plain_finds():
    raw = {"prompt_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 4}}
    obj = wrap(raw)
    assert isinstance(obj, Obj)
    # Verbatim, not a re-serialisation: usage_raw is the ground truth for cost.
    assert obj.to_dict() is raw
    assert getattr(obj, "model_dump", None) is None


# --- request/response plumbing ---------------------------------------------

def test_post_json_sends_body_headers_and_returns_decoded(server):
    server.script.append((200, {"choices": [{"message": {"content": "ok"}}]}, {}))

    out = post_json(
        f"{server.url}/v1/chat/completions",
        {"Authorization": "Bearer sk-test"},
        {"model": "m", "messages": []},
    )

    assert out == {"choices": [{"message": {"content": "ok"}}]}
    sent = server.requests[0]
    assert sent["method"] == "POST"
    assert sent["target"] == "/v1/chat/completions"
    assert sent["body"] == {"model": "m", "messages": []}
    assert sent["headers"]["Authorization"] == "Bearer sk-test"
    assert sent["headers"]["Content-Type"] == "application/json"
    assert sent["headers"]["User-Agent"] == "gllm"


def test_get_json_keeps_the_query_string(server):
    server.script.append((200, {"data": []}, {}))
    get_json(f"{server.url}/v1/models?limit=5", {})
    assert server.requests[0]["method"] == "GET"
    assert server.requests[0]["target"] == "/v1/models?limit=5"
    assert server.requests[0]["body"] is None


def test_api_error_carries_status_and_provider_body(server):
    server.script.append((400, {"error": {"message": "bad thinking type"}}, {}))

    with pytest.raises(APIError) as exc:
        post_json(f"{server.url}/v1/chat/completions", {}, {"model": "m"})

    assert exc.value.status == 400
    assert "bad thinking type" in str(exc.value)
    assert len(server.requests) == 1, "4xx must not be retried"


def test_body_that_is_not_json_fails_loudly(server):
    server.script.append((200, "<html>gateway</html>", {}))
    with pytest.raises(RuntimeError, match="not JSON"):
        post_json(f"{server.url}/v1/chat/completions", {}, {"model": "m"})


def test_unsupported_scheme_is_refused():
    with pytest.raises(RuntimeError, match="unsupported URL scheme"):
        post_json("ftp://example.invalid/v1", {}, {})


# --- retries ---------------------------------------------------------------

def test_429_is_retried_and_honours_retry_after(server, no_sleep):
    server.script.append((429, {"error": "slow down"}, {"Retry-After": "2"}))
    server.script.append((200, {"ok": True}, {}))

    assert post_json(f"{server.url}/v1", {}, {"model": "m"}) == {"ok": True}
    assert len(server.requests) == 2
    assert no_sleep == [2.0], "Retry-After beats the backoff curve"


def test_500_is_retried_with_exponential_backoff(server, no_sleep):
    server.script.extend([
        (500, {"error": "boom"}, {}),
        (503, {"error": "boom"}, {}),
        (200, {"ok": True}, {}),
    ])

    assert post_json(f"{server.url}/v1", {}, {"model": "m"}) == {"ok": True}
    assert len(no_sleep) == 2
    # 0.5 * 2**attempt, plus up to 1.0 of jitter.
    assert 0.5 <= no_sleep[0] < 1.5
    assert 1.0 <= no_sleep[1] < 2.0


def test_retry_after_that_is_an_http_date_falls_back_to_backoff(server, no_sleep):
    server.script.append((429, {}, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}))
    server.script.append((200, {"ok": True}, {}))

    post_json(f"{server.url}/v1", {}, {"model": "m"})
    assert 0.5 <= no_sleep[0] < 1.5


def test_exhausted_retries_raise_the_last_status(server, no_sleep):
    server.script.extend([(503, {"error": "down"}, {})] * 4)

    with pytest.raises(APIError) as exc:
        post_json(f"{server.url}/v1", {}, {"model": "m"}, max_retries=3)

    assert exc.value.status == 503
    assert len(server.requests) == 4, "1 attempt + 3 retries"
    assert len(no_sleep) == 3


def test_connection_failure_is_retried_then_raised(no_sleep):
    # Port 1 on loopback: nothing listens, so this is a connection refusal
    # rather than an HTTP status.
    with pytest.raises(OSError):
        post_json("http://127.0.0.1:1/v1", {}, {"model": "m"}, max_retries=2)
    assert len(no_sleep) == 2


# --- the retry budget ------------------------------------------------------
#
# Honouring Retry-After is right; doing it silently for 60s x 3 is not. A
# one-shot CLI that goes mute for three minutes reads as a hang — it was
# misdiagnosed as one during development before the backoff turned out to be the
# cause.

def test_a_retry_after_beyond_the_budget_gives_up_instead_of_waiting(server, no_sleep):
    server.script.append((429, {"error": "slow down"}, {"Retry-After": "600"}))

    with pytest.raises(APIError) as exc:
        post_json(f"{server.url}/v1", {}, {"model": "m"})

    assert exc.value.status == 429
    assert "gave up" in str(exc.value)
    assert "GLLM_RETRY_BUDGET" in str(exc.value), "the message must name the escape hatch"
    assert no_sleep == [], "it must not sleep at all before giving up"
    assert len(server.requests) == 1


def test_giving_up_says_why_on_stderr(server, no_sleep, capsys):
    server.script.append((429, {}, {"Retry-After": "600"}))
    with pytest.raises(APIError):
        post_json(f"{server.url}/v1", {}, {"model": "m"})
    err = capsys.readouterr().err
    assert "HTTP 429" in err
    assert "giving up" in err
    assert "600s" in err


def test_each_retry_is_announced_with_its_attempt_number(server, no_sleep, capsys):
    server.script.extend([(503, {}, {}), (503, {}, {}), (200, {"ok": True}, {})])

    assert post_json(f"{server.url}/v1", {}, {"model": "m"}) == {"ok": True}

    lines = [ln for ln in capsys.readouterr().err.splitlines() if "retrying" in ln]
    assert len(lines) == 2
    assert "(1/3)" in lines[0]
    assert "(2/3)" in lines[1]
    assert "HTTP 503" in lines[0]
    # The host, not the full URL — the path is noise in a progress line.
    assert "127.0.0.1" in lines[0]


def test_the_budget_is_spent_across_retries_not_per_retry(server, monkeypatch, capsys):
    """Three 12s waits would be 36s; the 30s budget stops the third."""
    monkeypatch.setattr(_http, "RETRY_BUDGET", 30.0)
    slept: list[float] = []
    monkeypatch.setattr(_http.time, "sleep", slept.append)
    server.script.extend([(429, {}, {"Retry-After": "12"})] * 4)

    with pytest.raises(APIError) as exc:
        post_json(f"{server.url}/v1", {}, {"model": "m"})

    assert slept == [12.0, 12.0], "third wait exceeds the 6s remaining"
    assert "gave up" in str(exc.value)


def test_a_connection_failure_beyond_the_budget_raises_the_original_error(monkeypatch):
    """Not an APIError: there was no response, so the socket error is the truth."""
    monkeypatch.setattr(_http, "RETRY_BUDGET", 0.0)
    with pytest.raises(OSError):
        post_json("http://127.0.0.1:1/v1", {}, {"model": "m"}, max_retries=3)
