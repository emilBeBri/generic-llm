"""Minimal JSON-over-HTTPS transport — the vendor SDKs' replacement.

Every provider gllm talks to is a JSON POST behind a bearer-ish header, and
every adapter already hand-builds the request body as a plain dict. The vendor
SDKs added nothing to that except import cost: `import anthropic` alone builds
462 pydantic model classes and loads 1247 modules (~670 ms) to validate a dict
we already had and serialise it back to the same JSON. This module is the
~200 lines that does the same work in ~25 ms.

`http.client` rather than `urllib.request`: urllib's opener/handler stack costs
an extra ~24 ms of import for redirect, proxy, auth and cookie machinery that a
one-shot API call never uses, and it hides the status/headers/socket control
that retries and (later) SSE need.

Two layers, deliberately separate:
- `post_json` / `get_json` return the decoded JSON verbatim — dicts and lists.
- `wrap` puts an attribute-access view over that (`resp.choices[0].message
  .content`), so the adapters' existing response parsing and ALL of
  `gllm.usage` keep working unchanged. `Obj.to_dict()` is what `usage._to_plain`
  finds instead of pydantic's `model_dump`, which makes `usage_raw` the
  provider's own bytes rather than a re-serialisation of them.

No connection pooling (one request per process), no redirect following (these
APIs don't redirect; a 3xx is a misconfigured base URL and should say so), and
no `Accept-Encoding` (we don't want to decompress). SSE is deliberately absent
until the Anthropic adapter — the only caller that needs it — lands.
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import sys
import time
from typing import Any
from urllib.parse import urlsplit

# Matches the Anthropic SDK's 10-minute default. A long extended-thinking
# generation can sit silent well past any "sensible" HTTP timeout, and a
# truncated request costs real money, so the socket waits.
DEFAULT_TIMEOUT = float(os.environ.get("GLLM_HTTP_TIMEOUT") or 600)
DEFAULT_RETRIES = 3

# Ceiling on TOTAL time spent sleeping between retries, across all of them.
#
# Honouring `Retry-After` is correct; doing it silently for 60s x 3 retries is
# not. gllm is a one-shot CLI, and a process that goes mute for three minutes
# reads as a hang — it was diagnosed as one during development before the
# backoff turned out to be the cause. So the waiting is capped and every wait is
# announced on stderr. Not silenced by -q: that flag is --quiet-effort, and this
# is the difference between "slow" and "broken".
RETRY_BUDGET = float(os.environ.get("GLLM_RETRY_BUDGET") or 30)

# Retried with backoff. 429 is rate limiting, 5xx is the provider's problem,
# 408/409 are the two request-level codes these APIs use for "try again".
_RETRY_STATUS = frozenset({408, 409, 429}) | frozenset(range(500, 600))

# Error bodies are usually a small JSON object, but a misrouted request can
# return a full HTML error page; keep the exception readable.
_MAX_ERROR_BODY = 2000


class APIError(RuntimeError):
    """A non-2xx response, surfaced with the provider's own body attached.

    `cli.main` prints `gllm: APIError: <this message>`, so the message has to
    stand alone — status, endpoint, and whatever the provider said about why.
    """

    def __init__(self, status: int, url: str, body: str):
        self.status = status
        self.url = url
        self.body = body
        if len(body) > _MAX_ERROR_BODY:
            body = body[:_MAX_ERROR_BODY] + f"... [{len(body) - _MAX_ERROR_BODY} more chars]"
        super().__init__(f"HTTP {status} from {url}: {body.strip() or '<empty body>'}")


class Obj:
    """Attribute-access view over a decoded JSON object.

    Exists so that response parsing written against the SDKs' pydantic models
    (`resp.choices[0].message.content`, `getattr(usage, "cached_tokens", 0)`)
    keeps working against plain dicts. Read-only and lazy — children are wrapped
    on access, not up front.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_d")
        if name not in d:
            # Loud and specific: a missing field means the provider changed its
            # response shape, and the available keys are the whole diagnosis.
            raise AttributeError(
                f"response object has no {name!r} (keys: {sorted(d)})"
            )
        return wrap(d[name])

    def to_dict(self) -> dict:
        """The underlying JSON, verbatim. `usage._to_plain` looks for this."""
        return object.__getattribute__(self, "_d")

    def __repr__(self) -> str:
        return f"Obj({object.__getattribute__(self, '_d')!r})"


def wrap(value: Any) -> Any:
    """Recursively put `Obj` over dicts, leaving lists and scalars alone."""
    if isinstance(value, dict):
        return Obj(value)
    if isinstance(value, list):
        return [wrap(v) for v in value]
    return value


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_RETRIES,
) -> Any:
    """POST `payload` as JSON, return the decoded response body."""
    return _request(
        "POST", url, headers, payload, timeout=timeout, max_retries=max_retries
    )


def get_json(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_RETRIES,
) -> Any:
    """GET and return the decoded response body (used by `--models`)."""
    return _request(
        "GET", url, headers, None, timeout=timeout, max_retries=max_retries
    )


def _request(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict | None,
    *,
    timeout: float,
    max_retries: int,
) -> Any:
    body = json.dumps(payload).encode() if payload is not None else None
    sent = {
        "Accept": "application/json",
        "User-Agent": "gllm",
        **headers,
    }
    if body is not None:
        sent["Content-Type"] = "application/json"

    last: Exception | None = None
    budget = RETRY_BUDGET
    for attempt in range(max_retries + 1):
        try:
            status, raw, retry_after = _once(method, url, sent, body, timeout)
        except (OSError, http.client.HTTPException) as e:
            # Connection reset, DNS failure, TLS hiccup, socket timeout. Worth a
            # retry; if it is the last one, the original error is the truth.
            last = e
            if attempt == max_retries:
                raise
            waited = _wait_before_retry(
                attempt, None, budget, url, type(e).__name__, max_retries
            )
            if waited is None:
                raise
            budget -= waited
            continue

        if status in _RETRY_STATUS and attempt < max_retries:
            waited = _wait_before_retry(
                attempt, retry_after, budget, url, f"HTTP {status}", max_retries
            )
            if waited is None:
                raise APIError(
                    status,
                    url,
                    raw.decode("utf-8", "replace")
                    + f" [gllm gave up: the {RETRY_BUDGET:g}s retry budget would "
                    f"be exceeded; raise GLLM_RETRY_BUDGET to wait longer]",
                )
            budget -= waited
            continue
        if not 200 <= status < 300:
            raise APIError(status, url, raw.decode("utf-8", "replace"))

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{url} returned HTTP {status} with a body that is not JSON "
                f"({e}): {raw[:_MAX_ERROR_BODY]!r}"
            ) from e

    # Unreachable: the loop either returns, raises, or continues. Kept so a
    # future edit to the retry conditions cannot fall through silently.
    raise RuntimeError(f"{url}: retries exhausted without a verdict ({last})")


def _once(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, bytes, str | None]:
    """One request/response round trip. Returns (status, body, Retry-After)."""
    parts = urlsplit(url)
    if parts.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parts.hostname or "",
            parts.port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
    elif parts.scheme == "http":
        # Plain HTTP exists for one reason: the loopback key broker (see
        # config.resolve_base_url). Not a fallback for a failed TLS handshake.
        conn = http.client.HTTPConnection(
            parts.hostname or "", parts.port, timeout=timeout
        )
    else:
        raise RuntimeError(f"unsupported URL scheme {parts.scheme!r} in {url!r}")

    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    try:
        conn.request(method, target, body=body, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read(), resp.getheader("Retry-After")
    finally:
        conn.close()


def _wait_before_retry(
    attempt: int,
    retry_after: str | None,
    budget: float,
    url: str,
    reason: str,
    max_retries: int,
) -> float | None:
    """Sleep before the next attempt and say so. Returns seconds slept.

    Returns **None** instead when the wait would blow the remaining retry
    budget, which tells the caller to give up now and report the real error
    rather than sit on it. That is the whole point: a one-shot CLI should fail in
    seconds with a reason, not succeed in three minutes of silence.
    """
    delay = _retry_delay(attempt, retry_after)
    if delay > budget:
        print(
            f"gllm: {reason} from {urlsplit(url).netloc}; giving up rather than "
            f"waiting {delay:g}s (only {max(budget, 0):g}s of retry budget left)",
            file=sys.stderr,
        )
        return None
    print(
        f"gllm: {reason} from {urlsplit(url).netloc}, retrying in {delay:g}s "
        f"({attempt + 1}/{max_retries})",
        file=sys.stderr,
    )
    time.sleep(delay)
    return delay


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """`Retry-After` seconds when the provider gave a usable one, else backoff.

    `Retry-After` may also be an HTTP-date; parsing that would pull in
    `email.utils` for a header these APIs send as an integer, so a non-integer
    value falls through to backoff rather than being decoded.
    """
    if retry_after:
        try:
            return float(retry_after.strip())
        except ValueError:
            pass
    # Jitter from os.urandom keeps `random` (and its ~5 ms of imports) out of
    # the startup path we are here to shrink.
    jitter = os.urandom(1)[0] / 255.0
    return min(0.5 * (2**attempt) + jitter, 30.0)
