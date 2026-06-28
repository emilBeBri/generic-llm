"""gllm CLI.

Reads stdin if piped, takes an optional positional prompt, prints model text
to stdout, logs to stderr. Supports --json and --schema for structured output.

Examples:
    echo "rewrite this in haiku" | gllm
    gllm "what is 2+2?"
    cat file.txt | gllm "summarize this"
    gllm -m claude-opus-4-7 "..."
    gllm --schema ./schema.json "extract from: $TEXT"
    gllm --json "list 3 capitals as {country: capital}"
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

from . import pricing
from . import reasoning as reasoning_mod
from .adapters._capabilities import (
    supports_image,
    supports_pdf,
    supports_reasoning,
    supports_strict_schema,
)
from .config import work_env
from .domain import Attachment, Request
from .ports import LLMProvider
from .routing import effective_model, provider_for

DEFAULT_MODEL = "deepseek-v4-flash"
# Config and keys load from this repo's own .env (repo root, beside
# pyproject.toml), resolved relative to this file so it is found regardless of
# cwd. cli.py lives at <root>/src/gllm/cli.py, so parents[2] is the repo root.
# See .llm-memory/IDEAS-key-loading-secret-managers.md for the longer-term plan.
CONFIG_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_user_env_file(path: Path) -> None:
    """Read KEY=value lines from `path` into os.environ (without overriding
    anything already set).

    `path` is a configured key source, so a missing or unreadable file is
    surfaced loudly on stderr instead of swallowed — otherwise it manifests
    downstream as a baffling "missing API key" with no hint why (e.g. when a
    sandbox doesn't bind-mount the file). We warn rather than abort: keys may
    legitimately come from the inherited environment, and the per-adapter key
    check is the real fatal gate."""
    if not path.is_file():
        print(
            f"gllm: key file not found at {path}; "
            "relying on inherited environment for API keys.",
            file=sys.stderr,
        )
        return
    try:
        text = path.read_text()
    except OSError as e:
        print(f"gllm: failed to read key file {path}: {e}", file=sys.stderr)
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _read_stdin_if_piped() -> str | None:
    if sys.stdin.isatty():
        return None
    data = sys.stdin.read()
    return data if data else None


# (magic, mime) pairs. First match wins.
_MAGIC_BYTES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
)


def _sniff_mime(data: bytes, path_hint: Path | None = None) -> str | None:
    """Detect a MIME type from the leading bytes, with extension fallback.

    Returns None if both fail — the caller decides whether that's fatal."""
    head = data[:16]
    for magic, mime in _MAGIC_BYTES:
        if head.startswith(magic):
            return mime
    # WebP: RIFF....WEBP
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if path_hint is not None:
        guess, _ = mimetypes.guess_type(str(path_hint))
        if guess:
            return guess
    return None


def _load_attachment(spec: str, mime_override: str | None) -> Attachment:
    """Read one `-f` argument into an Attachment.

    `spec == "-"` reads stdin as bytes (caller is responsible for ensuring
    text-stdin isn't also being consumed). Anything else is an open() target
    — including process substitution paths like /dev/fd/63 from bash <(...).
    """
    if spec == "-":
        if sys.stdin.isatty():
            raise RuntimeError(
                "`-f -` requested but stdin is a TTY (nothing to read)."
            )
        data = sys.stdin.buffer.read()
        label = "<stdin>"
        path_hint: Path | None = None
    else:
        p = Path(spec)
        data = p.read_bytes()
        label = spec
        path_hint = p

    mime = mime_override or _sniff_mime(data, path_hint)
    if not mime:
        raise RuntimeError(
            f"could not determine MIME type for {label!r}; pass --mime TYPE."
        )
    return Attachment(data=data, mime_type=mime, source_label=label)


def _read_text_arg(value: str) -> str:
    """`@path` means read from a file; otherwise the literal string."""
    if value.startswith("@"):
        return Path(value[1:]).read_text()
    return value


def _load_schema(value: str) -> dict:
    """Schema may be inline JSON or `@path/to/schema.json` or a bare path
    ending in .json. Returns the parsed dict."""
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text())
    stripped = value.lstrip()
    if stripped.startswith("{"):
        return json.loads(value)
    return json.loads(Path(value).read_text())


def _build_provider(name: str) -> LLMProvider:
    if name == "anthropic":
        from .adapters.anthropic import AnthropicProvider

        return AnthropicProvider()
    if name == "openai":
        from .adapters.openai import OpenAIProvider

        return OpenAIProvider()
    if name == "gemini":
        from .adapters.gemini import GeminiProvider

        return GeminiProvider()
    if name == "deepseek":
        from .adapters.deepseek import DeepSeekProvider

        return DeepSeekProvider()
    if name == "grok":
        from .adapters.grok import GrokProvider

        return GrokProvider()
    if name == "zai":
        from .adapters.zai import ZaiProvider

        return ZaiProvider()
    if name == "azure_openai":
        from .adapters.azure_openai import AzureOpenAIProvider

        return AzureOpenAIProvider()
    if name == "azure_anthropic":
        from .adapters.azure_anthropic import AzureAnthropicProvider

        return AzureAnthropicProvider()
    raise ValueError(f"unknown provider: {name}")


# Providers whose live API exposes a model catalog we can probe. Azure Foundry
# is excluded: it's deployment-scoped (you list *your* deployments, not a global
# catalog), so it has no equivalent `models.list()`.
_LISTABLE_PROVIDERS = ("anthropic", "openai", "gemini", "grok", "deepseek", "zai")


def _run_models(which: str) -> int:
    """`gllm --models`: print live `provider<TAB>model-id` rows, one per line.

    Probes each provider's API for the models it ACTUALLY serves right now —
    the single source of truth — instead of a hand-maintained catalog that
    drifts out of sync (the failure that made an agent declare a live model
    "retired"). `which == "*"` probes every listable provider; otherwise just
    the named one. A provider with no key or a failing call gets a loud stderr
    line and is skipped — never a silent drop.
    """
    if which and which != "*":
        if which not in _LISTABLE_PROVIDERS:
            print(
                f"gllm: --models: unknown provider {which!r}; choose from "
                f"{', '.join(_LISTABLE_PROVIDERS)}.",
                file=sys.stderr,
            )
            return 2
        targets: tuple[str, ...] = (which,)
    else:
        targets = _LISTABLE_PROVIDERS

    any_ok = False
    for name in targets:
        try:
            models = _build_provider(name).list_models()
        except Exception as e:
            print(f"gllm: {name}: skipped ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        for mid in models:
            print(f"{name}\t{mid}")
        any_ok = True
    return 0 if any_ok else 1


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gllm",
        description="Pipe-friendly LLM CLI. Reads stdin if piped, prints to stdout.",
    )
    p.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Optional positional prompt. Combined with stdin if both are given.",
    )
    p.add_argument(
        "-m",
        "--model",
        default=None,
        help=f"Model name. Default: $DEFAULT_MODEL or {DEFAULT_MODEL}.",
    )
    p.add_argument(
        "--models",
        nargs="?",
        const="*",
        default=None,
        metavar="PROVIDER",
        help=(
            "List the text-generation models each provider's API serves live "
            "(one `provider<TAB>id` per line; pipe to rg/fzf). Optionally "
            "restrict to one: --models gemini. Ignores the prompt."
        ),
    )
    p.add_argument(
        "-s",
        "--system",
        default=None,
        help="System prompt. Use @path to load from file.",
    )
    p.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Ask the model for JSON output (no schema).",
    )
    p.add_argument(
        "--schema",
        default=None,
        help="JSON Schema for structured output. Inline JSON, @path, or a "
        "path ending in .json. Implies --json.",
    )
    p.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=None,
    )
    p.add_argument(
        "-r",
        "--reasoning",
        choices=reasoning_mod.LEVELS,
        default=None,
        help=(
            "Reasoning effort: low/medium/high/xhigh. Translated to each "
            "provider's native control. Default: $DEFAULT_EFFORT or provider "
            "default. An explicit value fails on models with no reasoning "
            "control; a $DEFAULT_EFFORT default is silently dropped on them."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
    )
    p.add_argument(
        "-f",
        "--file",
        action="append",
        default=[],
        dest="files",
        metavar="PATH",
        help=(
            "Attach a file (image or PDF). Repeatable. Use `-` for stdin "
            "(mutually exclusive with text-on-stdin in that invocation). "
            "Process substitution `<(cmd)` works as a path."
        ),
    )
    p.add_argument(
        "--mime",
        default=None,
        help=(
            "Override MIME type for the next `-f` (applies to all -f in this "
            "invocation). Sniffed from bytes / extension by default."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log provider/model/token usage to stderr.",
    )
    p.add_argument(
        "--usage",
        action="store_true",
        help=(
            "Emit one machine-readable JSON usage record to stderr, prefixed "
            "'gllm-usage ' — provider, model, reasoning, input/output/cache/"
            "reasoning tokens, derived cost_usd (from the llm-prices.com feed, "
            "24h-cached), plus the provider's verbatim usage in usage_raw. "
            "stdout stays the model text only."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    _load_user_env_file(CONFIG_ENV_PATH)

    args = _parser().parse_args(argv)

    # `--models` is a discovery mode: probe live catalogs and exit before any
    # prompt/attachment handling (it needs neither).
    if args.models is not None:
        return _run_models(args.models)

    # Resolve -m manually so we can tell whether the user typed it.
    model_was_defaulted = args.model is None
    if model_was_defaulted:
        args.model = os.environ.get("DEFAULT_MODEL", DEFAULT_MODEL)

    # Track provenance: an explicit -r/--reasoning is a hard contract, but a
    # value inherited from $DEFAULT_EFFORT is just an ambient default that may be
    # silently dropped on models that can't reason (see the capability gate below).
    reasoning_was_defaulted = False
    if args.reasoning is None:
        env_reasoning = os.environ.get("DEFAULT_EFFORT")
        if env_reasoning:
            if env_reasoning not in reasoning_mod.LEVELS:
                expected = ", ".join(reasoning_mod.LEVELS)
                print(
                    f"gllm: DEFAULT_EFFORT must be one of {expected}; "
                    f"got {env_reasoning!r}.",
                    file=sys.stderr,
                )
                return 2
            args.reasoning = env_reasoning
            reasoning_was_defaulted = True

    # WORK mode redirects direct Anthropic/OpenAI models to their Azure Foundry
    # `-dev` deployment. Everything downstream sees the effective name.
    args.model = effective_model(args.model, work_env())
    provider_name = provider_for(args.model)

    # Reasoning capability gate. An explicit --reasoning a model can't honour is
    # a hard error (fail loud). But an ambient $DEFAULT_EFFORT must not block
    # non-reasoning models like gpt-4.1 — drop it silently and carry on. Done
    # before the status print so the printed model:reasoning line is truthful.
    if args.reasoning and not supports_reasoning(provider_name, args.model):
        if reasoning_was_defaulted:
            args.reasoning = None
        else:
            print(
                f"gllm: {provider_name} model {args.model!r} has no reasoning "
                f"control; drop --reasoning or use a reasoning-capable model "
                f"(gpt-5/o-series, claude-*, gemini-*, grok-*).",
                file=sys.stderr,
            )
            return 2

    if model_was_defaulted:
        print(
            f"{args.model}:{args.reasoning}" if args.reasoning else args.model,
            file=sys.stderr,
        )

    files: list[str] = args.files or []
    stdin_is_file = "-" in files
    if files.count("-") > 1:
        print("gllm: -f - can only be specified once.", file=sys.stderr)
        return 2

    # Load attachments first so a failure short-circuits before any LLM call.
    # If `-f -` is in play, stdin is bytes — skip the text-stdin read entirely.
    try:
        attachments = tuple(_load_attachment(s, args.mime) for s in files)
    except (OSError, RuntimeError) as e:
        print(f"gllm: -f: {e}", file=sys.stderr)
        return 2

    stdin_text = None if stdin_is_file else _read_stdin_if_piped()
    positional = args.prompt

    if positional and stdin_text:
        prompt = f"{positional}\n\n{stdin_text}"
    elif positional:
        prompt = positional
    elif stdin_text:
        prompt = stdin_text
    else:
        print(
            "gllm: no prompt. Pass one as an argument or pipe text via stdin.",
            file=sys.stderr,
        )
        return 2

    system = _read_text_arg(args.system) if args.system else None

    schema = None
    if args.schema:
        try:
            schema = _load_schema(args.schema)
        except (OSError, json.JSONDecodeError) as e:
            print(f"gllm: --schema: {e}", file=sys.stderr)
            return 2

    request = Request(
        prompt=prompt,
        system=system,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        schema=schema,
        json_mode=args.json or schema is not None,
        attachments=attachments,
        reasoning=args.reasoning,
    )

    # Strict-or-fail: --schema promises enforced structured output. Refuse it on
    # providers that can only fake it via prompt instructions (no guarantee) —
    # better a loud error than a false sense of enforcement. --json (best-effort)
    # is still fine there.
    if schema is not None and not supports_strict_schema(provider_name, args.model):
        print(
            f"gllm: {provider_name} model {args.model!r} has no native JSON-"
            f"schema enforcement; --schema would only be faked via prompt "
            f"instructions (no guarantee). Use --json for best-effort JSON, or "
            f"a model with native support (claude-*, gpt-*, gemini-*, grok-*).",
            file=sys.stderr,
        )
        return 2

    # Native-or-fail: refuse to dispatch if any attachment is unsupported.
    for a in attachments:
        kind = "image" if a.mime_type.startswith("image/") else (
            "pdf" if a.mime_type == "application/pdf" else a.mime_type
        )
        ok = (
            supports_image(provider_name) if kind == "image"
            else supports_pdf(provider_name, args.model) if kind == "pdf"
            else False
        )
        if not ok:
            print(
                f"gllm: {provider_name} does not accept {kind} inputs "
                f"(model={args.model}). Try a vision/document-capable model.",
                file=sys.stderr,
            )
            return 2

    if args.verbose:
        print(
            f"gllm: provider={provider_name} model={args.model} "
            f"json={request.json_mode} schema={'yes' if schema else 'no'}",
            file=sys.stderr,
        )

    try:
        provider = _build_provider(provider_name)
        response = provider.generate(request)
    except Exception as e:
        print(f"gllm: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    sys.stdout.write(response.text)
    if not response.text.endswith("\n"):
        sys.stdout.write("\n")

    if args.verbose:
        print(
            f"gllm: tokens in={response.input_tokens} out={response.output_tokens}",
            file=sys.stderr,
        )

    if args.usage:
        # Machine-readable sibling of --verbose. One JSON object on its own line,
        # prefixed so a caller can grep it out of mixed stderr. usage_raw carries
        # the provider's own numbers for exact per-model cost accounting; cost_usd
        # is derived from the llm-prices.com feed (priced_as names the matched
        # entry, null when the feed has no price for this model — e.g. GLM).
        usage = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cache_read_tokens": response.cache_read_tokens,
            "cache_write_tokens": response.cache_write_tokens,
            "reasoning_tokens": response.reasoning_tokens,
        }
        # Match on what answered, falling back to the requested name.
        candidates = list(dict.fromkeys([response.model, request.model]))
        record = {
            "provider": response.provider,
            "model": response.model,
            "reasoning": request.reasoning,
            **usage,
            **pricing.price_report(response.provider, candidates, usage),
            "max_tokens": request.max_tokens,
            "schema": schema is not None,
            "json": request.json_mode,
            "usage_raw": response.usage_raw,
        }
        print(
            "gllm-usage " + json.dumps(record, separators=(",", ":")),
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
