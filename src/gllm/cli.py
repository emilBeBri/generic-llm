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
import time
from datetime import UTC, datetime
from pathlib import Path

from . import calllog
from . import pricing
from . import reasoning as reasoning_mod
from .adapters._capabilities import (
    native_efforts,
    openai_file_mime_for_path,
    supports_attachment,
    supports_reasoning,
    supports_strict_schema,
    thinking_dialect,
)
from .config import work_env
from .domain import Attachment, Request
from .models import MODELS, context_window_for, max_output_for, wire_id_for
from .ports import LLMProvider
from .providers import DISCOVERABLE_PROVIDERS, PROVIDERS
from .routing import WORK_PROVIDER_REDIRECTS, effective_model, provider_for

DEFAULT_MODEL = "deepseek-v4-flash"
# Config and keys load from this repo's own .env (repo root, beside
# pyproject.toml), resolved relative to this file so it is found regardless of
# cwd. cli.py lives at <root>/src/gllm/cli.py, so parents[2] is the repo root.
# See .llm-memory/IDEAS-key-loading-secret-managers.md for the longer-term plan.
CONFIG_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

# Fallback output budget for models whose real ceiling this repo has not sourced
# (`ModelSpec.max_output is None`). Low on purpose: too high is a hard 400 on
# several providers, and an unknown model is exactly where a guess is worst.
DEFAULT_MAX_OUTPUT = 4096

# Chars per token, for sizing the default output budget against the context
# window. Deliberately pessimistic: the usual rule of thumb is 4, but 222,499
# chars of Danish filler measured 65,017 input tokens on GLM — 3.42 chars/token,
# so /4 would have UNDER-counted by 17%. Under-counting is the direction that
# 400s, so this rounds the other way. Only ever used to make the budget SMALLER,
# which is why a sloppy estimate is tolerable here and would not be for an
# input-size *refusal*.
_CHARS_PER_TOKEN = 3.0


def _resolve_max_tokens(
    explicit: int | None,
    provider: str,
    model: str,
    wire_effort: str,
    *,
    quiet: bool,
    input_chars: int = 0,
    has_attachments: bool = False,
) -> int:
    """The output budget actually sent, resolved once for every adapter.

    Three cases, and the distinction between the first two is the whole point of
    `--max-tokens` defaulting to None:

    - **No flag.** gllm picks: the model's documented ceiling when the registry
      knows it, else `DEFAULT_MAX_OUTPUT`, raised to the reasoning floor when
      `-r` is on. Silent — the user expressed no preference to override.
    - **Explicit, and adequate.** Sent verbatim.
    - **Explicit, below the reasoning floor.** Honoured anyway, with a warning,
      because a stated number is a decision and gllm does not quietly overrule
      one. The exception is a floor the API *enforces* (Anthropic's
      `budget_tokens < max_tokens`), which would be a guaranteed 400 — that
      raises ValueError so the caller can refuse before spending a request.
    """
    dialect = thinking_dialect(provider, model) if wire_effort else None
    floor = (
        reasoning_mod.min_output_tokens(model, wire_effort, dialect)
        if wire_effort
        else 0
    )

    if explicit is None:
        want = max(max_output_for(model) or DEFAULT_MAX_OUTPUT, floor)
        return _clamp_to_context(want, floor, model, input_chars, has_attachments, quiet)

    if explicit >= floor:
        return explicit

    hard = reasoning_mod.hard_min_output_tokens(model, wire_effort, dialect)
    if hard is not None and explicit < hard:
        raise ValueError(
            f"--max-tokens {explicit} is below what {model} requires with "
            f"-r: its thinking budget must be strictly less than max_tokens, "
            f"so the API needs at least {hard}. Raise --max-tokens or drop -r."
        )
    if not quiet:
        print(
            f"gllm: --max-tokens {explicit} is below the {floor} that reasoning "
            f"wants on {model}; sending {explicit} as asked. Thinking is spent "
            f"from this budget, so the answer may be truncated.",
            file=sys.stderr,
        )
    return explicit


def _clamp_to_context(
    want: int,
    floor: int,
    model: str,
    input_chars: int,
    has_attachments: bool,
    quiet: bool,
) -> int:
    """Shrink gllm's OWN default so input + output still fits the context window.

    The output budget is not a separate allowance — it shares the context window
    with the prompt. Verified to the token against GLM (context 131,072): a
    65,017-token input plus `max_tokens=66,000` succeeds at 131,017, and plus
    66,100 fails at 131,117. Both the input alone and that `max_tokens` alone are
    legal, so only the sum can explain it.

    Which makes this necessary rather than tidy: defaulting the budget to a
    model's full output ceiling (see `_resolve_max_tokens`) eats input headroom.
    On the six 200k-context Claude rows a 64,000 default leaves ~136,000 tokens
    for input, so a large document that used to fit now 400s — and the provider
    blames the prompt for it. GLM answers `1261 "Prompt exceeds max length"` when
    the prompt was 65,017 of 131,072, which is not a diagnosis anyone can act on.

    Only ever lowers, and only a value gllm chose: an explicit `--max-tokens`
    never reaches here.
    """
    if has_attachments:
        # Image and PDF-page cost is a function of pixel dimensions and page
        # count, not of character length, so there is no honest estimate to make
        # here. Leave the budget alone and let the API be the backstop rather
        # than clamp on a number that means nothing.
        return want

    headroom = context_window_for(model) - int(input_chars / _CHARS_PER_TOKEN)
    if headroom >= want:
        return want
    if headroom < floor and not quiet:
        # Below this the reasoning trace itself may not fit, so the answer is
        # likely to come back truncated (which `Response.truncated` will catch).
        print(
            f"gllm: input leaves only ~{max(headroom, 0)} of "
            f"{model}'s {context_window_for(model)}-token context for output, "
            f"under the {floor} reasoning wants. The answer may be truncated; "
            f"shorten the input or pick a longer-context model.",
            file=sys.stderr,
        )
    return max(headroom, 1)


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
        openai_file_mime = openai_file_mime_for_path(path_hint)
        if openai_file_mime:
            return openai_file_mime
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
    """Construct the adapter for a provider tag.

    Dispatch is on `ProviderSpec.adapter_kind`, so several hosts can share one
    adapter (groq and regolo are both `openai_compat`). Imports stay per-branch
    and lazy: a missing SDK must only break its own provider, never the CLI.
    """
    spec = PROVIDERS.get(name)
    if spec is None:
        raise ValueError(f"unknown provider: {name}")

    kind = spec.adapter_kind
    if kind == "anthropic":
        from .adapters.anthropic import AnthropicProvider

        return AnthropicProvider()
    if kind == "openai":
        from .adapters.openai import OpenAIProvider

        return OpenAIProvider()
    if kind == "gemini":
        from .adapters.gemini import GeminiProvider

        return GeminiProvider()
    if kind == "deepseek":
        from .adapters.deepseek import DeepSeekProvider

        return DeepSeekProvider()
    if kind == "grok":
        from .adapters.grok import GrokProvider

        return GrokProvider()
    if kind == "zai":
        from .adapters.zai import ZaiProvider

        return ZaiProvider()
    if kind == "kimi":
        from .adapters.kimi import KimiProvider

        return KimiProvider()
    if kind == "openai_compat":
        from .adapters.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(spec)
    if kind == "azure_openai":
        from .adapters.azure_openai import AzureOpenAIProvider

        return AzureOpenAIProvider()
    if kind == "azure_anthropic":
        from .adapters.azure_anthropic import AzureAnthropicProvider

        return AzureAnthropicProvider()
    raise ValueError(f"unknown adapter kind: {kind!r} (provider {name!r})")


def _provider_is_configured(name: str) -> bool:
    spec = PROVIDERS[name]
    has_key = any(os.environ.get(env_name) for env_name in spec.api_key_env)
    has_required = all(os.environ.get(env_name) for env_name in spec.required_env)
    return has_key and has_required


def _configured_model_providers(work: bool) -> tuple[str, ...]:
    return tuple(
        name
        for name in DISCOVERABLE_PROVIDERS
        if _provider_is_configured(name)
        and not (work and name in WORK_PROVIDER_REDIRECTS)
    )


def _registered_provider_models(name: str) -> list[str]:
    return sorted(key for key, spec in MODELS.items() if spec.provider == name)


def _run_models(
    which: str,
    work: bool = False,
    include_capabilities: bool = False,
) -> int:
    """`gllm --models`: print live `provider<TAB>model-id` rows, one per line.

    Probes each provider's API for the models it ACTUALLY serves right now —
    the single source of truth — instead of a hand-maintained catalog that
    drifts out of sync (the failure that made an agent declare a live model
    "retired").     With no provider argument, only providers configured in the current
    environment are queried. WORK mode replaces direct Anthropic/OpenAI with
    their Azure Foundry hosts. Foundry has no deployment-listing inference API,
    so its explicit deployment rows come from the model registry; every other
    provider is probed live.
    """
    if which and which != "*":
        if which not in DISCOVERABLE_PROVIDERS:
            print(
                f"gllm: --models: unknown provider {which!r}; choose from "
                f"{', '.join(DISCOVERABLE_PROVIDERS)}.",
                file=sys.stderr,
            )
            return 2
        target = WORK_PROVIDER_REDIRECTS.get(which, which) if work else which
        targets: tuple[str, ...] = (target,)
    else:
        targets = _configured_model_providers(work)

    if not targets:
        print(
            "gllm: --models: no configured model providers are available.",
            file=sys.stderr,
        )
        return 1

    any_ok = False
    for name in targets:
        spec = PROVIDERS[name]
        if not _provider_is_configured(name):
            print(
                f"gllm: {name}: skipped (provider is not fully configured)",
                file=sys.stderr,
            )
            continue
        try:
            models = (
                _registered_provider_models(name)
                if spec.registry_models
                else _build_provider(name).list_models()
            )
        except Exception as e:
            print(f"gllm: {name}: skipped ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        # Host providers namespace their registry keys, so print the key rather
        # than the bare wire id — a printed row should be pasteable into `-m`.
        prefix = spec.key_namespace or ""
        for mid in models:
            key = f"{prefix}{mid}"
            fields = [name, key]
            if include_capabilities:
                fields.append("reasoning" if supports_reasoning(name, key) else "default")
            print("\t".join(fields))
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
            "List text-generation models available through configured providers "
            "(one `provider<TAB>id` per line; pipe to rg/fzf). Optionally "
            "restrict to one: --models gemini. Ignores the prompt."
        ),
    )
    p.add_argument(
        "--model-capabilities",
        action="store_true",
        help=(
            "With --models, append a capability field: `reasoning` when the "
            "model accepts -r, otherwise `default`."
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
        # No argparse `choices`: the valid vocabulary depends on --native-effort,
        # which argparse cannot consult here. Validated after parsing instead, so
        # the error names the right vocabulary for the mode actually in use.
        default=None,
        metavar="LEVEL",
        help=(
            "Reasoning effort: low/medium/high/xhigh. Translated to each "
            "provider's native control. Default: $DEFAULT_EFFORT or provider "
            "default. An explicit value fails on models with no reasoning "
            "control; a $DEFAULT_EFFORT default is silently dropped on them. "
            "With --native-effort, takes the model's OWN value instead."
        ),
    )
    p.add_argument(
        "--native-effort",
        action="store_true",
        help=(
            "Pass -r through as the model's OWN effort value, untranslated. For "
            "when you need to know exactly what reached the provider: "
            "benchmarks, effort sweeps, reproducing a run. Off by default — the "
            "translated ladder is what keeps a script portable across models. "
            "Requires an explicit -r, and rejects a value the model does not "
            "have (see --models for each model's vocabulary)."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Ceiling on OUTPUT tokens, thinking included. Default: the model's "
            "documented maximum where known, otherwise "
            f"{DEFAULT_MAX_OUTPUT} (raised to fit reasoning). An explicit value "
            "is sent as given, even if reasoning may then truncate the answer."
        ),
    )
    p.add_argument(
        "-f",
        "--file",
        action="append",
        default=[],
        dest="files",
        metavar="PATH",
        help=(
            "Attach a provider-supported file. Repeatable. Use `-` for stdin "
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
        "-q",
        "--quiet-effort",
        action="store_true",
        help=(
            "Suppress the stderr notice printed when --reasoning is remapped "
            "onto a model's own effort vocabulary (e.g. xhigh -> 'max')."
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
            "reasoning tokens, derived cost_usd (from the llm-price-tracker "
            "book, offline) with price_window naming the peak/off-peak rate "
            "applied, plus the provider's verbatim usage in usage_raw. "
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
        return _run_models(args.models, work_env(), args.model_capabilities)
    if args.model_capabilities:
        print("gllm: --model-capabilities requires --models.", file=sys.stderr)
        return 2

    # Resolve -m manually so we can tell whether the user typed it.
    model_was_defaulted = args.model is None
    if model_was_defaulted:
        args.model = os.environ.get("DEFAULT_MODEL", DEFAULT_MODEL)

    # Track provenance: an explicit -r/--reasoning is a hard contract, but a
    # value inherited from $DEFAULT_EFFORT is just an ambient default that may be
    # silently dropped on models that can't reason (see the capability gate below).
    reasoning_was_defaulted = False
    if args.reasoning is None:
        # --native-effort is a request for exactness, so it will not inherit an
        # ambient default: $DEFAULT_EFFORT is a portable gllm rung, and silently
        # feeding it to a provider as a native value is precisely the confusion
        # the flag exists to remove.
        if args.native_effort:
            print(
                "gllm: --native-effort needs an explicit -r/--reasoning "
                "(it will not inherit $DEFAULT_EFFORT).",
                file=sys.stderr,
            )
            return 2
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
    elif not args.native_effort and args.reasoning not in reasoning_mod.LEVELS:
        # Validated here rather than by argparse `choices`, because the valid
        # vocabulary depends on --native-effort. Name the flag in the message:
        # someone typing `-r max` wants the model's top rung and has just found
        # the exact case --native-effort was added for.
        expected = ", ".join(reasoning_mod.LEVELS)
        print(
            f"gllm: -r/--reasoning must be one of {expected}; got "
            f"{args.reasoning!r}. To pass a provider's own value instead "
            f"(e.g. 'max'), add --native-effort.",
            file=sys.stderr,
        )
        return 2

    # WORK mode redirects direct Anthropic/OpenAI models to their Azure Foundry
    # `-dev` deployment. Everything downstream sees the effective name.
    args.model = effective_model(args.model, work_env())
    provider_name = provider_for(args.model)

    # Reasoning capability gate — ONE question: does this model have an effort
    # knob at all? gllm's four rungs always resolve onto a non-empty vocabulary
    # (reasoning.resolve_effort), so "a level this model can't take" no longer
    # exists. An explicit --reasoning on a knobless model is a hard error (fail
    # loud); an ambient $DEFAULT_EFFORT is dropped instead, so a global
    # DEFAULT_EFFORT=low doesn't break every pipe to gpt-4.1 or grok-build-0.1.
    # Done before the status print so the printed model:reasoning line is true.
    wire_effort = ""
    if args.reasoning and not supports_reasoning(provider_name, args.model):
        if reasoning_was_defaulted:
            args.reasoning = None
        else:
            print(
                f"gllm: {provider_name} model {args.model!r} has no reasoning "
                f"control; drop --reasoning or use a reasoning-capable model "
                f"(gpt-5/o-series, claude-*, gemini-*, grok-4.3/4.5, deepseek-*).",
                file=sys.stderr,
            )
            return 2

    if args.reasoning:
        native = native_efforts(provider_name, args.model)
        if args.native_effort:
            # No translation: the value goes to the provider exactly as typed.
            # It must therefore be one this model actually has — refusing here
            # beats letting the provider reject it after the payload is sent.
            if args.reasoning not in native:
                print(
                    f"gllm: --native-effort: {args.model} has no effort "
                    f"{args.reasoning!r}; it offers: {', '.join(native)}.",
                    file=sys.stderr,
                )
                return 2
            wire_effort = args.reasoning
        else:
            # Normalise the rung onto this model's OWN vocabulary. `xhigh` means
            # "the most this model has", so it can land on a differently-named
            # value (DeepSeek and GLM call their top rung `max`). Announce it
            # when that happens — a level silently meaning something else is the
            # sort of quiet degradation gllm refuses everywhere else. Silent on
            # a pass-through.
            wire_effort = reasoning_mod.resolve_effort(args.reasoning, native)
            if wire_effort != args.reasoning and not args.quiet_effort:
                print(
                    f"gllm: -r {args.reasoning} -> {wire_effort!r} "
                    f"({args.model} offers: {', '.join(native)})",
                    file=sys.stderr,
                )

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

    try:
        max_tokens = _resolve_max_tokens(
            args.max_tokens,
            provider_name,
            args.model,
            wire_effort,
            quiet=args.quiet_effort,
            # Resolved here rather than earlier because the clamp needs the
            # prompt: the output budget shares the context window with it.
            input_chars=len(prompt) + len(system or ""),
            has_attachments=bool(attachments),
        )
    except ValueError as e:
        print(f"gllm: {e}", file=sys.stderr)
        return 2

    request = Request(
        prompt=prompt,
        system=system,
        model=args.model,
        # Namespaced host keys ('groq:openai/gpt-oss-120b') are gllm's identity
        # for the model; the host itself only knows the bare id.
        wire_model=wire_id_for(args.model),
        max_tokens=max_tokens,
        temperature=args.temperature,
        schema=schema,
        json_mode=args.json or schema is not None,
        attachments=attachments,
        reasoning=args.reasoning,
        wire_effort=wire_effort,
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
        if not supports_attachment(provider_name, args.model, a):
            kind = (
                "image"
                if a.mime_type.startswith("image/")
                else "PDF"
                if a.mime_type == "application/pdf"
                else f"file type {a.mime_type}"
            )
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

    # Stamped at DISPATCH, not after the response: vendors with time-of-day
    # rates bill by when the request lands, and a call that starts at 03:58
    # UTC and returns at 04:02 must not be repriced by having been slow.
    # Outside the try so the failure path can log a timed record too — a call
    # that errored after 30s is exactly the kind of thing the log exists for.
    sent_at = datetime.now(UTC)
    started = time.perf_counter()
    try:
        provider = _build_provider(provider_name)
        response = provider.generate(request)
    except Exception as e:
        calllog.append(
            {
                "ts": sent_at.isoformat(),
                "elapsed_s": round(time.perf_counter() - started, 3),
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "provider": provider_name,
                "model": args.model,
                "reasoning": args.reasoning,
                # No response to record, but what was SENT is the interesting
                # half of a failure.
                **calllog.text_fields(prompt=prompt, system=request.system),
            }
        )
        print(f"gllm: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    elapsed_s = round(time.perf_counter() - started, 3)

    sys.stdout.write(response.text)
    if not response.text.endswith("\n"):
        sys.stdout.write("\n")

    # A truncated answer looks exactly like a complete one on stdout — a capped
    # Gemini call printed "1" for 23*47 and read as a confident wrong answer.
    # Deliberately NOT silenced by -q: that flag is --quiet-effort, scoped to the
    # effort-remap notice, and this is a correctness signal rather than chatter.
    if response.truncated:
        print(
            f"gllm: OUTPUT TRUNCATED — {response.provider} stopped at "
            f"{request.max_tokens} tokens (stop_reason={response.stop_reason!r}). "
            f"The answer above is cut off; raise --max-tokens."
            + (
                " Reasoning is spent from this same budget."
                if request.reasoning
                else ""
            ),
            file=sys.stderr,
        )

    if args.verbose:
        print(
            f"gllm: tokens in={response.input_tokens} out={response.output_tokens}",
            file=sys.stderr,
        )

    # Built once for two consumers. `calllog.enabled()` is checked first so a
    # disabled log costs nothing: building this record prices the call, which
    # loads the price book (~150ms of pydantic) that plain runs skip.
    if args.usage or calllog.enabled():
        # Machine-readable sibling of --verbose. One JSON object on its own line,
        # prefixed so a caller can grep it out of mixed stderr. usage_raw carries
        # the provider's own numbers for exact per-model cost accounting; cost_usd
        # is derived from the llm-price-tracker book plus the local overrides
        # (priced_as names the matched entry, null when neither prices the model).
        # price_window says which side of a vendor's peak/off-peak split was
        # billed, so a 2x cost_usd swing between two identical calls is legible
        # as DeepSeek's published policy rather than as a gllm bug.
        usage = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cache_read_tokens": response.cache_read_tokens,
            "cache_write_tokens": response.cache_write_tokens,
            "reasoning_tokens": response.reasoning_tokens,
        }
        # Match on what answered, falling back to the requested name, then the
        # bare wire id — the book is vendor-id-keyed, and Response.model is the
        # registry key on some adapters and the vendor's returned id on others.
        candidates = [
            c
            for c in dict.fromkeys(
                [response.model, request.model, request.wire_model]
            )
            if c
        ]
        record = {
            "provider": response.provider,
            "model": response.model,
            "reasoning": request.reasoning,
            **usage,
            **pricing.price_report(response.provider, candidates, usage, sent_at),
            "max_tokens": request.max_tokens,
            # The provider's own word, verbatim, plus gllm's reading of it — a
            # machine consumer should not have to know that Gemini says
            # MAX_TOKENS where OpenAI chat says length.
            "stop_reason": response.stop_reason,
            "truncated": response.truncated,
            "schema": schema is not None,
            "json": request.json_mode,
            "usage_raw": response.usage_raw,
        }
        if args.usage:
            print(
                "gllm-usage " + json.dumps(record, separators=(",", ":")),
                file=sys.stderr,
            )
        # Lengths always; the text itself only under the separate
        # GLLM_CALL_LOG_TEXT opt-in — see calllog's module docstring.
        calllog.append(
            {
                "ts": sent_at.isoformat(),
                "elapsed_s": elapsed_s,
                "ok": True,
                **record,
                "prompt_chars": len(prompt or ""),
                "response_chars": len(response.text or ""),
                "attachments": len(attachments),
                **calllog.text_fields(
                    prompt=prompt, response=response.text, system=request.system
                ),
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
