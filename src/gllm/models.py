"""Model registry: the explicit model axis.

The dict key is the app-facing model identity — what you type after `-m`, what
`--usage` reports, and what `data/prices.json` is keyed by. `wire_id` is the
literal string sent over the wire; the two differ only for namespaced host rows
(`groq:openai/gpt-oss-120b` -> `openai/gpt-oss-120b`).

`provider` is a key into `providers.PROVIDERS` and is the ONLY routing signal.
Never infer a provider from the model name: `groq:deepseek-*` contains
"deepseek" but is answered by Groq, and `regolo:glm5.2-beta` contains "glm" but
is answered by Regolo. `family` ties entries serving the same underlying open
model across hosts.

`caps` is what every capability gate reads. A model NOT in this registry still
works — `caps_for` falls back to the historical substring heuristics — because
model names rot faster than any hand-maintained catalog, and `gllm --models`
(a live API probe) remains the source of truth about what a vendor serves right
now. This registry is the source of truth about how to *drive* what it serves.

Key invariants (enforced by tests/test_registry.py):
  - keys are lowercase;
  - a key ends in `-dev` only if it is an Azure Foundry deployment row;
  - host rows are namespaced `<provider>:<slug>` as a naming convention only —
    routing always uses `ModelSpec.provider`;
  - `azure_alias` / `alt_model` targets must themselves be registry keys.
"""

from __future__ import annotations

from typing import NamedTuple


class ModelCaps(NamedTuple):
    """What a model can be asked to do. Read by `adapters._capabilities`.

    `reasoning_efforts` doubles as the reasoning-support flag: an empty tuple
    means "no reasoning control at all", and a non-empty one is the exact set of
    `--reasoning` levels the model accepts. `thinking_dialect` selects the wire
    translation in `gllm.reasoning`.
    """

    # 'responses' | 'chat' — which OpenAI API surface. None for non-OpenAI-family
    # providers (anthropic, gemini, zai, and the openai_compat hosts, which all
    # speak exactly one surface).
    api_surface: str | None = None
    reasoning_efforts: tuple[str, ...] = ()
    # 'anthropic_adaptive' | 'anthropic_budget' | 'gemini_budget' |
    # 'openai_effort' | 'zai_effort' | 'zai_thinking' | 'compat_effort' |
    # 'compat_thinking_flag'
    thinking_dialect: str | None = None
    supports_vision: bool = False
    supports_pdf: bool = False
    # Does the API NATIVELY ENFORCE a JSON Schema? Not "can it be asked nicely".
    supports_strict_schema: bool = False


class ModelSpec(NamedTuple):
    wire_id: str
    provider: str
    context_window: int
    caps: ModelCaps
    # Registry key of the Azure Foundry deployment to use under WORK=1.
    azure_alias: str | None = None
    alt_model: str | None = None
    # Groups rows serving the same underlying open model across hosts.
    family: str | None = None


# --------------------------------------------------------------------------- #
# Capability presets. Six fields x ~130 rows is unreadable spelled out per row,
# so rows reference a named preset. A model that genuinely differs gets its own.
# --------------------------------------------------------------------------- #

# --- Anthropic. Every active Claude does vision, PDF and native json_schema.
# The split that matters is the thinking interface: 4.6+ REQUIRE
# thinking.type=adaptive and reject enabled+budget_tokens; 4.5 and older are the
# reverse. Effort vocabulary per platform.claude.com/docs/en/build_with_claude/effort.
_CLAUDE_ADAPTIVE_XHIGH = ModelCaps(
    reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
    thinking_dialect="anthropic_adaptive",
    supports_vision=True,
    supports_pdf=True,
    supports_strict_schema=True,
)
# 4.6 / Sonnet 4.6: adaptive and `max`, but no `xhigh` rung.
_CLAUDE_ADAPTIVE = _CLAUDE_ADAPTIVE_XHIGH._replace(
    reasoning_efforts=("low", "medium", "high", "max")
)
_CLAUDE_BUDGET = _CLAUDE_ADAPTIVE_XHIGH._replace(
    reasoning_efforts=("low", "medium", "high", "xhigh"),
    thinking_dialect="anthropic_budget",
)

# --- OpenAI. PDF input is `input_file`, which exists only on the Responses API.
_GPT5_MAX = ModelCaps(
    api_surface="responses",
    reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
    thinking_dialect="openai_effort",
    supports_vision=True,
    supports_pdf=True,
    supports_strict_schema=True,
)
_GPT5 = _GPT5_MAX._replace(reasoning_efforts=("low", "medium", "high", "xhigh"))
_O_SERIES = _GPT5_MAX._replace(reasoning_efforts=("low", "medium", "high"))
# gpt-*-chat-latest are the non-reasoning chat tunings of the gpt-5 line: still
# Responses, but no effort knob.
_GPT5_CHAT = _GPT5_MAX._replace(reasoning_efforts=(), thinking_dialect=None)
_GPT4_CHAT = ModelCaps(
    api_surface="chat", supports_vision=True, supports_strict_schema=True
)

# --- Gemini. Thinking is a budget int; -1 = dynamic, which we map to xhigh.
_GEMINI = ModelCaps(
    reasoning_efforts=("low", "medium", "high", "xhigh"),
    thinking_dialect="gemini_budget",
    supports_vision=True,
    supports_pdf=True,
    supports_strict_schema=True,
)

# --- xAI. Responses API byte-for-byte, but reasoning_effort tops out at `high`
# (docs.x.ai: none|low|medium|high) and there is no input_file for PDFs.
_GROK = ModelCaps(
    api_surface="responses",
    reasoning_efforts=("low", "medium", "high"),
    thinking_dialect="openai_effort",
    supports_vision=True,
    supports_strict_schema=True,
)
_GROK_NO_REASONING = _GROK._replace(reasoning_efforts=(), thinking_dialect=None)

# --- DeepSeek. Reasons by default but exposes NO effort knob, and has only
# response_format=json_object (no schema enforcement). Nothing to gate on.
_DEEPSEEK = ModelCaps()

# --- Z.AI / GLM. Capabilities are split across model FAMILIES rather than
# gated per call: vision is a separate model line, and `reasoning_effort` is
# honoured only from glm-5.2 on (below that, thinking is a binary on/off).
_GLM_EFFORT = ModelCaps(
    reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
    thinking_dialect="zai_effort",
)
_GLM_THINK = ModelCaps(
    reasoning_efforts=("low", "medium", "high", "xhigh"),
    thinking_dialect="zai_thinking",
)
_GLM_NO_THINK = ModelCaps()
_GLM_VISION_EFFORT = _GLM_EFFORT._replace(supports_vision=True)
_GLM_VISION_THINK = _GLM_THINK._replace(supports_vision=True)
_GLM_VISION_NO_THINK = ModelCaps(supports_vision=True)

# --- OpenAI-compatible hosts (Groq, Regolo). Chat Completions only, no schema
# enforcement, no document input. Groq takes a bare `reasoning_effort`; Regolo
# needs a top-level `thinking` flag alongside it.
_GROQ_EFFORT = ModelCaps(
    reasoning_efforts=("low", "medium", "high"), thinking_dialect="compat_effort"
)
_GROQ_PLAIN = ModelCaps()
_REGOLO_THINK = ModelCaps(
    reasoning_efforts=("low", "medium", "high"),
    thinking_dialect="compat_thinking_flag",
)
_REGOLO_PLAIN = ModelCaps()


MODELS: dict[str, ModelSpec] = {
    # ----------------------------------------------------------------- #
    # Anthropic
    # ----------------------------------------------------------------- #
    "claude-fable-5": ModelSpec(
        "claude-fable-5", "anthropic", 1_000_000, _CLAUDE_ADAPTIVE_XHIGH,
        azure_alias="claude-fable-5-dev", alt_model="claude-opus-5",
    ),
    "claude-opus-5": ModelSpec(
        "claude-opus-5", "anthropic", 1_000_000, _CLAUDE_ADAPTIVE_XHIGH,
        alt_model="claude-sonnet-5",
    ),
    "claude-sonnet-5": ModelSpec(
        "claude-sonnet-5", "anthropic", 1_000_000, _CLAUDE_ADAPTIVE_XHIGH,
        alt_model="claude-opus-5",
    ),
    "claude-opus-4-8": ModelSpec(
        "claude-opus-4-8", "anthropic", 1_000_000, _CLAUDE_ADAPTIVE_XHIGH,
        azure_alias="claude-opus-4-8-dev", alt_model="claude-opus-5",
    ),
    "claude-opus-4-7": ModelSpec(
        "claude-opus-4-7", "anthropic", 1_000_000, _CLAUDE_ADAPTIVE_XHIGH,
        azure_alias="claude-opus-4-7-dev", alt_model="claude-opus-5",
    ),
    "claude-opus-4-6": ModelSpec(
        "claude-opus-4-6", "anthropic", 1_000_000, _CLAUDE_ADAPTIVE,
        azure_alias="claude-opus-4-6-dev", alt_model="claude-opus-5",
    ),
    "claude-sonnet-4-6": ModelSpec(
        "claude-sonnet-4-6", "anthropic", 1_000_000, _CLAUDE_ADAPTIVE,
        alt_model="claude-sonnet-5",
    ),
    "claude-opus-4-5": ModelSpec(
        "claude-opus-4-5", "anthropic", 200_000, _CLAUDE_BUDGET,
        azure_alias="claude-opus-4-5-dev", alt_model="claude-opus-5",
    ),
    "claude-sonnet-4-5": ModelSpec(
        "claude-sonnet-4-5", "anthropic", 200_000, _CLAUDE_BUDGET,
        alt_model="claude-sonnet-5",
    ),
    "claude-haiku-4-5": ModelSpec(
        "claude-haiku-4-5", "anthropic", 200_000, _CLAUDE_BUDGET,
        azure_alias="claude-haiku-4-5-dev", alt_model="claude-sonnet-5",
    ),
    # ----------------------------------------------------------------- #
    # OpenAI
    # ----------------------------------------------------------------- #
    # GPT-5.6 is the only line with the `max` effort rung. `gpt-5.6` is the
    # public alias of `gpt-5.6-sol`.
    "gpt-5.6": ModelSpec(
        "gpt-5.6", "openai", 1_050_000, _GPT5_MAX, alt_model="gpt-5.6-terra"
    ),
    "gpt-5.6-sol": ModelSpec(
        "gpt-5.6-sol", "openai", 1_050_000, _GPT5_MAX, alt_model="gpt-5.6-terra"
    ),
    "gpt-5.6-terra": ModelSpec(
        "gpt-5.6-terra", "openai", 1_050_000, _GPT5_MAX, alt_model="gpt-5.6-luna"
    ),
    "gpt-5.6-luna": ModelSpec(
        "gpt-5.6-luna", "openai", 1_050_000, _GPT5_MAX, alt_model="gpt-5.6-terra"
    ),
    # The docs publish only the <272K price tier for the 5.4/5.5 line and omit
    # their maximum context windows. Stay conservative at the largest fully
    # priced range rather than pretend an undocumented limit is exact.
    "gpt-5.5": ModelSpec(
        "gpt-5.5", "openai", 1_000_000, _GPT5,
        azure_alias="gpt-5.5-dev", alt_model="gpt-5.6-terra",
    ),
    "gpt-5.5-pro": ModelSpec(
        "gpt-5.5-pro", "openai", 272_000, _GPT5, alt_model="gpt-5.6-sol"
    ),
    "gpt-5.4": ModelSpec(
        "gpt-5.4", "openai", 272_000, _GPT5, alt_model="gpt-5.6-terra"
    ),
    "gpt-5.4-pro": ModelSpec(
        "gpt-5.4-pro", "openai", 272_000, _GPT5,
        azure_alias="gpt-5.4-pro-dev", alt_model="gpt-5.6-sol",
    ),
    "gpt-5.4-mini": ModelSpec(
        "gpt-5.4-mini", "openai", 272_000, _GPT5, alt_model="gpt-5.6-luna"
    ),
    "gpt-5.4-nano": ModelSpec(
        "gpt-5.4-nano", "openai", 272_000, _GPT5, alt_model="gpt-5.4-mini"
    ),
    "gpt-5.2": ModelSpec("gpt-5.2", "openai", 128_000, _GPT5, alt_model="gpt-5.1"),
    "gpt-5.2-pro": ModelSpec("gpt-5.2-pro", "openai", 128_000, _GPT5),
    "gpt-5.1": ModelSpec(
        "gpt-5.1", "openai", 128_000, _GPT5, azure_alias="gpt-5.1-dev"
    ),
    "gpt-5": ModelSpec("gpt-5", "openai", 128_000, _GPT5),
    "gpt-5-pro": ModelSpec("gpt-5-pro", "openai", 128_000, _GPT5),
    "gpt-5-mini": ModelSpec(
        "gpt-5-mini", "openai", 128_000, _GPT5, azure_alias="gpt-5-mini-dev"
    ),
    "gpt-5-nano": ModelSpec("gpt-5-nano", "openai", 128_000, _GPT5),
    # Codex / chat-latest tunings.
    "gpt-5.3-codex": ModelSpec("gpt-5.3-codex", "openai", 128_000, _GPT5),
    "gpt-5.2-codex": ModelSpec(
        "gpt-5.2-codex", "openai", 128_000, _GPT5, alt_model="gpt-5.1-codex"
    ),
    "gpt-5.1-codex": ModelSpec(
        "gpt-5.1-codex", "openai", 128_000, _GPT5, alt_model="gpt-5-codex"
    ),
    "gpt-5.1-codex-max": ModelSpec("gpt-5.1-codex-max", "openai", 128_000, _GPT5),
    "gpt-5.1-codex-mini": ModelSpec("gpt-5.1-codex-mini", "openai", 128_000, _GPT5),
    "gpt-5-codex": ModelSpec("gpt-5-codex", "openai", 128_000, _GPT5),
    "codex-mini-latest": ModelSpec("codex-mini-latest", "openai", 128_000, _GPT5),
    "gpt-5.2-chat-latest": ModelSpec(
        "gpt-5.2-chat-latest", "openai", 128_000, _GPT5_CHAT,
        alt_model="gpt-5.1-chat-latest",
    ),
    "gpt-5.1-chat-latest": ModelSpec(
        "gpt-5.1-chat-latest", "openai", 128_000, _GPT5_CHAT
    ),
    # GPT-4 line: Chat Completions, so no reasoning and no PDF.
    "gpt-4.1": ModelSpec(
        "gpt-4.1", "openai", 128_000, _GPT4_CHAT, alt_model="gpt-5.4"
    ),
    "gpt-4.1-mini": ModelSpec("gpt-4.1-mini", "openai", 128_000, _GPT4_CHAT),
    "gpt-4.1-nano": ModelSpec(
        "gpt-4.1-nano", "openai", 128_000, _GPT4_CHAT,
        azure_alias="gpt-4.1-nano-dev",
    ),
    "gpt-4o": ModelSpec("gpt-4o", "openai", 128_000, _GPT4_CHAT),
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", "openai", 128_000, _GPT4_CHAT),
    # o-series: effort tops out at `high`.
    "o1": ModelSpec("o1", "openai", 128_000, _O_SERIES, alt_model="o3"),
    "o1-pro": ModelSpec(
        "o1-pro", "openai", 128_000, _O_SERIES, alt_model="gpt-5.5-pro"
    ),
    "o1-mini": ModelSpec(
        "o1-mini", "openai", 128_000, _O_SERIES, alt_model="gpt-5.4-mini"
    ),
    "o3": ModelSpec(
        "o3", "openai", 200_000, _O_SERIES, azure_alias="o3-dev", alt_model="o1"
    ),
    "o3-pro": ModelSpec("o3-pro", "openai", 200_000, _O_SERIES, alt_model="o3"),
    "o3-mini": ModelSpec(
        "o3-mini", "openai", 200_000, _O_SERIES, alt_model="gpt-5.4-mini"
    ),
    "o3-deep-research": ModelSpec("o3-deep-research", "openai", 200_000, _O_SERIES),
    "o4-mini": ModelSpec(
        "o4-mini", "openai", 200_000, _O_SERIES, alt_model="gpt-5.4-mini"
    ),
    "o4-mini-deep-research": ModelSpec(
        "o4-mini-deep-research", "openai", 200_000, _O_SERIES
    ),
    # ----------------------------------------------------------------- #
    # Google Gemini
    # ----------------------------------------------------------------- #
    "gemini-3.6-flash": ModelSpec(
        "gemini-3.6-flash", "gemini", 1_048_576, _GEMINI,
        alt_model="gemini-3.5-flash",
    ),
    "gemini-3.5-flash": ModelSpec(
        "gemini-3.5-flash", "gemini", 1_048_576, _GEMINI,
        alt_model="gemini-3.6-flash",
    ),
    "gemini-3.5-flash-lite": ModelSpec(
        "gemini-3.5-flash-lite", "gemini", 1_048_576, _GEMINI,
        alt_model="gemini-3.1-flash-lite",
    ),
    "gemini-3.1-flash-lite": ModelSpec(
        "gemini-3.1-flash-lite", "gemini", 1_048_576, _GEMINI,
        alt_model="gemini-3.5-flash-lite",
    ),
    "gemini-3.1-pro-preview": ModelSpec(
        "gemini-3.1-pro-preview", "gemini", 1_000_000, _GEMINI,
        alt_model="gemini-3.6-flash",
    ),
    "gemini-3-deep-think-preview": ModelSpec(
        "gemini-3-deep-think-preview", "gemini", 1_000_000, _GEMINI,
        alt_model="gemini-3.6-flash",
    ),
    "gemini-3-flash-preview": ModelSpec(
        "gemini-3-flash-preview", "gemini", 200_000, _GEMINI,
        alt_model="gemini-3.6-flash",
    ),
    "gemini-3-flash-lite-preview": ModelSpec(
        "gemini-3-flash-lite-preview", "gemini", 200_000, _GEMINI,
        alt_model="gemini-3.5-flash-lite",
    ),
    "gemini-3-flash-lite": ModelSpec(
        "gemini-3-flash-lite", "gemini", 200_000, _GEMINI,
        alt_model="gemini-3.5-flash-lite",
    ),
    "gemini-2.5-pro": ModelSpec("gemini-2.5-pro", "gemini", 1_048_576, _GEMINI),
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", "gemini", 1_048_576, _GEMINI),
    "gemini-2.5-flash-lite": ModelSpec(
        "gemini-2.5-flash-lite", "gemini", 1_048_576, _GEMINI
    ),
    # ----------------------------------------------------------------- #
    # DeepSeek (first-party). Also served third-party — see the host rows.
    # ----------------------------------------------------------------- #
    "deepseek-v4-pro": ModelSpec(
        "deepseek-v4-pro", "deepseek", 1_000_000, _DEEPSEEK,
        alt_model="deepseek-v4-flash", family="deepseek-v4",
    ),
    "deepseek-v4-flash": ModelSpec(
        "deepseek-v4-flash", "deepseek", 1_000_000, _DEEPSEEK,
        alt_model="deepseek-v4-pro", family="deepseek-v4",
    ),
    # ----------------------------------------------------------------- #
    # xAI Grok. grok-4.3 is the general flagship, grok-4.5 the coding one;
    # ~30 older grok-* names (incl. the retired grok-4-1-fast-* and
    # grok-4-fast-* slugs) are server-side aliases to grok-4.3 and are
    # deliberately omitted — they resolve, but they are not distinct models.
    # ----------------------------------------------------------------- #
    # Coding/agentic flagship. Note the shorter window and the higher price:
    # over 200k prompt tokens xAI bills the WHOLE request at 2x ($4/$12).
    # Aliased by grok-build-latest, i.e. it supersedes grok-build-0.1.
    "grok-4.5": ModelSpec(
        "grok-4.5", "grok", 500_000, _GROK, alt_model="grok-4.3"
    ),
    "grok-4.3": ModelSpec(
        "grok-4.3", "grok", 1_000_000, _GROK, alt_model="grok-4.5"
    ),
    "grok-4.20-0309-reasoning": ModelSpec(
        "grok-4.20-0309-reasoning", "grok", 1_000_000, _GROK, alt_model="grok-4.3"
    ),
    "grok-4.20-0309-non-reasoning": ModelSpec(
        "grok-4.20-0309-non-reasoning", "grok", 1_000_000, _GROK_NO_REASONING,
        alt_model="grok-4.3",
    ),
    # Multi-agent: reasoning effort controls agent count (4 vs 16), not depth.
    "grok-4.20-multi-agent-0309": ModelSpec(
        "grok-4.20-multi-agent-0309", "grok", 2_000_000, _GROK, alt_model="grok-4.3"
    ),
    "grok-build-0.1": ModelSpec(
        "grok-build-0.1", "grok", 256_000, _GROK, alt_model="grok-4.5"
    ),
    # ----------------------------------------------------------------- #
    # Z.AI / GLM. Bare ids — nothing else in the registry contains "glm".
    # ----------------------------------------------------------------- #
    "glm-5.2": ModelSpec(
        "glm-5.2", "zai", 1_000_000, _GLM_EFFORT, alt_model="glm-5.1",
        family="glm-5.2",
    ),
    "glm-5.1": ModelSpec("glm-5.1", "zai", 200_000, _GLM_THINK, alt_model="glm-5"),
    "glm-5": ModelSpec("glm-5", "zai", 200_000, _GLM_THINK, alt_model="glm-4.7"),
    "glm-5-turbo": ModelSpec(
        "glm-5-turbo", "zai", 200_000, _GLM_THINK, alt_model="glm-4.7"
    ),
    "glm-4.7": ModelSpec(
        "glm-4.7", "zai", 200_000, _GLM_THINK, alt_model="glm-4.7-flash"
    ),
    "glm-4.7-flashx": ModelSpec(
        "glm-4.7-flashx", "zai", 200_000, _GLM_THINK, alt_model="glm-4.7-flash"
    ),
    "glm-4.7-flash": ModelSpec(
        "glm-4.7-flash", "zai", 200_000, _GLM_THINK, alt_model="glm-4.7-flashx"
    ),
    "glm-4.6": ModelSpec("glm-4.6", "zai", 200_000, _GLM_THINK, alt_model="glm-4.5"),
    "glm-4.5": ModelSpec(
        "glm-4.5", "zai", 128_000, _GLM_THINK, alt_model="glm-4.5-air"
    ),
    "glm-4.5-x": ModelSpec("glm-4.5-x", "zai", 128_000, _GLM_THINK, alt_model="glm-4.5"),
    "glm-4.5-air": ModelSpec(
        "glm-4.5-air", "zai", 128_000, _GLM_THINK, alt_model="glm-4.5-flash"
    ),
    "glm-4.5-airx": ModelSpec(
        "glm-4.5-airx", "zai", 128_000, _GLM_THINK, alt_model="glm-4.5-air"
    ),
    "glm-4.5-flash": ModelSpec(
        "glm-4.5-flash", "zai", 128_000, _GLM_THINK, alt_model="glm-4.5-air"
    ),
    # Pre-4.5: no `thinking` block at all.
    "glm-4-32b-0414-128k": ModelSpec(
        "glm-4-32b-0414-128k", "zai", 128_000, _GLM_NO_THINK, alt_model="glm-4.5-flash"
    ),
    # Vision lives in SEPARATE models; the text GLMs reject image content.
    "glm-5v-turbo": ModelSpec(
        "glm-5v-turbo", "zai", 200_000, _GLM_VISION_THINK, alt_model="glm-4.6v"
    ),
    "glm-4.6v": ModelSpec(
        "glm-4.6v", "zai", 128_000, _GLM_VISION_THINK, alt_model="glm-4.6v-flashx"
    ),
    "glm-4.6v-flashx": ModelSpec(
        "glm-4.6v-flashx", "zai", 128_000, _GLM_VISION_THINK, alt_model="glm-4.6v-flash"
    ),
    "glm-4.6v-flash": ModelSpec(
        "glm-4.6v-flash", "zai", 128_000, _GLM_VISION_THINK, alt_model="glm-4.6v-flashx"
    ),
    "glm-4.5v": ModelSpec(
        "glm-4.5v", "zai", 64_000, _GLM_VISION_THINK, alt_model="glm-4.6v"
    ),
    "glm-ocr": ModelSpec(
        "glm-ocr", "zai", 8_192, _GLM_VISION_NO_THINK, alt_model="glm-4.6v"
    ),
    # ----------------------------------------------------------------- #
    # Groq — a HOST, not a lab. Keys are namespaced `groq:<real-api-id>`;
    # wire_id carries the bare id. This is why routing cannot be a substring
    # guess: `groq:openai/gpt-oss-120b` contains "gpt" and
    # `groq:deepseek-*` would contain "deepseek".
    # ----------------------------------------------------------------- #
    "groq:openai/gpt-oss-120b": ModelSpec(
        "openai/gpt-oss-120b", "groq", 131_072, _GROQ_EFFORT,
        alt_model="groq:openai/gpt-oss-20b", family="gpt-oss-120b",
    ),
    "groq:openai/gpt-oss-20b": ModelSpec(
        "openai/gpt-oss-20b", "groq", 131_072, _GROQ_EFFORT,
        alt_model="groq:openai/gpt-oss-120b", family="gpt-oss-20b",
    ),
    "groq:openai/gpt-oss-safeguard-20b": ModelSpec(
        "openai/gpt-oss-safeguard-20b", "groq", 131_072, _GROQ_EFFORT,
        alt_model="groq:openai/gpt-oss-20b",
    ),
    "groq:qwen/qwen3-32b": ModelSpec(
        "qwen/qwen3-32b", "groq", 131_072, _GROQ_EFFORT,
        alt_model="groq:llama-3.3-70b-versatile",
    ),
    "groq:llama-3.3-70b-versatile": ModelSpec(
        "llama-3.3-70b-versatile", "groq", 131_072, _GROQ_PLAIN,
        alt_model="groq:llama-3.1-8b-instant", family="llama-3.3-70b",
    ),
    "groq:llama-3.1-8b-instant": ModelSpec(
        "llama-3.1-8b-instant", "groq", 131_072, _GROQ_PLAIN,
        alt_model="groq:llama-3.3-70b-versatile",
    ),
    "groq:meta-llama/llama-4-scout-17b-16e-instruct": ModelSpec(
        "meta-llama/llama-4-scout-17b-16e-instruct", "groq", 131_072, _GROQ_PLAIN,
        alt_model="groq:llama-3.3-70b-versatile",
    ),
    "groq:meta-llama/llama-4-maverick-17b-128e-instruct": ModelSpec(
        "meta-llama/llama-4-maverick-17b-128e-instruct", "groq", 131_072, _GROQ_PLAIN,
        alt_model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
    ),
    "groq:moonshotai/kimi-k2-instruct-0905": ModelSpec(
        "moonshotai/kimi-k2-instruct-0905", "groq", 262_144, _GROQ_PLAIN,
        alt_model="groq:llama-3.3-70b-versatile",
    ),
    "groq:mistral-saba-24b": ModelSpec(
        "mistral-saba-24b", "groq", 32_768, _GROQ_PLAIN,
        alt_model="groq:llama-3.3-70b-versatile",
    ),
    # Agentic systems: tools built in, not priced per token.
    "groq:groq/compound": ModelSpec(
        "groq/compound", "groq", 131_072, _GROQ_PLAIN,
        alt_model="groq:groq/compound-mini",
    ),
    "groq:groq/compound-mini": ModelSpec(
        "groq/compound-mini", "groq", 131_072, _GROQ_PLAIN,
        alt_model="groq:groq/compound",
    ),
    # ----------------------------------------------------------------- #
    # Regolo — EU host for open models. wire_id carries regolo's real,
    # sometimes mixed-case api id (live catalog: https://api.regolo.ai/models,
    # rotates ~daily). Context windows are not published; 131_072 is a
    # conservative placeholder. alt_model deliberately crosses hosts.
    # ----------------------------------------------------------------- #
    "regolo:gpt-oss-120b": ModelSpec(
        "gpt-oss-120b", "regolo", 131_072, _REGOLO_THINK,
        alt_model="groq:openai/gpt-oss-120b", family="gpt-oss-120b",
    ),
    "regolo:gpt-oss-20b": ModelSpec(
        "gpt-oss-20b", "regolo", 131_072, _REGOLO_THINK,
        alt_model="regolo:gpt-oss-120b", family="gpt-oss-20b",
    ),
    "regolo:llama-3.3-70b-instruct": ModelSpec(
        "Llama-3.3-70B-Instruct", "regolo", 131_072, _REGOLO_PLAIN,
        alt_model="groq:llama-3.3-70b-versatile", family="llama-3.3-70b",
    ),
    "regolo:mistral-small-4-119b": ModelSpec(
        "mistral-small-4-119b", "regolo", 131_072, _REGOLO_PLAIN,
        alt_model="regolo:llama-3.3-70b-instruct",
    ),
    "regolo:qwen3.5-122b": ModelSpec(
        "qwen3.5-122b", "regolo", 131_072, _REGOLO_PLAIN,
        alt_model="regolo:gpt-oss-120b",
    ),
    "regolo:qwen3-coder-next": ModelSpec(
        "qwen3-coder-next", "regolo", 131_072, _REGOLO_PLAIN,
        alt_model="regolo:qwen3.5-122b",
    ),
    # A Z.AI model served by a European host — the decoupling in one row.
    "regolo:glm5.2-beta": ModelSpec(
        "glm5.2-beta", "regolo", 131_072, _REGOLO_PLAIN,
        alt_model="glm-5.2", family="glm-5.2",
    ),
    # ----------------------------------------------------------------- #
    # Azure Foundry. Keys ARE the user-created deployment names, so key ==
    # wire_id. Reached via WORK=1 (ModelSpec.azure_alias) or by typing the
    # -dev name directly. Deployment inventory is LIVE DATA: a public model
    # existing says nothing about a deployment existing.
    # ----------------------------------------------------------------- #
    "claude-fable-5-dev": ModelSpec(
        "claude-fable-5-dev", "azure_anthropic", 1_000_000, _CLAUDE_ADAPTIVE_XHIGH
    ),
    "claude-opus-4-8-dev": ModelSpec(
        "claude-opus-4-8-dev", "azure_anthropic", 1_000_000, _CLAUDE_ADAPTIVE_XHIGH
    ),
    "claude-opus-4-7-dev": ModelSpec(
        "claude-opus-4-7-dev", "azure_anthropic", 1_000_000, _CLAUDE_ADAPTIVE_XHIGH
    ),
    "claude-opus-4-6-dev": ModelSpec(
        "claude-opus-4-6-dev", "azure_anthropic", 1_000_000, _CLAUDE_ADAPTIVE,
        alt_model="claude-opus-4-5-dev",
    ),
    "claude-opus-4-5-dev": ModelSpec(
        "claude-opus-4-5-dev", "azure_anthropic", 200_000, _CLAUDE_BUDGET
    ),
    "claude-haiku-4-5-dev": ModelSpec(
        "claude-haiku-4-5-dev", "azure_anthropic", 200_000, _CLAUDE_BUDGET
    ),
    "gpt-5.5-dev": ModelSpec(
        "gpt-5.5-dev", "azure_openai", 1_000_000, _GPT5, alt_model="gpt-5.1-dev"
    ),
    "gpt-5.4-pro-dev": ModelSpec(
        "gpt-5.4-pro-dev", "azure_openai", 272_000, _GPT5, alt_model="gpt-5.5-dev"
    ),
    "gpt-5.1-dev": ModelSpec("gpt-5.1-dev", "azure_openai", 128_000, _GPT5),
    "gpt-5-mini-dev": ModelSpec(
        "gpt-5-mini-dev", "azure_openai", 128_000, _GPT5, alt_model="gpt-5.1-dev"
    ),
    # Non-reasoning, fastest Azure deployment.
    "gpt-4.1-nano-dev": ModelSpec(
        "gpt-4.1-nano-dev", "azure_openai", 128_000, _GPT4_CHAT
    ),
    "o3-dev": ModelSpec("o3-dev", "azure_openai", 128_000, _O_SERIES),
}


# --------------------------------------------------------------------------- #
# Legacy name heuristics. These are the pre-registry substring rules, kept as
# the FALLBACK for names the registry has never heard of. Unknown must not mean
# error: a vendor can ship a model faster than this file gets updated, and
# `gllm --models` (a live probe) is the authority on what exists. So an unknown
# name still routes and still runs — it just gets guessed capabilities.
#
# Do not extend these. A new model gets a MODELS row.
# --------------------------------------------------------------------------- #
_RESPONSES_API_PREFIXES = ("o1", "o3", "o4", "gpt-5", "codex", "grok")
_CHAT_COMPLETIONS_PREFIXES = ("gpt-4", "gpt-3.5")
_GLM_VISION_PREFIXES = ("glm-5v", "glm-4.6v", "glm-4.5v", "glm-ocr")
_GLM_NO_THINKING_PREFIXES = ("glm-ocr", "glm-4-32b")
_ANTHROPIC_ADAPTIVE_MARKERS = ("4-6", "4-7", "4-8", "fable", "opus-5", "sonnet-5")

# The full ladder. An unknown model is granted every rung so gllm never blocks a
# real capability it hasn't been told about — the API is the one that 400s.
_ALL_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _legacy_use_responses_api(model: str) -> bool:
    m = (model or "").strip().lower()
    if any(m.startswith(p) for p in _RESPONSES_API_PREFIXES):
        return True
    return not any(m.startswith(p) for p in _CHAT_COMPLETIONS_PREFIXES)


def _legacy_is_glm_vision(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _GLM_VISION_PREFIXES)


def _legacy_glm_thinking(model: str) -> bool:
    m = (model or "").lower()
    return not any(m.startswith(p) for p in _GLM_NO_THINKING_PREFIXES)


def _legacy_glm_effort(model: str) -> bool:
    return (model or "").lower().startswith("glm-5.2")


def _legacy_anthropic_dialect(model: str) -> str:
    m = (model or "").lower()
    adaptive = any(marker in m for marker in _ANTHROPIC_ADAPTIVE_MARKERS)
    return "anthropic_adaptive" if adaptive else "anthropic_budget"


def _legacy_caps(provider: str, model: str) -> ModelCaps:
    """Guessed capabilities for a model with no registry row."""
    if provider in ("anthropic", "azure_anthropic"):
        return ModelCaps(
            reasoning_efforts=_ALL_EFFORTS,
            thinking_dialect=_legacy_anthropic_dialect(model),
            supports_vision=True,
            supports_pdf=True,
            supports_strict_schema=True,
        )
    if provider == "gemini":
        return ModelCaps(
            reasoning_efforts=_ALL_EFFORTS,
            thinking_dialect="gemini_budget",
            supports_vision=True,
            supports_pdf=True,
            supports_strict_schema=True,
        )
    if provider in ("openai", "azure_openai", "grok"):
        responses = _legacy_use_responses_api(model)
        return ModelCaps(
            api_surface="responses" if responses else "chat",
            reasoning_efforts=_ALL_EFFORTS if responses else (),
            thinking_dialect="openai_effort" if responses else None,
            supports_vision=True,
            # PDF is `input_file`, Responses-only — and xAI has no equivalent.
            supports_pdf=responses and provider != "grok",
            supports_strict_schema=True,
        )
    if provider == "zai":
        vision = _legacy_is_glm_vision(model)
        if not _legacy_glm_thinking(model):
            return ModelCaps(supports_vision=vision)
        dialect = "zai_effort" if _legacy_glm_effort(model) else "zai_thinking"
        return ModelCaps(
            reasoning_efforts=_ALL_EFFORTS,
            thinking_dialect=dialect,
            supports_vision=vision,
        )
    if provider in ("groq", "regolo"):
        return ModelCaps(
            reasoning_efforts=("low", "medium", "high"),
            thinking_dialect=(
                "compat_thinking_flag" if provider == "regolo" else "compat_effort"
            ),
        )
    # deepseek and anything genuinely unknown: no control surface we can assume.
    return ModelCaps()


# --------------------------------------------------------------------------- #
# Public lookups
# --------------------------------------------------------------------------- #
def spec_for(model: str) -> ModelSpec | None:
    return MODELS.get((model or "").strip().lower())


def caps_for(model: str, provider: str) -> ModelCaps:
    """Capabilities for a model. Registry row if we have one, guess if not."""
    spec = spec_for(model)
    return spec.caps if spec is not None else _legacy_caps(provider, model)


def wire_id_for(model: str) -> str:
    """The literal string to send over the wire. Unregistered names pass
    through unchanged — the adapter forwards whatever it was given."""
    spec = spec_for(model)
    return spec.wire_id if spec is not None else model


def context_window_for(model: str, default: int = 128_000) -> int:
    spec = spec_for(model)
    return spec.context_window if spec is not None else default
