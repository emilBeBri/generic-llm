"""Capability gates: what can this (provider, model) actually be asked to do?

Every gate here is a thin read of `models.caps_for(model, provider)` — one
`ModelCaps` per model, either from its registry row or guessed by the legacy
substring heuristics when the name is unregistered. The gates keep their old
names and signatures so `cli.py` and the adapters read the same as before; what
changed is that the answers now come from data instead of from `if "codex" in
model` scattered across adapters.

Philosophy is unchanged: **native or fail**. A capability a model cannot honour
is a loud refusal, never a silent degradation or a text-extraction fallback.

`is_text_generation_model` is the one function here that is NOT registry-backed,
by design: it filters `gllm --models` output, which is a live probe of ids the
registry has by definition never seen.
"""

from __future__ import annotations

from ..models import (
    _legacy_glm_effort,
    _legacy_glm_thinking,
    _legacy_is_glm_vision,
    caps_for,
    spec_for,
)


def use_responses_api(model: str) -> bool:
    """Responses API (`/v1/responses`) or Chat Completions?

    Reasoning/agentic models (o-series, gpt-5 incl. 5.6, codex, grok) speak
    Responses; the classic chat line (gpt-4, gpt-4o, gpt-3.5) speaks Chat
    Completions. Unknown slugs default to Responses — OpenAI's "Responses for
    everything new" direction, and the strict superset of the two.
    """
    # The provider hint only matters for unregistered names, and every model
    # that reaches this question is OpenAI-family.
    return caps_for(model, "openai").api_surface != "chat"


def supports_image(provider: str, model: str = "") -> bool:
    """Can this model take image input?

    `model` is optional only so older provider-level callers keep working; pass
    it. It is what makes the GLM vision split declarative — the text GLMs reject
    images, and that used to be an adapter-internal raise.
    """
    return caps_for(model, provider).supports_vision


def supports_pdf(provider: str, model: str) -> bool:
    """Native document input. Anthropic and Gemini take PDFs on every model;
    OpenAI only via `input_file` on the Responses API; nobody else."""
    return caps_for(model, provider).supports_pdf


def supports_reasoning(provider: str, model: str) -> bool:
    """Does this model have an effort knob at all?

    One question, not two: gllm's four rungs ALWAYS map onto a non-empty
    vocabulary via `reasoning.resolve_effort`, so there is no such thing as a
    level a reasoning-capable model cannot honour. The only refusal left is
    "this model has no knob" — DeepSeek's `grok-build-0.1`-shaped case, where
    the model reasons but cannot be graded.
    """
    return bool(caps_for(model, provider).native_efforts)


def native_efforts(provider: str, model: str) -> tuple[str, ...]:
    """The model's own effort vocabulary, cheapest first (for the -r notice)."""
    return caps_for(model, provider).native_efforts


def thinking_dialect(provider: str, model: str) -> str | None:
    """Which wire translation in `gllm.reasoning` this model's thinking uses."""
    return caps_for(model, provider).thinking_dialect


def supports_strict_schema(provider: str, model: str) -> bool:
    """Does the API NATIVELY ENFORCE a `--schema`, or would we only be asking
    nicely in the prompt? gllm refuses `--schema` rather than fake enforcement."""
    return caps_for(model, provider).supports_strict_schema


# --- GLM / Z.AI family splits, still exported for the zai adapter ------------
# Registry-backed where a row exists, substring-guessed where one doesn't.


def is_glm_vision_model(model: str) -> bool:
    spec = spec_for(model)
    if spec is not None:
        return spec.caps.supports_vision
    return _legacy_is_glm_vision(model)


def glm_supports_thinking(model: str) -> bool:
    """GLM-4.5+ chat/vision models take `thinking.type`; glm-ocr and the pre-4.5
    glm-4-32b do not — sending the block to them risks an error."""
    spec = spec_for(model)
    if spec is not None:
        return spec.caps.thinking_dialect is not None
    return _legacy_glm_thinking(model)


def glm_supports_reasoning_effort(model: str) -> bool:
    """Only glm-5.2 (and above) honour `reasoning_effort`; on every other GLM
    thinking is a binary on/off and the field is meaningless."""
    spec = spec_for(model)
    if spec is not None:
        return spec.caps.thinking_dialect == "zai_effort"
    return _legacy_glm_effort(model)


# --- `--models` output filter (NOT registry-backed; see module docstring) -----
# Two providers leak non-text models past their structured signals:
# OpenAI-compatible `models.list()` carries no capability metadata at all (just
# ids), and Gemini's `supported_actions` reports `generateContent` for
# TTS/image/music models too. So we also blocklist by name: these substrings
# flag the non-text-generation families (embeddings, speech, image, video,
# music, robotics, computer-use, moderation). Substring match on the lowercased
# id. Heuristic by necessity: a genuinely new text model carrying one of these
# tokens would be wrongly hidden — the accepted cost of name-based filtering.
_NON_TEXT_GEN_MARKERS = (
    "embedding",
    "embed-",
    "tts",
    "whisper",
    "dall-e",
    "moderation",
    "-audio",
    "audio-",
    "-image",
    "image-",
    "-realtime",
    "realtime-",
    "-transcribe",
    "transcribe-",
    "imagen",
    "veo",
    "sora",
    "video",
    "imagine",
    "lyria",
    "nano-banana",
    "robotics",
    "computer-use",
    "music",
)


def is_text_generation_model(model_id: str) -> bool:
    """Heuristic: is this model id a text-generation chat/responses model (not
    embeddings/audio/image/video/music/moderation)? Used ONLY to filter
    `--models` output — dispatch never consults it, so a false negative here
    hides a row but never blocks a real call."""
    m = (model_id or "").lower()
    return not any(marker in m for marker in _NON_TEXT_GEN_MARKERS)
