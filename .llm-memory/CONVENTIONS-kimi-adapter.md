# Moonshot Kimi is a standalone one-shot Chat Completions adapter

tags = #architecture-decision-record #provider #kimi #moonshot #reasoning

Kimi uses OpenAI-compatible Chat Completions at
`https://api.moonshot.ai/v1` with `MOONSHOT_API_KEY` or `KIMI_API_KEY`, but
does not fit the generic compatibility adapter because reasoning control
differs by family:

- `kimi-k3` takes `reasoning_effort` in `low|high|max` and always reasons;
- `kimi-k2.7-code*` always reasons but rejects control parameters;
- `kimi-k2.6` takes a binary `thinking.type=enabled` block and rejects effort.

The gllm port follows [[CONVENTIONS-porting-adapters-from-reference]]: it keeps
wire facts, images, model capabilities, top-level `usage.cached_tokens`, and
pricing, but drops tools, streaming, and multi-turn `reasoning_content`
round-tripping. The adapter always uses `max_completion_tokens` with a 16k
floor because thinking and visible output share that budget. Temperature is
fixed by the provider, so an explicit CLI temperature fails rather than being
silently ignored.

Every current Kimi row accepts images through base64 `image_url`; none accepts
inline PDFs or enforces JSON Schema. See [[ADR-provider-model-axis]] for why
bare `kimi-*` rows route to the first-party Moonshot host while the existing
`groq:moonshotai/kimi-*` row remains a distinct Groq rental.

Official Moonshot material confirms K3/K2.6 vision and reasoning controls,
JSON-object mode, `models.list()`, and top-level cached tokens. The K2.7 Code
identifiers and behavior, all-row vision claim, fixed generation parameters,
and 16k minimum are inherited from the bebri-chat reference because they were
not present in the accessible official repositories on 2026-08-03.
