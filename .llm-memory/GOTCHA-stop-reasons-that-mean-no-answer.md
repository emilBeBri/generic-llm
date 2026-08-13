# Gotcha: three stop reasons that produce output nobody can trust

A 200 with an empty or cut-off body is the worst failure gllm can have, because
stdout looks like a successful run. Three separate cases produce it, each found a
different way, and each is handled differently.

`#gotcha` `#adapters` `#anthropic` `#openai`

## 1. Budget exhausted — warn, keep the text

`domain._TRUNCATION_REASONS` = `max_tokens` (Anthropic), `max_output_tokens`
(OpenAI Responses, inside `incomplete_details.reason`), `length` (every Chat
Completions host), plus Anthropic's `model_context_window_exceeded`. Compared
case-folded, which is what lets Gemini's `MAX_TOKENS` **enum name** share the
`max_tokens` entry — `str()` on that enum gives `FinishReason.MAX_TOKENS` and
matches nothing, so `.name` is the only usable value.

`Response.truncated` drives the stderr warning. The partial text is kept: it is
usually most of the answer. An **unrecognised** reason is deliberately not
reported as truncation — a missed warning is recoverable, a false one is not.

`model_context_window_exceeded` is a different cause (input plus generation
overflowed the window rather than hitting the cap) with the same consequence for
a reader, which is why it lives in the same set.

## 2. Refusal — raise, there is nothing to keep

`stop_reason == "refusal"` means streaming classifiers intervened, and the
message arrives with **no text block at all**. Joining text blocks yields `""`,
so without a guard gllm prints a blank line and exits 0 — indistinguishable from
a model that had nothing to say. `anthropic.raise_if_refused` raises instead,
naming the `stop_details.category` (`cyber`, `bio`, `frontier_llm`, …) and the
`explanation`, which the docs warn is not stable and is therefore surfaced
verbatim rather than matched on.

The OpenAI Responses surface has the same shape in a different place: a `refusal`
content part with no `output_text` sibling. `openai._output_text` raises there for
the identical reason.

## 3. The full Anthropic enum, so the `else` branch is a known quantity

`end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn`, `refusal`,
`model_context_window_exceeded`. Note `pause_turn`: the docs say to hand the
response back as-is in a subsequent request to let the model continue. gllm is
one-shot and does not, so a paused turn reads as a short answer — untested and
unhandled, worth knowing before someone reports a truncated Claude reply that
`truncated` says nothing about.

## Where these came from

`max_tokens` was found live: a capped Gemini call printed `1` for `23*47` — a
confident wrong answer rather than a visibly cut-off one. `refusal` and
`model_context_window_exceeded` were found by **reading bebri-chat's adapter**,
which handled both while gllm handled neither; that repo's
`silent-tool-call-truncation-bug` note had diagnosed the same class of bug months
earlier. Reading a sibling implementation found what neither testing nor the docs
had prompted.

Related: [[ADR-output-budget-resolution]] (the budget that gets exhausted, and the
clamp that keeps input and output inside the window),
[[ADR-stdlib-http-transport]] (why each of these is read from raw JSON now).
