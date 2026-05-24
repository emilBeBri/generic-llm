# Convention: schema & instruction libraries

How reusable JSON Schemas and system prompts ("instructions") are organized and authored for `gllm`. Decided 2026-05-24 alongside the structured-output design.

## Where things live — two tiers

```
<repo>/data/                                # tier 1: BUNDLED (this repo, version-controlled)
├── instructions/<name>.md
└── schemas/<name>.json

~/.config/gllm/                             # tier 2: USER OVERLAY (per-machine, optional, future)
├── instructions/<name>.md
└── schemas/<name>.json
```

Both tiers contribute names to `--schema NAME` / `--system NAME`. **Resolution order: user overlay first, bundled fallback.** A file in `~/.config/gllm/schemas/foo.json` overrides `<repo>/data/schemas/foo.json` of the same name — no forking required to customize.

Naming: kebab-case, no spaces. The bare name (without extension) is what gets resolved.

Why two tiers? The bundled set ships with the tool — known-good starter recipes, always present, useful for someone landing on a fresh checkout with zero config. The user overlay is per-machine for personal additions and overrides. The project owns the bundled set; the user owns the overlay.

Until the named-lookup feature ships (see `/home/emil/.claude/plans/how-do-we-work-cosmic-scroll.md`), reference files by absolute path via `@`:

```sh
gllm --schema @<repo>/data/schemas/<name>.json
gllm --system @<repo>/data/instructions/<name>.md
```

## Schema convention: all-required + empty-string sentinel

Every property listed in `properties` is also listed in `required`. There are no truly optional fields — instead, fields that may be "absent" are typed `string` (or array/object) and the model is instructed via `description` to use an empty value when not applicable. `additionalProperties: false` is always set.

**Why this convention exists** — OpenAI's `strict: true` constrained-decoding mode requires every property in `properties` to also be in `required`, and requires `additionalProperties: false` on every object. Anthropic and Gemini are more permissive. By authoring schemas in the strict shape, the *same file* works portably across all three providers, and we avoid maintaining per-provider variants.

**How to apply** — when adding a new schema:
- Default to listing every property in `required`.
- For "optional" fields, use empty string / empty array / empty object as the absent sentinel and say so in the `description` ("Empty string if no scope applies.").
- Always set `"additionalProperties": false` on every object.
- Use `enum` for closed sets — the constrained decoding actually enforces it.

Once `_normalize_strict_schema()` is ported into our OpenAI adapter (planned), schemas that violate strict mode will be patched automatically, but it's still cleaner to author them in strict shape from the start.

## Instruction convention: imperative, output-shaped, no preamble-suppression

Instructions are written as imperative directives to the model. Two patterns worth keeping consistent:

- **Lead with the role/task**, then list rules as bullets. Models follow bulleted rules more reliably than prose paragraphs.
- **Explicitly forbid preamble/sign-off** when the output is meant to be piped (`commit-msg.md`, `code-review.md`). Otherwise you get "Here is your commit message:" wrapping the actual content, and downstream consumers have to strip it. Phrase: *"Output the X and nothing else. No code fences, no preamble."*

When an instruction expects a specific input format (e.g. unified diff), say so in the first sentence: *"You will be given a unified diff..."*. This shifts the model into the right mode immediately rather than having it try to infer.

## Starter library (seeded 2026-05-24)

Three of each, picked to cover different shapes:

| Kind | Name | Notes |
|---|---|---|
| instruction | `terse` | One-line answer, no preamble — also a useful smoke-test instruction. |
| instruction | `commit-msg` | Consumes a `git diff --cached` on stdin; writes Conventional Commits. |
| instruction | `code-review` | Consumes a diff; reports only real concerns, returns `LGTM` if none. |
| schema | `pick-int` | Scalar wrapper — simplest shape, useful for demos. |
| schema | `email-extract` | Flat object with enums (`intent`, `urgency`) and required strings. |
| schema | `commit-message` | Richer object with enum + array + the empty-string-sentinel pattern. |

The content of these files is the documentation for the conventions — when in doubt about authoring a new one, copy whichever existing example has the closest shape.

## Related
- [[IDEAS-key-loading-secret-managers]] — same idea (XDG config dir), different content (API keys).
- Plan file (outside project, can move): `/home/emil/.claude/plans/how-do-we-work-cosmic-scroll.md` — full design for the named-lookup feature that will graduate `@path` references to bare-name references.
