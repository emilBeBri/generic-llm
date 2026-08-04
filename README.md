# gllm — generic LLM CLI

A minimal Unix-pipe-friendly CLI for calling LLMs. Reads stdin if piped, takes
an optional positional prompt, prints the model's response to stdout. Errors
and verbose logs go to stderr.

Supports Anthropic (Claude), OpenAI (GPT / o-series / gpt-5), Google (Gemini),
DeepSeek, xAI (Grok), Z.AI (GLM), Moonshot (Kimi), and Azure AI Foundry
(OpenAI + Anthropic).
Provider comes from a model registry, not from the model name — see [Model routing](#model-routing).

## Install

```sh
cd generic-llm
uv sync
# `gllm` is now on $PATH inside the project's .venv
```

## API keys

`gllm` looks for keys in two places, in this order:

1. Process environment.
2. A hardcoded `.env` file at `/home/emil/prog/prj/bebri-chat/.env`
   (temporary — see `.llm-memory/IDEAS-key-loading-secret-managers.md`).

Per provider:

| Provider | Key(s) | Other env |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | |
| OpenAI | `OPENAI_API_KEY` | |
| Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | |
| DeepSeek | `DEEPSEEK_API_KEY` | |
| xAI (Grok) | `XAI_API_KEY` | |
| Z.AI (GLM) | `ZAI_API_KEY` | |
| Moonshot (Kimi) | `MOONSHOT_API_KEY` (or `KIMI_API_KEY`) | |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `AZURE_FOUNDRY_ENDPOINT` |
| Azure Anthropic | `AZURE_ANTHROPIC_API_KEY` | `AZURE_FOUNDRY_ENDPOINT` |

Long-term plan: move to `~/.config/gllm/.env` (chmod 600) or a secret-manager
integration. For now, the path is hardcoded so `gllm` reuses the keys that
already live in the `bebri-chat` checkout.

## Model routing

**Models and providers are two independent axes.** The provider comes from the
model's row in `src/gllm/models.py`, never from a substring of its name —
because the lab that *trained* a model tells you nothing about the host that
*serves* it. `openai/gpt-oss-120b` is an OpenAI model answered by Groq;
`glm5.2-beta` is a Z.AI model answered by Regolo.

| Provider | Adapter | Key |
|---|---|---|
| `anthropic` | `anthropic.py` | `ANTHROPIC_API_KEY` |
| `openai` | `openai.py` | `OPENAI_API_KEY` |
| `gemini` | `gemini.py` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `deepseek` | `deepseek.py` | `DEEPSEEK_API_KEY` |
| `grok` | `grok.py` | `XAI_API_KEY` |
| `zai` (GLM) | `zai.py` | `ZAI_API_KEY` |
| `kimi` | `kimi.py` | `MOONSHOT_API_KEY` / `KIMI_API_KEY` |
| `groq` | `openai_compat.py` | `GROQ_API_KEY` |
| `regolo` | `openai_compat.py` | `REGOLO_API_KEY` |
| `azure_openai` | `azure_openai.py` | `AZURE_OPENAI_API_KEY` + `AZURE_FOUNDRY_ENDPOINT` |
| `azure_anthropic` | `azure_anthropic.py` | `AZURE_ANTHROPIC_API_KEY` + `AZURE_FOUNDRY_ENDPOINT` |

```sh
gllm -m claude-opus-5 -r max "..."
gllm -m gpt-5.6 "..."
gllm -m deepseek-v4-pro "..."
gllm -m grok-4.3 "..."
gllm -m glm-5.2 -r high "..."         # Z.AI / GLM
gllm -m glm-4.6v -f shot.png "..."    # GLM vision model
gllm -m kimi-k3 -r xhigh "..."        # Moonshot Kimi
gllm -m gpt-5.1-dev "..."             # Azure OpenAI (Foundry MaaS)
gllm -m claude-opus-4-8-dev "..."     # Azure Anthropic (Foundry)
```

### Host providers: `groq:` and `regolo:` keys

Models served by a host rather than by their own lab carry a namespaced key
naming the host. Type the prefix; gllm strips it before the wire.

```sh
gllm -m groq:openai/gpt-oss-120b "..."   # OpenAI's open model, via Groq
gllm -m regolo:glm5.2-beta "..."         # Z.AI's GLM, via Regolo (EU)
gllm -m glm-5.2 "..."                    # ...the same family, first-party
```

Adding another OpenAI-compatible host is a `PROVIDERS` entry plus `MODELS` rows
— no new adapter code.

### Unknown models are warned, not blocked

A name with no registry row still runs: gllm guesses the provider from the name,
says so once on stderr, and dispatches.

```
gllm: 'gemini-9-flash-imaginary' is not in the model registry; routing to
'gemini' by name guess.
```

That is deliberate — vendors ship models faster than this repo gets updated, so
**there is still no model allowlist**. Treat the warning as a red flag on the
*name* (usually a misremembered or invented slug), then check with `--models`.

### Listing models

The registry says how to *drive* a model; the live API says what *exists*. Only
the second question rots, so it is asked live. Don't trust a hand-maintained
catalog for existence (a model the README calls "retired" may be live, and
vice-versa) — ask the API:

```sh
gllm --models              # every provider with a key: one `provider<TAB>key` per line
gllm --models gemini       # just one provider
gllm --models | rg flash   # plain lines — pipe to rg/fzf
```

Host rows are printed with their `groq:`/`regolo:` prefix, so any line pastes
straight into `-m`.

`--models` considers only providers configured in the current environment, then
probes each provider's live `models.list()` endpoint and prints its
**text-generation** models (embeddings/audio/image/video are filtered out).
Azure Foundry is deployment-scoped and exposes no equivalent inference endpoint,
so its rows come from the explicit `-dev` deployment entries in the registry.
Under `WORK=1`, direct Anthropic/OpenAI discovery is replaced by those configured
Foundry hosts. An explicitly requested missing or failing provider is still
reported on stderr rather than disguised as an empty list.

The families below are illustrative orientation, **not** an authoritative list —
`gllm --models` is the source of truth:

| Provider | Models (examples) |
|---|---|
| Anthropic | `claude-opus-4-5/6/7/8`, `claude-sonnet-4-5/6`, `claude-haiku-4-5/6` |
| OpenAI | `gpt-5{,-mini,-nano,-pro}`, `gpt-5.1`–`gpt-5.5`, `gpt-5-codex`, `gpt-4.1{,-mini,-nano}`, `gpt-4o{,-mini}`, `o1/o3/o4-mini` |
| Gemini | `gemini-3.5-flash`, `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-3.1-pro-preview` |
| DeepSeek | `deepseek-v4-pro`, `deepseek-v4-flash` |
| xAI Grok | `grok-4.3`, `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning`, `grok-4.20-multi-agent-0309` |
| Z.AI / GLM | text: `glm-5.2` (reasoning_effort), `glm-5.1/5/4.7/4.6/4.5`; vision: `glm-4.6v`, `glm-4.5v`, `glm-5v-turbo`, `glm-ocr` |
| Moonshot Kimi | `kimi-k3`, `kimi-k2.7-code`, `kimi-k2.7-code-highspeed`, `kimi-k2.6` |
| Azure OpenAI (`-dev`) | `gpt-5{,-mini}-dev`, `gpt-5.1-dev`, `gpt-5.2-dev`, `gpt-5.4{,-pro}-dev`, `gpt-5.5-dev`, `o3-dev` |
| Azure Anthropic (`-dev`) | `claude-opus-4-5/6/7/8-dev` |

### WORK mode (corporate / Azure)

`WORK=1` (or `WORK_ENV=1`) is the corporate/Azure switch — it redirects direct
Anthropic/OpenAI models to their Azure Foundry deployment by appending the
`-dev` marker. It has **nothing** to do with reasoning (that's `--reasoning`,
below). Default off. No effect on Gemini/Grok/DeepSeek/GLM/Kimi (no Azure variant).

```sh
WORK=1 gllm -m claude-opus-4-8 "..."   # -> azure_anthropic, deployment claude-opus-4-8-dev
WORK=1 gllm -m gpt-5.1 "..."           # -> azure_openai,    deployment gpt-5.1-dev
gllm -m claude-opus-4-8 "..."          # -> anthropic (direct)
gllm -m claude-opus-4-8-dev "..."      # -> azure_anthropic (explicit -dev, any WORK)
```

## Reasoning effort

`-r/--reasoning low|medium|high|xhigh` is one abstract knob that each provider
translates to its native control. If omitted, `$DEFAULT_EFFORT` is used when
set. If neither is set, reasoning is **hands-off** — no reasoning param is sent,
so the provider's own default applies (no behaviour change).

**Four rungs, always — that is the point.** Providers share almost no
vocabulary (DeepSeek publishes `{high, max}`, gpt-5.6 `{none..max}`, Gemini
`{minimal..high}`), so gllm normalises rather than exposing each one's dialect.
A script written against `-r high` keeps working when you change the model.

- `xhigh` always means **the most this model has**, whatever it is called there;
- every other rung keeps its own name where the provider has it, and clamps to
  the nearest where it doesn't.

So `-r low` is always the cheapest setting available — never silently upgraded.

When a rung is remapped, gllm says so on stderr (silent on a pass-through,
suppressed with `-q`):

```
gllm: -r xhigh -> 'max' (deepseek-v4-pro offers: high, max)
```

On a model with **no** effort knob at all (gpt-4o, grok-build-0.1), an
*explicit* `-r` fails loudly with exit 2. A value inherited from
`$DEFAULT_EFFORT` is an ambient default, not a request, so it is dropped there
instead — you can keep a global `DEFAULT_EFFORT=low` and still pipe to
anything.

```sh
gllm -r xhigh -m claude-opus-5 "prove it step by step"   # -> effort 'max'
gllm -r xhigh -m gpt-5.6 "tricky logic puzzle"           # -> effort 'max'
gllm -r xhigh -m deepseek-v4-pro "..."                   # -> effort 'max'
gllm -r low   -m gemini-3.5-flash "quick sanity check"
```

| Provider | Native control | how the rungs land |
|---|---|---|
| OpenAI / Grok / Azure OpenAI (Responses) | `reasoning.effort` | the level, verbatim |
| Anthropic adaptive line — 4.6+ and all of Claude 5 (direct + Azure) | `thinking.adaptive` + `output_config.effort` | the level, verbatim |
| Anthropic 4.5 & older | `thinking` budget | 8k / 16k / 32k / 32k |
| Groq / Regolo | `reasoning_effort` (+ `thinking` flag on Regolo) | the level; xhigh/max clamp to high |
| Gemini | `thinking_budget` | 4k / 8k / 16k / dynamic (`-1`) |
| Z.AI GLM-5.2 | `thinking.enabled` + `reasoning_effort` | the level, verbatim (`low`/`medium`→high, `xhigh`→max internally) |
| Z.AI GLM 4.5–5.1 | `thinking.enabled` (binary) | thinking on; effort ignored |
| DeepSeek | `thinking.enabled` + `reasoning_effort` | only {high, max}; xhigh → max |
| Kimi K3 | `reasoning_effort` | low / low / high / max |
| Kimi K2.6 | binary `thinking.enabled` | thinking on; effort ignored |
| Kimi K2.7 Code | none | always reasons, no knob → exit 2 |
| OpenAI Chat (gpt-4o), grok-build-0.1, grok-4.20-0309-*, glm-ocr / glm-4-32b | none | no knob → exit 2 |

For Anthropic/OpenAI, setting a level also bumps `max_tokens` so reasoning
doesn't starve the answer, and drops `temperature` (reasoning models reject a
custom one). One real constraint (found by live testing): modern Claude
(4.6+, and the whole Claude 5 line) **rejects** the old `thinking.type=enabled` budget shape — it needs
`adaptive` + `output_config.effort`. Azure Foundry supports `output_config.effort`
too (per Microsoft's docs), so effort grades there as well. `xhigh` may exceed
what an older model supports (some o-series, `grok-3-mini`) — a loud API 400.

## Usage

```sh
# Pipe text in, get text out
echo "rewrite as a haiku: the rain falls" | gllm

# Positional prompt
gllm "what is 2 + 2?"

# Combine: positional is the instruction, stdin is the data
cat README.md | gllm "summarize this in one sentence"

# Pick a model (provider auto-detected from the name)
gllm -m claude-opus-4-8 "explain monads"
gllm -m gpt-5-nano "..."
gllm -m gemini-3.5-flash "..."

# System prompt
gllm -s "you are terse" "what is the meaning of life?"
gllm -s @./prompts/translator.md "Hello, world"

# JSON output
gllm --json "list three planets as a JSON array of strings"

# Structured output via JSON Schema
gllm --schema ./schema.json "extract from: $(cat email.txt)"
gllm --schema '{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}' "pick a number"

# Verbose (provider/model/tokens to stderr)
gllm -v "hello" 2>>gllm.log
```

## JSON & structured output

Two flags, two different promises:

- **`--json`** — "give me JSON, *any* shape." The model picks the fields.
- **`--schema`** — "give me JSON in *exactly this* shape." You hand it a blueprint; the output is forced to match.

```sh
# --json: you don't say the shape
gllm --json "give me a person"
# {"name": "Alice", "age": 30}           ← one run
# {"full_name": "Alice", "years_old": 30} ← another run: keys can change

# --schema: you give the exact shape
gllm --schema '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}' "give me a person"
# {"name": "Alice", "age": 30}           ← same keys + types, EVERY run
```

| | what you ask for | what you get |
|---|---|---|
| `--json` | "some JSON" | valid JSON, **shape not guaranteed** |
| `--schema` | "JSON like *this*" | valid JSON, **exact shape guaranteed** |

Analogy: `--json` is "write me a note in JSON." `--schema` is "fill out *this form*" — you give it the blank form (`name: ___`, `age: ___`) and it must fill those exact blanks.

**When to use which:** `--json` for quick stuff you'll read with your own eyes; `--schema` when a program/script parses the output and needs the same fields every time (piping into `jq`, loading into code).

**Why `--schema` sometimes errors:** strict shape enforcement is a native API feature — gllm uses it on Anthropic, OpenAI, Azure-OpenAI, Gemini, and Grok. **DeepSeek and Z.AI/GLM can't** force the shape (both have only `response_format=json_object`, no native schema mode); it can only be *asked nicely* in the prompt, with no guarantee. Rather than hand you JSON that looks enforced but isn't, gllm **refuses `--schema`** there (exit 2) and tells you to use `--json` or a model with native support. `--json` (the looser ask) still works on both — it never promised an exact shape.

> Azure Anthropic (Foundry) exposes `output_config`, so `--schema` is attempted natively there too — but Foundry's `output_config.format` (strict schema) is undocumented and not yet verified; if it isn't supported the API fails loudly rather than faking. See `AZURE-FOUNDRY-SMOKE-TEST.md`.

Schema source: inline JSON, `@path/to/schema.json`, or a bare path ending in `.json`. See [Schema convention](#schema-convention-all-required--empty-string-sentinel) for the all-required authoring rule.

## File inputs

`-f PATH` attaches a binary file (image or PDF) to the request. It's repeatable.
Use `-` to read from stdin, or bash process substitution `<(cmd)` — `-f` only
needs *a path*, and the shell already knows how to compose paths with pipes.

```sh
# Plain path
gllm -m claude-opus-4-8 -f ./cat.png "describe this"

# stdin
curl -s https://example.com/img.jpg | gllm -f - --mime image/jpeg "describe"

# Process substitution — totally Unix, no special code in gllm
gllm -m gemini-3.1-pro-preview -f <(curl -s https://example.com/x.png) "ocr"

# Multiple files in one call
gllm -m claude-opus-4-8 -f a.pdf -f b.pdf "what's different?"

# xargs fan-out
fd -e png . | xargs -I{} gllm -f {} "one-line caption"
```

MIME type is sniffed from the leading bytes (PNG/JPEG/GIF/WebP/PDF magic) and
falls back to the file extension. Use `--mime TYPE` to override.

### What attaches where (native or fail)

Each provider uses its own native attachment API. If the provider has no native
mechanism for that file type, `gllm` fails fast (exit 2) — no silent
text-extraction fallback. Pick a model that fits the data.

| Provider | Image | PDF |
|---|---|---|
| Anthropic / Azure Anthropic | yes (image block) | yes (document block) |
| OpenAI / Azure OpenAI (Responses: gpt-5, o-series, codex) | yes (`input_image`) | yes (`input_file`) |
| OpenAI / Azure OpenAI (Chat: gpt-4*, gpt-3.5) | yes (`image_url`) | no |
| Gemini | yes (inline Part) | yes (inline Part) |
| xAI Grok | yes (inherits OpenAI Responses) | no |
| Z.AI / GLM | vision models only (`glm-4.6v`, `glm-4.5v`, `glm-5v-turbo`, `glm-ocr`; `image_url`) | no |
| Moonshot Kimi | yes (`image_url`) | no |
| DeepSeek | no | no |

Text files go through the existing `cat … \| gllm` pipe — `-f` is for the binary
content you can't pipe sensibly.

## Recipes — instruction & schema libraries

Reusable system prompts and JSON Schemas ship with `gllm` as plain files under
`data/`. These are the *bundled* set — always present, version-controlled,
copied in by the install. Once you create a `~/.config/gllm/` overlay (planned),
files there will be looked up first and override bundled entries by name.

```
generic-llm/data/                            # bundled (this repo)
├── instructions/
│   ├── terse.md
│   ├── commit-msg.md
│   └── code-review.md
└── schemas/
    ├── pick-int.json
    ├── email-extract.json
    └── commit-message.json

~/.config/gllm/                              # user overlay (future)
├── instructions/<name>.md                   # overrides bundled <name>.md
└── schemas/<name>.json                      # overrides bundled <name>.json
```

### Use today (path syntax)

The named-lookup feature isn't built yet, so reference files by absolute path:

```sh
GLLM_DATA=/home/emil/prog/prj/generic-llm/generic-llm/data   # or wherever your checkout lives

# System prompt from the bundled library
git diff --cached | gllm --system @$GLLM_DATA/instructions/commit-msg.md

# Structured output from the bundled library
echo "I'm John (john@x.com), urgent help needed" \
  | gllm --schema @$GLLM_DATA/schemas/email-extract.json | jq

# Both at once
git diff | gllm \
  --system @$GLLM_DATA/instructions/code-review.md \
  --schema @$GLLM_DATA/schemas/commit-message.json
```

### Use after the plan lands (named syntax)

```sh
git diff --cached | gllm --system commit-msg
echo "..."        | gllm --schema email-extract
```

Resolution order: `~/.config/gllm/{instructions,schemas}/NAME.{md,json}` first,
then bundled `data/{instructions,schemas}/NAME.{md,json}`. Drop a same-named
file in your config overlay to override a bundled one without forking.

### Picker UX via fzf

`gllm` stays a pure Unix filter — fzf integration lives in your shell config,
not in the tool. Add to `~/.zshrc`:

```sh
glx()  { gllm --schema  "$(gllm --list-schemas      | fzf)" "$@"; }
gli()  { gllm --system  "$(gllm --list-instructions | fzf)" "$@"; }
glxi() { gllm --schema  "$(gllm --list-schemas      | fzf)" \
              --system  "$(gllm --list-instructions | fzf)" "$@"; }
```

(`--list-schemas` and `--list-instructions` arrive with the same plan as the
named lookup. Until then, list bundled with `ls data/schemas/`.)

### Schema convention: all-required + empty-string sentinel

Schemas in the library mark every property as `required` and use empty strings
as the "absent" sentinel rather than truly optional fields. Reason: OpenAI's
`strict: true` mode requires every listed property to be in `required` — using
empty-string-as-absent keeps a single schema portable across Anthropic, OpenAI,
and Gemini without per-provider variants. Reflect this in your own schemas.

## Defaults

| Setting | Default |
|---|---|
| Model | `$DEFAULT_MODEL`, else `deepseek-v4-flash` |
| Reasoning | `$DEFAULT_EFFORT`, else provider default |
| Max tokens | 4096 |
| Temperature | provider default |

## Layout

```
src/gllm/
├── cli.py              # argparse + stdin/stdout
├── config.py           # WORK / WORK_ENV toggle
├── domain.py           # Request, Response
├── ports.py            # LLMProvider ABC
├── models.py           # MODELS registry: the model axis (+ caps)
├── providers.py        # PROVIDERS registry: the provider axis
├── routing.py          # model → provider (registry lookup)
├── reasoning.py        # --reasoning ladder → per-provider native shape
└── adapters/
    ├── _capabilities.py # Responses-vs-Chat dispatch + capability gates (shared)
    ├── anthropic.py     # output_config.format json_schema
    ├── openai.py        # Responses + Chat Completions, json_schema
    ├── gemini.py        # response_json_schema
    ├── deepseek.py      # OpenAI-compatible @ api.deepseek.com
    ├── zai.py           # GLM, OpenAI-compatible Chat @ api.z.ai (thinking + vision)
    ├── kimi.py          # Kimi, OpenAI-compatible Chat @ api.moonshot.ai
    ├── grok.py          # OpenAIProvider subclass @ api.x.ai/v1
    ├── azure_openai.py  # OpenAIProvider subclass @ Foundry MaaS
    └── azure_anthropic.py # AnthropicFoundry + native thinking
```
