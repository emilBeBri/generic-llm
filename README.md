# gllm — generic LLM CLI

A minimal Unix-pipe-friendly CLI for calling LLMs. Reads stdin if piped, takes
an optional positional prompt, prints the model's response to stdout. Errors
and verbose logs go to stderr.

Supports Anthropic (Claude), OpenAI (GPT / o-series / gpt-5), Google (Gemini),
DeepSeek, xAI (Grok), and Azure AI Foundry (OpenAI + Anthropic). Provider is
selected from the model name — see [Model routing](#model-routing).

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
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `AZURE_FOUNDRY_ENDPOINT` |
| Azure Anthropic | `AZURE_ANTHROPIC_API_KEY` | `AZURE_FOUNDRY_ENDPOINT` |

Long-term plan: move to `~/.config/gllm/.env` (chmod 600) or a secret-manager
integration. For now, the path is hardcoded so `gllm` reuses the keys that
already live in the `bebri-chat` checkout.

## Model routing

Provider is inferred from the model name. The Azure Foundry `-dev` suffix is
the explicit Azure marker and is checked first.

| Model name matches | Provider |
|---|---|
| ends in `-dev`, contains `claude` | `azure_anthropic` |
| ends in `-dev` (otherwise) | `azure_openai` |
| contains `claude` | `anthropic` |
| contains `gemini` | `gemini` |
| contains `deepseek` | `deepseek` |
| contains `grok` | `grok` |
| anything else (`gpt-*`, `o1/o3/o4`, `codex`) | `openai` |

```sh
gllm -m deepseek-v4-pro "..."
gllm -m grok-4.3 "..."
gllm -m gpt-5.1-dev "..."             # Azure OpenAI (Foundry MaaS)
gllm -m claude-opus-4-7-dev "..."     # Azure Anthropic (Foundry)
```

### WORK mode (Azure Anthropic forced thinking)

Set `WORK=1` (or `WORK_ENV=1`) to force maximum extended thinking on Azure
Anthropic `-dev` models — adaptive thinking for Claude 4.6/4.7, a fixed budget
for 4.5 and older. Azure Foundry doesn't expose Anthropic's `output_config`
effort control, so this is the only thinking knob. It also bumps `max_tokens`
and drops `temperature` (extended thinking pins it to 1). No effect on any
other provider.

```sh
WORK=1 gllm -m claude-opus-4-7-dev "think hard about this"
```

## Usage

```sh
# Pipe text in, get text out
echo "rewrite as a haiku: the rain falls" | gllm

# Positional prompt
gllm "what is 2 + 2?"

# Combine: positional is the instruction, stdin is the data
cat README.md | gllm "summarize this in one sentence"

# Pick a model (provider auto-detected from the name)
gllm -m claude-opus-4-7 "explain monads"
gllm -m gpt-5-nano "..."
gllm -m gemini-3-flash-preview "..."

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
| Model | `$GLLM_MODEL`, else `deepseek-v4-flash` |
| Max tokens | 4096 |
| Temperature | provider default |

## Layout

```
src/gllm/
├── cli.py              # argparse + stdin/stdout
├── config.py           # WORK / WORK_ENV toggle
├── domain.py           # Request, Response
├── ports.py            # LLMProvider ABC
├── routing.py          # model-name → provider
└── adapters/
    ├── _capabilities.py # OpenAI Responses-vs-Chat dispatch (shared)
    ├── anthropic.py     # output_config.format json_schema
    ├── openai.py        # Responses + Chat Completions, json_schema
    ├── gemini.py        # response_json_schema
    ├── deepseek.py      # OpenAI-compatible @ api.deepseek.com
    ├── grok.py          # OpenAIProvider subclass @ api.x.ai/v1
    ├── azure_openai.py  # OpenAIProvider subclass @ Foundry MaaS
    └── azure_anthropic.py # AnthropicFoundry + WORK-mode thinking
```
