# gllm — generic LLM CLI

A minimal Unix-pipe-friendly CLI for calling LLMs. Reads stdin if piped, takes
an optional positional prompt, prints the model's response to stdout. Errors
and verbose logs go to stderr.

Supports Anthropic (Claude), OpenAI (GPT / o-series / gpt-5), and Google
(Gemini). Provider is selected from the model name.

## Install

```sh
cd generic-llm
uv sync
# `gllm` is now on $PATH inside the project's .venv
```

## API keys

`gllm` looks for keys in two places, in this order:

1. Process environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`).
2. A hardcoded `.env` file at `/home/emil/prog/prj/bebri-chat/.env`
   (temporary — see `.llm-memory/IDEAS-key-loading-secret-managers.md`).

Long-term plan: move to `~/.config/gllm/.env` (chmod 600) or a secret-manager
integration. For now, the path is hardcoded so `gllm` reuses the keys that
already live in the `bebri-chat` checkout.

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
| Model | `$GLLM_MODEL`, else `gemini-3-flash-preview` |
| Max tokens | 4096 |
| Temperature | provider default |

## Layout

```
src/gllm/
├── cli.py              # argparse + stdin/stdout
├── domain.py           # Request, Response
├── ports.py            # LLMProvider ABC
├── routing.py          # model-name → provider
└── adapters/
    ├── anthropic.py    # output_config.format json_schema
    ├── openai.py       # Responses + Chat Completions, json_schema
    └── gemini.py       # response_json_schema
```
