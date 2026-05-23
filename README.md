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
2. `~/.config/gllm/.env` — simple `KEY=value` lines.

Drop your keys in `~/.config/gllm/.env` with `chmod 600` and forget about it,
or export them in your shell — both work. The file path is deliberately *not*
the cwd, so `gllm` behaves the same no matter where you pipe to it from.

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
