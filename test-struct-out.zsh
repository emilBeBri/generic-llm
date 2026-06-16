#!/usr/bin/env zsh
# Structured-output behaviour matrix for gllm. Run it yourself:
#
#   cd /home/emil/prog/prj/generic-llm && ./test-struct-out.zsh
#
# Two distinct features are exercised:
#   --schema  -> STRICT, schema-enforced output. Native on Anthropic(direct)/
#               OpenAI/Azure-OpenAI/Gemini/Grok; REFUSED (exit 2) on providers
#               that can only fake it (DeepSeek, Azure-Anthropic).
#   --json    -> best-effort "just give me JSON", no schema. Native json mode on
#               OpenAI/Azure-OpenAI/Grok/DeepSeek/Gemini; instruction-only on
#               Anthropic (its API has NO schemaless json mode — verified against
#               platform.claude.com/docs/en/build-with-claude/structured-outputs).
#
# Parts 1-2 need the relevant provider key (loaded from bebri-chat/.env).
# Part 3 (the refusals) needs NO key — the gate runs before any API call.

set -u
cd ${0:A:h}

GLLM=(uv run gllm)          # override if needed, e.g. GLLM=(.venv/bin/gllm)

# Project schema convention: all-required + additionalProperties:false.
SCHEMA='{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"city":{"type":"string"}},"required":["name","age","city"],"additionalProperties":false}'
PROMPT="Invent a fictional person; return their name, age, and city."

# Run a call, print stdout, and check it parses as JSON.
jcheck() {
    local label=$1; shift
    print -P "%F{cyan}-- ${label} --%f"
    local out ec
    out=$("${GLLM[@]}" "$@" 2>/dev/null); ec=$?
    print -r -- "$out"
    if [[ $ec -ne 0 ]]; then
        print -P "%F{red}exit=$ec (call failed)%f"
    elif print -r -- "$out" | python3 -c 'import sys,json;json.load(sys.stdin)' 2>/dev/null; then
        print -P "%F{green}exit=0  parses as JSON ✓%f"
    else
        print -P "%F{yellow}exit=0  NOT valid JSON ✗%f"
    fi
    print
}

# Run a call that SHOULD be refused; show the stderr message and check exit==2.
expectfail() {
    local label=$1; shift
    print -P "%F{cyan}-- ${label} --%f"
    local err ec
    err=$("${GLLM[@]}" "$@" 2>&1 >/dev/null); ec=$?
    print -r -- "$err" | rg -v VIRTUAL_ENV
    if [[ $ec -eq 2 ]]; then
        print -P "%F{green}exit=2 (refused, as expected) ✓%f"
    else
        print -P "%F{red}exit=$ec (expected 2!) ✗%f"
    fi
    print
}

print -P "%F{yellow}### Part 1 — --schema: STRICT, schema-enforced (each should parse) ###%f"
jcheck "anthropic   claude-opus-4-8 (native output_config.format)" --schema "$SCHEMA" -m claude-opus-4-8 "$PROMPT"
jcheck "openai      gpt-5.1 (Responses, json_schema strict)"       --schema "$SCHEMA" -m gpt-5.1 "$PROMPT"
jcheck "openai-chat gpt-4o (Chat Completions, json_schema strict)" --schema "$SCHEMA" -m gpt-4o "$PROMPT"
jcheck "gemini      gemini-3.5-flash (response_json_schema)"       --schema "$SCHEMA" -m gemini-3.5-flash "$PROMPT"
jcheck "grok        grok-4 (Responses, json_schema strict)"        --schema "$SCHEMA" -m grok-4 "$PROMPT"

print -P "%F{yellow}### Part 2 — --json: best-effort JSON, NO schema (should parse, not guaranteed-shaped) ###%f"
jcheck "anthropic   claude-opus-4-8 (INSTRUCTION-only — no native json mode)" --json -m claude-opus-4-8 "one person as a JSON object: name, age, city"
jcheck "openai      gpt-5.1 (native json_object)"                  --json -m gpt-5.1 "one person as a JSON object: name, age, city"
jcheck "deepseek    deepseek-v4-flash (native json_object)"        --json -m deepseek-v4-flash "one person as a JSON object: name, age, city"
jcheck "gemini      gemini-3.5-flash (native application/json)"    --json -m gemini-3.5-flash "one person as a JSON object: name, age, city"

print -P "%F{yellow}### Part 3 — --schema on the one faker (DeepSeek): REFUSED, exit 2 (no API key needed) ###%f"
expectfail "deepseek    deepseek-v4-flash --schema (no native enforcement)" --schema "$SCHEMA" -m deepseek-v4-flash "$PROMPT"
# DeepSeek can still do best-effort --json (proves the gate targets --schema only):
jcheck "deepseek    deepseek-v4-flash --json (best-effort still works)" --json -m deepseek-v4-flash "one person as a JSON object: name, age, city"
# NOTE: Azure Anthropic (claude-opus-4-8-dev --schema) is NO LONGER refused — Foundry
# exposes output_config, so gllm attempts native enforcement there. That path is
# unverified and needs Azure keys; it's smoke-tested separately in
# AZURE-FOUNDRY-SMOKE-TEST.md, not here.

print -P "%F{yellow}### Part 4 — contrast: no flags = plain prose (not JSON) ###%f"
jcheck "anthropic   claude-opus-4-8 (plain text — expect NOT valid JSON)" -m claude-opus-4-8 "Name one fictional person in a sentence."

print -P "%F{green}done.%f"
