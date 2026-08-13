# Azure Foundry (Claude) smoke test — run on the WORK box

**For:** an agent with access to the work machine (Azure Foundry keys). The personal
box this was written on has **no Azure keys**, so the Azure paths below are
implemented but **unverified**. Verify them, then report back / patch as noted.

## Why this exists

The codebase used to assume *"Azure Foundry has no `output_config`"* and therefore
(1) dropped reasoning `effort` on Azure and (2) refused `--schema` on Azure.
On 2026-06-17 Microsoft's docs were found to **contradict** that:

- Foundry documents **Effort** as a capability and every example passes
  `output_config={"effort": ...}`:
  https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/claude-models
  https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-claude

So the code was changed to send `output_config` on Azure too (via `extra_body`),
for **both** `effort` (documented) and `format` json_schema (NOT documented for
Foundry — the unverified bit). Relevant files:
- `src/gllm/adapters/azure_anthropic.py` — `generate()` builds `extra_body={"output_config": {...}}`
- `src/gllm/adapters/_capabilities.py` — `_STRICT_SCHEMA_PROVIDERS` now includes `azure_anthropic`
- `src/gllm/reasoning.py` — `anthropic_thinking()` returns the `effort` string for 4.6/4.7/4.8

## Prereqs

Needs `AZURE_ANTHROPIC_API_KEY` + `AZURE_FOUNDRY_ENDPOINT` in the environment (gllm
auto-loads `bebri-chat/.env` if present). Models route to the Azure adapter via the
`-dev` suffix, so use `-m claude-opus-4-8-dev` (or `WORK=1 -m claude-opus-4-8`).
Run gllm with `uv run gllm` from the repo root.

---

## TEST 1 — reasoning `effort` on Azure (should WORK; documented)

```sh
HARD="How many distinct ways can a 4x10 rectangle be tiled by 1x2 dominoes? Give the exact integer. Think carefully."
uv run gllm -v -r low   -m claude-opus-4-8-dev "$HARD"
uv run gllm -v -r xhigh -m claude-opus-4-8-dev "$HARD"
```

**PASS:** both exit 0 (no 400), and the `-v` `out=` token count is clearly HIGHER for
`xhigh` than `low` (effort is being graded — `output_config.effort` reached Foundry).
On the direct API this scaled ~734 → ~1514.

**FAIL modes:**
- **400 error** mentioning `output_config` / `effort` → Foundry rejects it. Unlikely
  (it's documented), but if so, report the exact message.
- **Exit 0 but tokens DON'T scale** (low ≈ xhigh) → `effort` is being ignored. Most
  likely cause: `extra_body` isn't reaching Foundry. **Fix:** in
  `azure_anthropic.generate()` (and `anthropic.generate()`), pass `output_config` as a
  **direct kwarg** to `messages.stream()/create()` instead of via `extra_body` —
  Microsoft's own examples pass `output_config={...}` directly, not under `extra_body`.
  (gllm uses `extra_body` because the stable `anthropic` SDK rejected it as a kwarg;
  the Foundry client may differ. If you change it, re-check the **direct** adapter too.)

---

## TEST 2 — `--schema` strict structured output on Azure (UNVERIFIED; the important one)

`output_config.format` json_schema is **not** in Foundry's documented capability list.
Three outcomes — find out which:

```sh
SCHEMA='{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"city":{"type":"string"}},"required":["name","age","city"],"additionalProperties":false}'
uv run gllm --schema "$SCHEMA" -m claude-opus-4-8-dev "Invent a fictional person; return name, age, city."
```

Then the **enforcement probe** — a schema that forbids prose, with a prompt that begs for it.
Compare Azure (`-dev`) against the direct API (known-enforced) as a control:

```sh
S2='{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}'
P2="Think step by step and explain your full reasoning, then answer: what is 17 + 25?"
uv run gllm --schema "$S2" -m claude-opus-4-8     "$P2"   # direct control: must be exactly {"answer": 42}
uv run gllm --schema "$S2" -m claude-opus-4-8-dev "$P2"   # Azure: does it match?
```

**Interpretation:**

| Azure result on the probe | Meaning | Action |
|---|---|---|
| Exactly `{"answer": 42}` (no prose, no extra keys), like the direct control | **Native enforcement works on Foundry** ✓ | Keep as-is. Update the docs/memory to drop the "unverified" caveats; mark `output_config.format` confirmed on Foundry. |
| **400 error** (e.g. "output_config.format not supported") | Foundry rejects `format` — fails loudly, no faking | **Re-gate Azure:** remove `"azure_anthropic"` from `_STRICT_SCHEMA_PROVIDERS` in `_capabilities.py`, and restore a `raise` on `request.schema is not None` in `azure_anthropic.generate()`. Then `--schema` errors cleanly (exit 2) instead of 400ing. |
| Prose / extra keys / not strictly the schema (i.e. it answered like plain `--json`) | **Silent-ignore — the worst case** (looks enforced, isn't) | Same fix as the 400 row: re-gate Azure so `--schema` is refused. This is exactly the "fake strict" outcome the gate exists to prevent. |

Also sanity-check that **`--json` (best-effort) still works** on Azure (it uses an
instruction, not output_config):

```sh
uv run gllm --json -m claude-opus-4-8-dev "one person as a JSON object: name, age, city"
```
Should return valid JSON, exit 0.

---

## Report back

For each test: the exact command, exit code, the `-v` token line (Test 1), and the raw
output (Test 2). If any FAIL row triggered, say which, and apply the noted fix (or leave
it for review). The general structured-output behaviour matrix across all providers is in
`./test-struct-out.zsh` if you want broader coverage.

---

# TODO added 2026-08-13: the stdlib-transport rewrite also needs verifying here

Both Azure adapters were rewritten off the vendor SDKs (see
`.llm-memory/ADR-stdlib-http-transport.md`) and **neither path could be exercised
live** — this box has no Azure keys, so they are covered by unit tests only. Run
these on the work box alongside the tests above.

## `azure_openai` — URL derivation and both surfaces

The adapter no longer uses the `openai` SDK; it builds the URL itself as
`<AZURE_FOUNDRY_ENDPOINT>/v1` + `/responses` or `/chat/completions`, and
authenticates with `Authorization: Bearer <key>` (the same header the SDK sent).
Unit-tested: `/v1` is appended to a bare endpoint and not doubled on one that
already ends in `/v1`.

```sh
gllm -v -m gpt-5.1-dev "reply with exactly: azure responses ok"   # Responses surface
gllm -v -m gpt-4o-dev  "reply with exactly: azure chat ok"        # Chat surface
gllm --models azure_openai                                        # registry rows, no live probe
```

- **A 404 means the URL shape is wrong** — most likely the endpoint needs the
  classic `/openai/deployments/<name>/...` path (Azure OpenAI *Service*) rather
  than Foundry MaaS `/v1/`. That distinction is called out at the top of
  `azure_openai.py`; if it is wrong, fix it there.
- **A 401 means the auth header is wrong** — Azure OpenAI Service uses an
  `api-key:` header rather than `Authorization: Bearer`. The SDK sent Bearer, so
  this should be unchanged, but it has never been observed working.

## `azure_anthropic` — SSE reassembly

This one changed more. It always streams (long thinking generations can hit
Anthropic's documented 10-minute non-streaming ceiling, and gllm now defaults
Claude to a 128k output budget, making that likelier). Where it used to call
`stream.get_final_message()` in the SDK, it now parses the SSE event stream
itself and rebuilds the final message from `message_start` /
`content_block_delta` / `message_delta`.

```sh
gllm -v -m claude-opus-4-8-dev -r high "explain TCP slow start in two sentences"
```

Check all of: non-empty text, a sane `-v` token line (input AND output non-zero —
output tokens arrive only in `message_delta`, so a zero there means the reassembly
missed that event), and `--usage` reporting `stop_reason` `end_turn` rather than
`null`.

- **Empty output with a 200** means the event reassembly is wrong for Foundry —
  most likely Foundry frames SSE differently. Dump the raw stream and compare
  against the shapes in `platform.claude.com/docs/en/build-with-claude/streaming`.
- Also confirm `--usage` shows non-zero `input_tokens`: those come from
  `message_start`, a different event than the output count.
