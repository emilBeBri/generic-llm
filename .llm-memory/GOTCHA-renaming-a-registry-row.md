# Renaming a registry row: five coupled surfaces, one of which fails a test

tags = #gotcha #registry #pricing #azure-foundry #model-selection

A model bump is never a one-line rename. `gpt-5.6-luna` -> `gpt-6-luna`
(2026-09-27, the worked example) had to land in all of these together:

1. The `MODELS` key in `src/gllm/models.py` **and** its `wire_id` — identical
   here; they diverge only for host-namespaced rows.
2. The row's `azure_alias`, which names a real deployment, so the `-dev` row
   must be renamed and kept as a registry key of its own.
3. Every `alt_model` in OTHER rows that pointed at the old key. Miss one and
   the target dangles: `tests/test_registry.py` asserts every
   `alt_model`/`azure_alias` target is itself a registry key.
4. `data/prices.json` — keyed by REGISTRY KEY, so the `-dev` overlay row must
   be renamed in the same commit. `test_bundled_price_overrides_name_real_models`
   fails on a price key that is not a model.
5. Call sites: `tests/test_routing.py` (the WORK-redirect case,
   `effective_model("gpt-6-luna", True) == "gpt-6-luna-dev"`) and
   `scripts/latency-bench.zsh`, whose default model list is a pair of bare ids.

**Rates do not travel with the key.** The overlay's `-dev` rows are documented
as "priced from the matching public model until a Foundry-specific rate is
documented", so the renamed row has to carry the NEW model's public rates
(`gpt-6-luna`: 0.10 in / 0.50 out / 0.01 cached). A pure key rename keeps
billing the new deployment at the old model's card, and nothing complains.

The corollary bit us once in the same sweep: `gpt-5.6` (the bare alias) was still
sitting at 5/30/0.5 — the rates it had *before* the 2026-07-30 cut that took
5.6-sol to 4/20/0.4. Nothing updates a gap row when the vendor repriced its
model, so a stale card survives indefinitely; check these rows against the book
whenever the file is open anyway.

The rename is also an assumption about live Azure inventory: under `WORK=1`
`routing.effective_model` resolves through `azure_alias`, so an alias with no
deployment behind it fails only at request time. Deployment inventory is not
something the registry or its tests can check — see
[[GOTCHA-azure-foundry-constraints]].

Verified after the bump with `.venv/bin/python -m pytest`: 400 passed,
1 skipped. Rows carrying a *documented* output ceiling are the exception, not
the rule — `ModelSpec.max_output` is `None` when unsourced, and the new row
omits it exactly as its 5.6 siblings do, even though the model page states
128,000. See [[ADR-provider-model-axis]] and
[[CONVENTIONS-usage-cost-emission]].

## Wiring a new row into its generation (sol, same day)

Adding the second row of a generation is where the edges get decided.
`alt_model` is metadata — nothing under `src/gllm/` reads it — but it is
test-enforced to point at a registry key, so it has to be *right*, not merely
harmless. bebri-chat (the reference implementation) keeps alt edges INSIDE the
generation: a host outage takes the whole family anyway, while a rate-limited
sibling can retry on the other one instead of falling back a generation. So
`gpt-6-sol` <-> `gpt-6-luna` and `gpt-6-sol-dev` <-> `gpt-6-luna-dev`, and the
older rows that used to lean on `gpt-5.6-sol` (`gpt-5.5-pro`, `gpt-5.4-pro`)
now lean on `gpt-6-sol`.

Moving a generation leaves exactly one dangling reference behind: `gpt-5.6`
(the public alias of the 5.6-line sol) carried
`azure_alias="gpt-5.6-sol-dev"`, and that key disappears with the rename.
Keeping a dead `gpt-5.6-sol-dev` row alive purely as an alias target was
rejected — it would preserve a name that is no longer used. The alias instead
resolves to `gpt-6-sol-dev`, so under WORK a `gpt-5.6` call runs on the 6-Sol
deployment. That is a deliberate substitution and gllm announces the effective
model on stderr rather than hiding it; if a 5.6-line deployment comes back on
Foundry, this one line is what changes back.

## The other direction: porting a row whose fallback target is gone (2026-10-19)

`gpt-6-astra` and `gpt-6.1-sol` were added from the reference registry, and the
rename above is exactly why one of them could not be copied. The reference
points astra's `alt_model` at `gpt-5.6-sol`; that key left this registry in the
2026-09-27 sweep, so copied verbatim the edge would dangle and
`tests/test_registry.py` would fail. The rule from the sol/luna case settles
it: alt edges stay inside the generation, so both rows fall back to
`gpt-6-sol`. Astra's step down (10/50 -> 2/10) is large, which is why the
remap announcement matters — a fallback is a failure path, not a silent
downgrade.

`gpt-6.1-sol` is a MID-generation refresh, not a bump: 6.1 keeps sol's 2/10
card and 1,050,000 window, so it is a sibling row with its own key, and no
existing row's edges move. Only a change to the *model behind a key* is a
rename; a new version number alongside the old one is a new row.

Initially neither row got an `azure_alias`: no `-dev` deployment was registered
for either, and `WORK=1` therefore resolved `<key>-dev` with the documented
warning. Inventing a deployment name is the one thing worse than the warning,
because it fails only at request time.

## Registering the 6.1 Sol deployment (2026-10-21)

The user supplied `gpt-6.1-sol-dev` as the Azure deployment name. It now has
an explicit `azure_openai` row matching the public model's window and
capabilities, with `alt_model="gpt-6-sol-dev"`, and the public row points at
it through `azure_alias`. Its price overlay follows the matching public
model's 2/10/0.2 card until a Foundry-specific rate is documented, as for
`gpt-6-sol-dev`. Both explicit `-m gpt-6.1-sol-dev` and WORK redirection now
route without name guesses; Azure model listings include it. Astra still has
no registered deployment or `azure_alias`.
