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
