# DeepSeek V4.1: a versionless id, and retired models whose ids still answer

tags = #gotcha #registry #deepseek #model-selection #vision

2026-09-10, DeepSeek-V4.1-Flash. Three separate assumptions broke at once, and
none of them produced an error — which is why they are worth a note.

## 1. The id carries no version

It ships as **`deepseek-flash`**, not `deepseek-v4.1-flash`. Anything that
parses a version out of a model id gets nothing. This cost the sibling repo
its alert tier (llm-price-tracker's `watch.toml` thresholds a family on the
first dotted number, found none, and demoted the vendor's flagship to
background on launch day — see that repo's
`.llm-memory/alert-tiers-are-not-data-tiers.md`). In gllm the registry key is
the join key for pricing and `--usage`, so a versionless id is merely ugly,
not dangerous. Assume the *next* Pro is `deepseek-pro`.

## 2. A retired model's id still answers — and is NOT in `/models`

V4-Flash and V4-Flash-Vision-Exp were retired the same day, but
`deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` still resolve: DeepSeek
serves them with V4.1-Flash and bills at the Flash rate. Meanwhile
`gllm --models deepseek` (a live catalog probe, per
[[ADR-model-listing-live-probe]]) returns only `deepseek-flash` and
`deepseek-v4-pro`.

**So the catalog is a list of what is CURRENT, not of what will work.** A name
absent from `--models` may still answer, and a name present in the registry
may be an alias for something else entirely. Do not "clean up" a registry row
because the live probe stopped listing it — check whether it still resolves.

`deepseek-v4-flash` is kept as a row, re-pointed at `family="deepseek-v4.1-
flash"` and given vision, because that is what the id now *is*.
`deepseek-v4-flash-vision-exp` is deliberately left unregistered: it resolves
too, and a row for every dead alias is how a registry starts lying.

## 3. Vision became a per-model fact on DeepSeek

V4.1-Flash reads images natively — OpenAI-shaped `image_url` parts with a
base64 `data:` URL, JPEG/PNG/GIF/WebP, format sniffed from content rather than
filename or declared MIME. That retired the separate vision-exp model, so
DeepSeek now looks like GLM: `supports_image` has no honest provider-level
answer and is read off the registry row. `deepseek-v4-pro` has no image input.
The adapter's blanket "deepseek does not accept file attachments" raise is
gone; PDFs are still refused everywhere on DeepSeek. Verified live 2026-09-10
with a generated 64x64 PNG, correctly described.

## Also changed, less structurally

`reasoning_effort` gained a **`low`** rung (was `high|max`, now `low|high|max`),
so gllm's `-r low` is finally a real low instead of resolving up to `high` —
accepted live on both `deepseek-flash` and `deepseek-v4-pro`. And every price
fell: cache-hit -57%, cache-miss input -32%, output -9%. `deepseek-v4-pro`
routes to V4.1-Flash at Flash rates from **04:00 UTC 2026-09-14** until
V4.1-Pro ships, so re-check what it actually costs after that date; the
pricing page may keep publishing the V4-Pro card regardless.

See [[CONVENTIONS-one-shot-workload-no-cache]], whose DeepSeek benchmark
tables all predate this and were measured against V4-Flash at the old rates,
and [[CONVENTIONS-file-attachments]] for the capability matrix row.
