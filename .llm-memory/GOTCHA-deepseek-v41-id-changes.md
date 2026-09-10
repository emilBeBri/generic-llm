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
`.llm-memory/alert-tiers-are-not-data-tiers.md`). Assume the *next* Pro is
`deepseek-pro`.

**gllm does not adopt the versionless name.** The registry key is
`deepseek-v4.1-flash` and `wire_id` carries the vendor's `deepseek-flash` —
the same key/wire split that lets `groq:openai/gpt-oss-120b` exist, used here
for identity rather than namespacing. User preference, and the reason is
durable: a key is what you type, what `--usage` writes, and what a log has to
still be readable as next year, and `deepseek-flash` will silently mean V4.2
the day they ship it. `deepseek-flash` is kept as a compat row (same caps) so
an existing script does not quietly fall to guessed caps and lose vision.

That preference forced a related fix: `--usage` reported `response.model`, the
vendor's echo, so it printed `deepseek-flash` no matter what you typed.
`models.py` had always documented the KEY as what `--usage` reports, and the
adapters disagreed among themselves about what `Response.model` holds. It now
reports `request.model`, with the vendor's own string kept beside it as
`model_reported` whenever the two differ. `priced_as` still shows the BOOK id
(`deepseek-flash`) — the tracker is keyed by vendor id on purpose, and that
is the join actually performed.

## 1b. Which ids the API actually takes — probed, not assumed

Asserted first, verified 2026-09-10 second, which is the wrong order. The
results, straight off `POST /chat/completions`:

| sent | status | response `model` |
| --- | --- | --- |
| `deepseek-flash` | 200 | `deepseek-flash` |
| `deepseek-v4-flash` | 200 | `deepseek-flash` |
| `deepseek-v4-flash-vision-exp` | 200 | `deepseek-flash` |
| `deepseek-chat` | 200 | `deepseek-flash` |
| `deepseek-v4-pro` | 200 | `deepseek-v4-pro` |
| `deepseek-v4.1-flash` | 400 | — |
| `deepseek-v41-flash` | 400 | — |
| `deepseek-v4.1-pro` | 400 | — |

So the wire_id split is load-bearing, not stylistic: the explicit key really
is rejected.

**The 400 message lies by omission.** It reads *"The supported API model names
are deepseek-flash, deepseek-v4-pro, but you passed X"* — yet three ids that
answer 200 are absent from that list. It enumerates the CANONICAL names, not
the accepted ones. Never use that error to decide what an id does; send it.

**The response's `model` field is the honest witness.** Every alias echoes
back `deepseek-flash`, which is what exposes the reroute — and is exactly why
`--usage` keeps `model_reported` beside the registry key.

`deepseek-chat` deserves a flag: a pre-V4 name, undocumented in the current
docs, unlisted in the error, absent from the price book — and very much alive,
answering as V4.1-Flash. Vendors keep more doors open than they admit.

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

Both `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` are kept as rows,
re-pointed at `family="deepseek-v4.1-flash"` and given vision, because that is
what those ids now *are*. The vision-exp row was initially left out on the
argument that a row per dead alias makes a registry lie — reversed once the
probe showed it answering 200: an unregistered id keeps working but falls to
guessed caps, so the "clean" registry would have silently refused images to
the one model best at them. Omitting a live id is not restraint, it is a
capability regression with no error attached.

## 3. Vision became a per-model fact on DeepSeek

V4.1-Flash reads images natively — OpenAI-shaped `image_url` parts with a
base64 `data:` URL, JPEG/PNG/GIF/WebP, format sniffed from content rather than
filename or declared MIME. That retired the separate vision-exp model, so
DeepSeek now looks like GLM: `supports_image` has no honest provider-level
answer and is read off the registry row. `deepseek-v4-pro` has no image input.
The adapter's blanket "deepseek does not accept file attachments" raise is
gone; PDFs are still refused everywhere on DeepSeek. Verified live 2026-09-10
with a generated 64x64 PNG, correctly described.

### The resize is a SCALE, never a crop — measured 2026-09-10

The docs describe a pre-inference resize but never say whether an oversized or
oddly-shaped image loses part of itself. It does not. Two probes through
`gllm -f`, both fully recovered:

* 2400x2400 with a different colour in each corner and an empty middle — all
  four corners named correctly, so nothing is centre-cropped.
* 4000x250 (16:1) with markers at both extreme ends — both ends reported.

Aspect ratio is preserved and the whole frame survives. There is also **no
tiling** (unlike OpenAI's 512px tiles): one global scale, one token budget.

Token cost follows the pixel count into a clamped band. Measured with an
identical 32-token text prompt, so subtract that for the image share:

| image | `input_tokens` | image share |
| --- | --- | --- |
| none (baseline) | 32 | — |
| 100x100 | 216 | ~184 |
| 544x544 | 216 | ~184 |
| 4000x250 (1.0 MP) | 616 | ~584 |
| 1300x1300 (1.69 MP) | 1026 | ~994 |
| 2400x2400 (5.76 MP) | 1026 | ~994 |

So `tokens ≈ 1024 * pixels / 1.69M`, clamped to roughly [184, 1024]. Both
clamps are load-bearing:

* **Above ~1.69 MP you pay for nothing and see nothing.** A 4K screenshot
  costs exactly what a 1300x1300 costs, and the extra pixels are discarded
  before the model ever sees them. Downscale locally — it is free quality-wise
  and saves the upload.
* **Below ~544x544 there is a floor**, images are scaled UP and billed for it.
  A 100x100 icon costs the same as a 544x544. Shrinking past that saves nothing.

The practical consequence is that **detail loss is yours to manage**. One
global scale means fine text in a dense screenshot is destroyed (3840x2160 →
~1732x974), and an extreme aspect ratio starves its short axis (6000x500 →
~4500x375). Nothing is cut off, but nothing is preserved either. Crop to the
region of interest, or tile into several images — each image gets its OWN
resize and its own ~1024-token budget, so 4 crops cost ~4x and keep ~4x the
detail. That is a trade the caller controls; the API will not make it.

At the Flash rate a capped image is ~$0.00015 off-peak, so cost is rarely the
reason to care — legibility is.

`image_url` also takes a `detail` field (`low` = "downscaled to 512x512",
`high`/`original`/`auto` = keep). gllm never sends it, so `-f` always gets
`original`. Note the docs omit the "preserving aspect ratio" qualifier from
the `detail: low` description that the main resize section carries — whether
it squashes an oddly-shaped image is genuinely unclear, and untested.

gllm only ever sends the inline base64 path, so the size ceiling that applies
is the **48 MiB request body** (base64 inflates ~33%, so ~36 MiB of real
image); DeepSeek's 32 MiB-per-image and 64 MiB Files-API routes are not
reachable from here. Given the 1.69 MP cap you should never be near it.

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
