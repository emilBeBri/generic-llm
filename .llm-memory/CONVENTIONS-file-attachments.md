# Convention: file attachments via `-f`, native-or-fail

How `gllm` accepts binary inputs (images, PDFs) without breaking the Unix pipe model. Designed 2026-05-31 alongside the Opus-4.8 routing fix.

`#convention` `#architecture-decision-record`

## The shape: `-f PATH` (repeatable), `-` is stdin

`cli.py` exposes `-f / --file PATH` as `action="append"`. The value is anything `open()` accepts — including process substitution paths from bash `<(cmd)`. `-f -` reads `sys.stdin.buffer` as bytes (mutex with text-on-stdin in that invocation — we suppress `_read_stdin_if_piped()` when any `-f -` is present). `--mime TYPE` overrides MIME detection.

This is the *Unix* answer to "how do I pipe a file in": `-f` only needs a path; the shell already solved composition. We didn't reinvent piping.

## Domain — `Attachment` rides in `Request`

`domain.Attachment(data: bytes, mime_type: str, source_label: str)`, frozen. `Request.attachments: tuple[Attachment, ...] = ()`. Tuple because `Request` is frozen. `source_label` is the original path or `"<stdin>"` — used for error messages and as a fallback filename for OpenAI's `input_file`.

## Native or fail (no text-extraction fallback)

Each provider uses its own native attachment API. Mismatches fail loudly with exit 2 — we rejected bebri-chat's silent "convert PDF to text via MarkItDown" path as un-Unix and dishonest about routing failures.

Capability matrix lives in `adapters/_capabilities.py` (`supports_image`, `supports_pdf`). The CLI checks it *before* invoking the adapter so the error is crisp; adapters keep their own defensive raises for programmatic callers.

| Provider | Image (native shape) | PDF (native shape) |
|---|---|---|
| `anthropic` / `azure_anthropic` | `image` block, base64 | `document` block, base64 |
| `openai` / `azure_openai` (Responses path: gpt-5*, o-series, codex) | `input_image` data-URI | `input_file` with base64 `file_data` |
| `openai` / `azure_openai` (Chat path: gpt-4*, gpt-3.5) | `image_url` data-URI | fail — no content-block type |
| `gemini` | `types.Part.from_bytes` inline | `types.Part.from_bytes` inline (~20 MB) |
| `grok` | inherits OpenAI Responses path | fail — xAI Responses has no `input_file` |
| `deepseek` | fail | fail |

## MIME sniffing

`cli._sniff_mime(data, path_hint)` checks magic bytes first (PNG `\x89PNG`, JPEG `\xff\xd8\xff`, GIF `GIF87a/GIF89a`, WebP `RIFF…WEBP`, PDF `%PDF-`). Falls back to `mimetypes.guess_type` on the path. Returns `None` if both fail (caller errors with "pass --mime TYPE"). Magic bytes win over extension — the user's fixture `tests/test-img.png` is actually JPEG and the sniffer caught it.

## Shared helpers

- `adapters.anthropic._anthropic_content(prompt, attachments)` — produces either the bare prompt string (unchanged wire format when no attachments) or a list of content blocks. Reused by `azure_anthropic.py` for shape parity.
- `adapters.openai._responses_input(prompt, attachments)` — same idea for the Responses API `input=` field. PDFs use `input_file` here.
- `adapters.openai._chat_user_content(prompt, attachments)` — Chat Completions equivalent. Images only; raises on PDF.

## Related
- [[CONVENTIONS-multi-provider-routing]] — the routing map and the OpenAI-compatible subclass pattern.
- [[GOTCHA-azure-foundry-constraints]] — Foundry quirks unrelated to attachments.
