# Convention: file attachments via `-f`, native-or-fail

How `gllm` accepts provider-native file inputs without breaking the Unix pipe
model. Designed 2026-05-31 alongside the Opus-4.8 routing fix; extended
2026-08-04 with OpenAI's documented Office/text/spreadsheet inputs.

`#convention` `#architecture-decision-record`

## The shape: `-f PATH` (repeatable), `-` is stdin

`cli.py` exposes `-f / --file PATH` as `action="append"`. The value is anything `open()` accepts — including process substitution paths from bash `<(cmd)`. `-f -` reads `sys.stdin.buffer` as bytes (mutex with text-on-stdin in that invocation — we suppress `_read_stdin_if_piped()` when any `-f -` is present). `--mime TYPE` overrides MIME detection.

This is the *Unix* answer to "how do I pipe a file in": `-f` only needs a path; the shell already solved composition. We didn't reinvent piping.

## Domain — `Attachment` rides in `Request`

`domain.Attachment(data: bytes, mime_type: str, source_label: str)`, frozen.
`Request.attachments: tuple[Attachment, ...] = ()`. Tuple because `Request` is
frozen. `source_label` is the original path or `"<stdin>"` — used for error
messages and to derive the filename OpenAI requires.

## Native or fail (no text-extraction fallback)

Each provider uses its own native attachment API. Mismatches fail loudly with exit 2 — we rejected bebri-chat's silent "convert PDF to text via MarkItDown" path as un-Unix and dishonest about routing failures.

Capability matrix lives in `adapters/_capabilities.py` (`supports_image`, `supports_pdf`). The CLI checks it *before* invoking the adapter so the error is crisp; adapters keep their own defensive raises for programmatic callers.

| Provider | Image (native shape) | PDF (native shape) | Other documents |
|---|---|---|---|
| `anthropic` / `azure_anthropic` | `image`, base64 | `document`, base64 | fail |
| public `openai` Responses | `input_image` data URI | `input_file` | `input_file` |
| public `openai` Chat | `image_url` data URI | `file` | `file` |
| `azure_openai` Responses / Chat | same image shape | `input_file` / `file` | fail |
| `gemini` | `types.Part.from_bytes` | `types.Part.from_bytes` | fail |
| `grok` | inherits OpenAI image path | fail | fail |
| GLM / Kimi vision models | `image_url` | fail | fail |
| Groq / Regolo / DeepSeek | fail | fail | fail |

Public OpenAI's accepted non-PDF set is the documented Office, presentation,
spreadsheet, text, and source-code table. Responses uses `input_file`; Chat
Completions uses a nested `file` content part. Azure is deliberately separate:
bebri-chat verified that Azure rejects non-PDF Office files with
`unsupported_file`, although PDFs work on both Azure surfaces.

Non-PDF OpenAI documents are text-only: embedded images/charts do not enter
context. Spreadsheet augmentation reads at most the first 1,000 rows per sheet.
The API limit is 50 MB per file and 50 MB combined. gllm documents those limits
but leaves provider-specific size rejection to the API, matching existing
image/PDF behavior.

## MIME sniffing

`cli._sniff_mime(data, path_hint)` checks magic bytes first (PNG `\x89PNG`,
JPEG `\xff\xd8\xff`, GIF `GIF87a/GIF89a`, WebP `RIFF…WEBP`, PDF `%PDF-`).
It then consults the documented OpenAI file-extension registry before falling
back to `mimetypes.guess_type`. The explicit registry fixes stdlib mismatches
such as TSV, C++, Rust, YAML, and TOML. Returns `None` if all checks fail
(caller errors with "pass --mime TYPE"). Magic bytes still win over extension.

## Shared helpers

- `adapters.anthropic._anthropic_content(prompt, attachments)` — produces either the bare prompt string (unchanged wire format when no attachments) or a list of content blocks. Reused by `azure_anthropic.py` for shape parity.
- `adapters.openai._responses_input(...)` — images become `input_image`; PDFs
  and accepted documents become `input_file`.
- `adapters.openai._chat_user_content(...)` — images become `image_url`; PDFs
  and accepted documents become nested `file` parts.

## Related
- [[CONVENTIONS-multi-provider-routing]] — the routing map and the OpenAI-compatible subclass pattern.
- [[GOTCHA-azure-foundry-constraints]] — Foundry quirks unrelated to attachments.
