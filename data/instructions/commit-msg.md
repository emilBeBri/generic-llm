You will be given a unified diff (the output of `git diff` or `git diff --cached`). Write a Conventional Commits message for it.

Rules:
- First line: `<type>[(scope)]: <subject>` — imperative mood, under 60 chars, no trailing period.
- Type: one of `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `build`, `ci`.
- If the diff is substantive, add a blank line and a body (wrap at 72 cols) explaining the *why*, not the what — the diff is the what.
- If the diff alters public API, append a `BREAKING CHANGE: <description>` paragraph.
- Output the commit message and nothing else. No code fences, no preamble, no "Here is your commit message:".
