# Gotcha: running gllm's tests in the claude-jail (uv wipes the venv chasing Artifactory)

How to run `gllm`'s test suite inside the claude-jail sandbox, and why the naive `uv run pytest` destroys the venv there. Learned 2026-07-31 on x1 (the private box) while verifying a reasoning-ladder refactor.

`#gotcha` `#environment`

## gllm's deps are pure public PyPI — nothing here needs Artifactory

`uv.lock` resolves entirely to `files.pythonhosted.org` (384 URLs) and `pypi.org` (36); **zero `artifactory.sydbank.dev`**. So any Artifactory reference you see while working on gllm is NOT a project dependency — it is host-level toolchain cruft leaking in (see below). Do not "fix" it by adding an index to `pyproject.toml`/`uv.lock`.

## What breaks: `uv run` / `uv sync` in the jail

`uv run pytest` (and bare `uv sync`) resolve uv's **default** index. On this machine that default is the Sydbank Artifactory mirror — not from any gllm file, but from uv's HTTP cache (`~/.cache/uv`, bound into the jail) which was populated on the host under the WORK toolchain config. In the jail the Artifactory token isn't available, so the wheel fetch returns **HTTP 401**, and uv responds by **deleting `.venv` and recreating it empty** — leaving you worse off than before (imports now fail). See control-center's `artifactory-netrc-credential-model` and `START-HERE` notes for the host-side mechanism.

## The fix: rebuild off public PyPI, run pytest directly

1. Rebuild the venv from public PyPI. `--default-index` overrides the Artifactory default and still honours `uv.lock`, so versions are byte-identical to the lock:

       uv sync --default-index https://pypi.org/simple

   (The jail *can* reach pypi.org — the 401 was auth, not a network block.)

2. Run tests **without** `uv run` (which would re-sync against the Artifactory default and re-break the venv). Go through the venv interpreter, and clear `PYTEST_ADDOPTS`:

       PYTEST_ADDOPTS= .venv/bin/python -m pytest -q     # 130 passed, 0.3s

   `PYTEST_ADDOPTS=-p bebri_pytest_clipboard` is exported by the user's shell profile; that plugin isn't installed in the jail, so pytest ImportErrors at startup until the var is cleared.

## Takeaways

- In the jail, prefer `.venv/bin/python -m pytest` over `uv run pytest` for gllm.
- If the venv is already nuked, `uv sync --default-index https://pypi.org/simple` restores it losslessly.
- The Artifactory 401 is environmental, never a gllm bug. Related: `GOTCHA-azure-foundry-constraints.md` (the other place WORK-mode/corporate config touches gllm).
