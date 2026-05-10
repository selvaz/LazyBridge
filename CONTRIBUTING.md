# Contributing to LazyBridge

Welcome.  This page is the minimum you need to run the test suite,
lint, type-check, and build the docs locally.  See
[`IMPLEMENTATION.md`](IMPLEMENTATION.md) for the overarching roadmap
and [`SECURITY.md`](SECURITY.md) for the threat model.

## Bootstrap

```bash
# Editable install with the test extra (pytest, pytest-asyncio,
# pytest-cov, pydantic, pyyaml).  Adding [all] pulls every provider
# SDK + opt-in extras so the full suite runs.
python -m pip install -e '.[test,all]'
```

If you only need the core test suite (no provider SDKs):

```bash
python -m pip install -e '.[test]'
```

If you only want to hack on docs:

```bash
python -m pip install -e '.[docs]'
```

## Run the checks

| Command | Closes which CI job |
|---|---|
| `pytest` | unit tests (1610+ in 0.7.9) |
| `python -m ruff check lazybridge tests` | lint |
| `python -m ruff format --check lazybridge tests` | format |
| `python -m mypy lazybridge` | type-check (default + Phase-4/6 strict tiers) |
| `python -m lazybridge.skill_docs._build --check` | SKILL.md drift |
| `python -m mkdocs build --strict` | docs build |

All five must be green before a PR merges.

## pre-commit

```bash
python -m pip install pre-commit
pre-commit install
```

The hook configuration lives in `.pre-commit-config.yaml` and runs
ruff, ruff-format, check-yaml, check-toml, detect-private-key,
end-of-file-fixer, and trailing-whitespace.

## Common pitfalls

- **`pytest -q` reports "async def functions are not natively
  supported"** — you installed without the `[test]` extra.  Run
  `pip install -e '.[test]'` (or `[test,all]`).  The repo's
  `tests/conftest.py` will also surface a clearer one-line install
  hint if it detects this state.

- **`mypy lazybridge/` fails on a freshly cloned repo** — the
  strict-tier overrides in `pyproject.toml` cover
  `lazybridge.envelope` / `predicates` / `sentinels` (Phase 4) and
  `lazybridge.engines.*` / `lazybridge.core.providers.*` (Phase 6).
  Adding a bare `Envelope`/`list`/`dict` annotation to those modules
  trips strict mode even if the rest of the codebase is silent.

- **`mkdocs build --strict` fails with `No module named mkdocs`** —
  the `[docs]` extra wasn't installed.  Run `pip install -e '.[docs]'`.

- **The provider-capability table in `docs/reference/providers.md`
  doesn't render** — that page uses the `tools/mkdocs_provider_table.py`
  hook (registered in `mkdocs.yml`).  The hook materialises the
  `<!-- PROVIDER_CAPABILITY_TABLE -->` marker at build time from
  `lazybridge.matrix.provider_capabilities()`.  Edit per-provider
  `ClassVar`s, not the marker.

## Code style

Default to writing no comments.  Add one only when the WHY is
non-obvious (hidden constraint, subtle invariant, workaround for a
specific upstream bug).  See
[`docs/for-llms/codegen-contract.md`](docs/for-llms/codegen-contract.md)
for the canonical-form rules — they apply equally to humans.

## PR checklist

- [ ] `pytest` green locally
- [ ] `ruff check` + `ruff format --check` clean
- [ ] `mypy lazybridge` clean (default and strict tiers)
- [ ] If touching `SKILL.md`, run `python -m lazybridge.skill_docs._build --check`
- [ ] If touching `docs/`, run `mkdocs build --strict`
- [ ] CHANGELOG entry under `[Unreleased]` or the active version section
