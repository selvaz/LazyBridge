# Evals specification task report

## Completed

- Added `docs/design/evals/SPEC.md`, defining project goals, decisions D1–D3, invariants, requirements R1–R18, acceptance scenarios, exclusions, assumptions, and a requirement-to-check validation matrix.
- Added `docs/design/evals/TASKS.md`, defining a one-PR-per-task dependency graph with requirement coverage, files, dependencies, exact checks, prerequisites, cloud/local execution, and opposite-engine review.
- Kept the change documentation-only and used only generic references to the private task-set repository.
- Committed each requested design document in its own commit.

## Validation

- Passed a Python consistency check confirming R1–R18 appear in the validation matrix and task plan, plus Given/When/Then acceptance vocabulary.
- Passed a Python content check confirming both documents end in a newline and contain no machine-local paths.
- Passed `git diff --check HEAD~2..HEAD`.
- `pytest -q tests/test_evals.py` could not start because `pytest-asyncio` is not installed in the environment.
- `mkdocs build --strict` could not start because `mkdocs` is not installed in the environment.

## Delivery

- Branch: `docs/evals-spec`.
- Document commits: `7b45bb5` and `fdc54bc`.
- Push was attempted, but this clone has no configured `origin` remote; the branch remains committed locally for the bridge to deliver.
