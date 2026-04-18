"""Drift-guard for the tier matrix (audit F2 / C2).

The canonical matrix lives in ``lazy_wiki/human/agents.md``.  This test
re-generates the matrix from the per-provider ``_TIER_ALIASES`` tables
and asserts the canonical page still matches.  If anyone edits a tier
table and forgets to re-run ``tools/generate_tier_matrix.py``, this
test fails at PR review time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL = _REPO_ROOT / "lazy_wiki" / "human" / "agents.md"


def _extract_matrix(md: str) -> str | None:
    """Pull the tier matrix out of ``agents.md``.

    The canonical section starts with a ``| tier |`` header row, takes the
    row-separator row, then 5 data rows (top → super_cheap).
    """
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("| tier |") and "cheap" in md and "top" in md:
            table = [line] + lines[i + 1 : i + 7]
            return "\n".join(table) + "\n"
    return None


def _generator_output() -> str:
    from tools.generate_tier_matrix import render  # type: ignore

    return render()


def test_generator_importable():
    # The generator script must be importable as a module so tests and
    # CI can use it without running a subprocess.
    import sys

    sys.path.insert(0, str(_REPO_ROOT))
    try:
        from tools.generate_tier_matrix import _PROVIDER_COLUMNS, _TIER_ORDER, render  # noqa: F401
    finally:
        if str(_REPO_ROOT) in sys.path:
            sys.path.remove(str(_REPO_ROOT))


def test_canonical_matrix_matches_provider_tables():
    import sys

    sys.path.insert(0, str(_REPO_ROOT))
    try:
        generated = _generator_output()
    finally:
        if str(_REPO_ROOT) in sys.path:
            sys.path.remove(str(_REPO_ROOT))

    md = _CANONICAL.read_text()
    canonical = _extract_matrix(md)
    assert canonical is not None, f"Could not locate the tier matrix in {_CANONICAL}; did the heading structure change?"
    # Compare line by line to make diffs readable.
    gen_lines = [ln.rstrip() for ln in generated.splitlines() if ln.strip()]
    can_lines = [ln.rstrip() for ln in canonical.splitlines() if ln.strip()]
    assert gen_lines == can_lines, (
        "Tier matrix drift detected between provider tables and "
        f"{_CANONICAL}.\n\n"
        f"GENERATED:\n{generated}\n"
        f"CANONICAL:\n{canonical}\n"
        "Run `python tools/generate_tier_matrix.py` and paste the output "
        "into lazy_wiki/human/agents.md, README.md, and "
        "lazy_wiki/bot/00_quickref.md."
    )


@pytest.mark.parametrize(
    "path,must_contain",
    [
        (_REPO_ROOT / "README.md", "| tier | `anthropic`"),
        (_REPO_ROOT / "lazy_wiki/bot/00_quickref.md", "| tier | `anthropic`"),
        (_REPO_ROOT / "docs/course/01-first-agent.md", "provider-relative **tier**"),
    ],
)
def test_secondary_references_exist(path, must_contain):
    """The condensed copies / pointers must exist so users discover the
    tier system from multiple entry points."""
    content = path.read_text()
    assert must_contain in content, f"{path} missing tier section ({must_contain!r})"
