"""Architectural guard: core (``lazybridge/*``) never imports from ext.

Per the core-vs-ext policy (``docs/guides/core-vs-ext.md``), extensions
depend on core but the reverse must never happen — otherwise core
stability becomes hostage to ext velocity, and circular imports become
possible.  This test enforces the rule with a static AST scan of every
core module.

The guard is intentionally simple (no import-linter dependency): it
walks the source tree, parses each ``.py`` file under ``lazybridge/``
that does NOT live under ``lazybridge/ext/``, and asserts that no
``import`` or ``from … import …`` statement targets ``lazybridge.ext``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

CORE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "lazybridge"
EXT_PREFIX = "lazybridge.ext"


def _core_python_files() -> list[pathlib.Path]:
    """Every ``.py`` file in ``lazybridge/`` that is not under ``lazybridge/ext/``.

    ``skill_docs`` is included on purpose — it builds reference pages
    introspecting the public API; importing from ext from there would
    let the build script accidentally pull in optional deps.
    """
    files: list[pathlib.Path] = []
    for path in CORE_ROOT.rglob("*.py"):
        # Skip any file under the ext sub-tree (those are allowed to
        # import from each other or from core).
        rel = path.relative_to(CORE_ROOT)
        if rel.parts and rel.parts[0] == "ext":
            continue
        files.append(path)
    return files


def _imports_from_ext(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, statement)] for any import targeting lazybridge.ext."""
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        pytest.fail(f"{path} has a syntax error: {e}")

    offences: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == EXT_PREFIX or alias.name.startswith(EXT_PREFIX + "."):
                    offences.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == EXT_PREFIX or module.startswith(EXT_PREFIX + "."):
                names = ", ".join(a.name for a in node.names)
                offences.append((node.lineno, f"from {module} import {names}"))

    return offences


def test_core_never_imports_from_lazybridge_ext() -> None:
    """Core (``lazybridge/*`` minus ``lazybridge/ext/*``) must not import
    from ``lazybridge.ext.*`` anywhere — top-level, function-local, or
    inside ``TYPE_CHECKING`` doesn't matter; we forbid the syntax.
    """
    files = _core_python_files()
    assert files, "expected at least one core .py file to scan"

    failures: list[str] = []
    for path in files:
        for lineno, statement in _imports_from_ext(path):
            failures.append(f"  {path.relative_to(CORE_ROOT.parent)}:{lineno} → {statement}")

    if failures:
        msg = (
            "Core modules must not import from lazybridge.ext (see "
            "docs/guides/core-vs-ext.md, rule #1). Found:\n" + "\n".join(failures)
        )
        pytest.fail(msg)


def test_top_level_package_is_alpha() -> None:
    """Pre-1.0: the package declares a single ``alpha`` stability marker."""
    import lazybridge

    assert lazybridge.__stability__ == "alpha"
