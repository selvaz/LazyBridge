"""Architectural guard: core never imports from ext / external_tools /
external_pipelines.

Per the core-vs-ext policy (``docs/guides/core-vs-ext.md``), the three
non-core subtrees depend on core but the reverse must never happen —
otherwise core stability becomes hostage to extension velocity, and
circular imports become possible.

The reverse rule (extensions never reaching into ``lazybridge.core.*``
internals) is enforced separately by ``tools/check_ext_imports.py`` in
CI.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

CORE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "lazybridge"
NON_CORE_SUBTREES = ("ext", "external_tools", "external_pipelines")
FORBIDDEN_PREFIXES = tuple(f"lazybridge.{name}" for name in NON_CORE_SUBTREES)


def _core_python_files() -> list[pathlib.Path]:
    """Every ``.py`` file under ``lazybridge/`` except the three non-core
    subtrees. ``skill_docs`` is included on purpose — it builds
    reference pages introspecting the public API; importing from an
    extension from there would let the build script accidentally pull
    in optional deps.
    """
    files: list[pathlib.Path] = []
    for path in CORE_ROOT.rglob("*.py"):
        rel = path.relative_to(CORE_ROOT)
        if rel.parts and rel.parts[0] in NON_CORE_SUBTREES:
            continue
        files.append(path)
    return files


def _imports_from_extensions(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, statement)] for any import targeting one of the
    non-core subtrees."""
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        pytest.fail(f"{path} has a syntax error: {e}")

    offences: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == p or alias.name.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                    offences.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                names = ", ".join(a.name for a in node.names)
                offences.append((node.lineno, f"from {module} import {names}"))

    return offences


def test_core_never_imports_from_extension_subtrees() -> None:
    """Core (``lazybridge/*`` minus ext/external_tools/external_pipelines)
    must not import from any of those subtrees — top-level, function-local,
    or inside ``TYPE_CHECKING`` doesn't matter; we forbid the syntax.
    """
    files = _core_python_files()
    assert files, "expected at least one core .py file to scan"

    failures: list[str] = []
    for path in files:
        for lineno, statement in _imports_from_extensions(path):
            failures.append(f"  {path.relative_to(CORE_ROOT.parent)}:{lineno} → {statement}")

    if failures:
        msg = (
            "Core modules must not import from lazybridge.ext / "
            "lazybridge.external_tools / lazybridge.external_pipelines "
            "(see docs/guides/core-vs-ext.md, rule #1). Found:\n" + "\n".join(failures)
        )
        pytest.fail(msg)


def test_top_level_package_is_alpha() -> None:
    """Pre-1.0: the package declares a single ``alpha`` stability marker."""
    import lazybridge

    assert lazybridge.__stability__ == "alpha"
