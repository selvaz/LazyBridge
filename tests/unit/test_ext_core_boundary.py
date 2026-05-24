"""Architectural guard: ext may only import from the public
``lazybridge`` surface — never from ``lazybridge.core.*``.

The reverse rule (core never importing from extensions) is enforced
by the sibling ``test_core_ext_boundary.py``. Together the two tests
encode the core-vs-ext policy in
``docs/guides/core-vs-ext.md``: core stays self-contained, extensions
depend only on the public API, and circular imports stay impossible.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PKG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "lazybridge"

# Subtrees whose imports we audit.  Any ``.py`` file under one of
# these may not reach into ``lazybridge.core.*``.
GUARDED_SUBTREES = ("ext",)

# Forbidden import prefixes when reaching from a guarded subtree.
FORBIDDEN_PREFIXES = ("lazybridge.core",)


def _python_files(root: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _imports_into_core(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, statement)] for any forbidden import in *path*."""
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"{path} has a syntax error: {exc}")
        return []  # unreachable — pytest.fail raises; appeases static analysers.

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


def test_extensions_never_import_from_core_internals() -> None:
    """``lazybridge.ext`` must use the public package surface
    (``from lazybridge import X``), never reach into
    ``lazybridge.core.*`` directly.
    """
    failures: list[str] = []
    scanned_any = False

    for subtree in GUARDED_SUBTREES:
        root = PKG_ROOT / subtree
        if not root.is_dir():
            continue
        for path in _python_files(root):
            scanned_any = True
            for lineno, statement in _imports_into_core(path):
                failures.append(f"  {path.relative_to(PKG_ROOT.parent)}:{lineno} → {statement}")

    assert scanned_any, "expected at least one ext .py file to scan"

    if failures:
        msg = (
            "Import boundary violation: ext may not "
            "import from internal lazybridge.core.* submodules. Use the "
            "public API (``from lazybridge import X``) instead "
            "(see docs/guides/core-vs-ext.md, rule #2). Found:\n" + "\n".join(failures)
        )
        pytest.fail(msg)


def _module_level_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, name)] for top-level (eager) imports in *path*.

    Imports nested inside a function body are skipped — only eager
    module-level imports are forbidden. (A future ``lazytools`` re-export
    would have to be lazy, inside ``__getattr__``, to stay legal.)
    """
    tree = ast.parse(path.read_text(), filename=str(path))

    def walk(body: list[ast.stmt]) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        for node in body:
            if isinstance(node, ast.Import):
                found += [(node.lineno, a.name) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                found.append((node.lineno, node.module or ""))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # function-body imports are lazy — allowed
            elif isinstance(node, ast.ClassDef):
                found += walk(node.body)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                found += walk(node.body)
                found += walk(getattr(node, "orelse", []))
                found += walk(getattr(node, "finalbody", []))
        return found

    return walk(tree.body)


def test_lazybridge_never_eagerly_imports_lazytools() -> None:
    """``lazybridge`` (shipped code) must not import ``lazytools`` at module
    level. Because ``lazytools -> lazybridge``, an eager re-export would create
    a circular import and violate the dependency rule — lazybridge has no
    runtime dependency on the toolkit.
    """
    offenders: list[str] = []
    for path in _python_files(PKG_ROOT):
        for lineno, module in _module_level_imports(path):
            if module == "lazytools" or module.startswith("lazytools."):
                offenders.append(f"  {path.relative_to(PKG_ROOT.parent)}:{lineno} → {module}")

    if offenders:
        pytest.fail(
            "lazybridge must not eagerly import lazytools (use a lazy "
            "__getattr__ shim instead). Found:\n" + "\n".join(offenders)
        )
