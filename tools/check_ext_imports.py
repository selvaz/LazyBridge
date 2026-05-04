"""Architectural guard: ext, external_tools, external_pipelines may only
import from public lazybridge.* — never from lazybridge.core.*.

The rule is one half of the core-vs-ext policy
(``docs/guides/core-vs-ext.md``):

1. **Core never imports from ext / external_tools / external_pipelines.**
   Enforced by ``tests/unit/test_core_ext_boundary.py``.
2. **ext / external_tools / external_pipelines never import from
   internal core submodules** (``lazybridge.core.*``).
   Enforced here.

Run as::

    python tools/check_ext_imports.py

Exits 0 on success, 1 on violations. Used as a CI step.
"""

from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "lazybridge"

# Subtrees subject to the "no internal imports" rule.
GUARDED_SUBTREES = ("ext", "external_tools", "external_pipelines")

# Forbidden import prefixes when reaching from a guarded subtree.
FORBIDDEN_PREFIXES = ("lazybridge.core",)


def _python_files(root: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _violations(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, statement)] for any forbidden import in *path*."""
    text = path.read_text()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        print(f"SYNTAX ERROR: {path}: {exc}", file=sys.stderr)
        return []

    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                names = ", ".join(a.name for a in node.names)
                out.append((node.lineno, f"from {module} import {names}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == p or alias.name.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                    out.append((node.lineno, f"import {alias.name}"))
    return out


def main() -> int:
    failures: list[str] = []

    for subtree in GUARDED_SUBTREES:
        root = PKG / subtree
        if not root.is_dir():
            continue
        for path in _python_files(root):
            for lineno, statement in _violations(path):
                rel = path.relative_to(REPO)
                failures.append(f"  {rel}:{lineno} → {statement}")

    if failures:
        print(
            "Import boundary violation: ext / external_tools / external_pipelines\n"
            "may not import from internal lazybridge.core.* submodules.\n"
            "Use the public API (``from lazybridge import X``) instead.\n"
            "See docs/guides/core-vs-ext.md, rule #2.\n\n"
            "Found:",
            file=sys.stderr,
        )
        print("\n".join(failures), file=sys.stderr)
        return 1

    print("OK: ext / external_tools / external_pipelines respect the import boundary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
