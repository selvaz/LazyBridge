"""Example-rot guard — every file under ``examples/`` must stay in sync
with the public API.

The ``examples/`` directory is never executed in CI (most scripts need
provider keys or third-party frameworks), so an API rename silently broke
examples until a user hit the ImportError.  This guard closes the gap
cheaply, without running anything:

* every example must **compile** (syntax holds across Python versions);
* every ``import lazybridge...`` / ``from lazybridge... import X`` in an
  example must resolve against the installed package — the exact failure
  a reader would hit on line 1 of a rotten example.

Third-party imports (crewai, langgraph, ...) are deliberately NOT
resolved: interop examples must stay checkable without installing the
frameworks they demonstrate.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).parents[2] / "examples"

_EXAMPLE_FILES = sorted(
    p for p in _EXAMPLES_DIR.rglob("*.py") if "__pycache__" not in p.parts
)


def _ids(path: Path) -> str:
    return str(path.relative_to(_EXAMPLES_DIR))


def test_examples_directory_found() -> None:
    """Guard the guard: a moved/renamed examples dir must fail loudly,
    not silently parametrize zero tests."""
    assert _EXAMPLE_FILES, f"no example files found under {_EXAMPLES_DIR}"


@pytest.mark.parametrize("path", _EXAMPLE_FILES, ids=_ids)
def test_example_compiles(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


@pytest.mark.parametrize("path", _EXAMPLE_FILES, ids=_ids)
def test_example_lazybridge_imports_resolve(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) are example-local — skip.
            if node.level or not node.module or not node.module.startswith("lazybridge"):
                continue
            try:
                mod = importlib.import_module(node.module)
            except ImportError as exc:
                problems.append(f"line {node.lineno}: cannot import {node.module!r}: {exc}")
                continue
            for alias in node.names:
                if alias.name != "*" and not hasattr(mod, alias.name):
                    problems.append(
                        f"line {node.lineno}: {node.module!r} has no attribute {alias.name!r}"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("lazybridge"):
                    continue
                try:
                    importlib.import_module(alias.name)
                except ImportError as exc:
                    problems.append(f"line {node.lineno}: cannot import {alias.name!r}: {exc}")

    assert not problems, f"{path.relative_to(_EXAMPLES_DIR)} is out of sync with the API:\n" + "\n".join(
        problems
    )
