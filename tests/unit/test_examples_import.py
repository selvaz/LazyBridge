"""Anti-rot guard for the ``examples/`` tree.

The shipped examples are real, copy-pasteable code but nothing else in the
suite touches them, so they silently rot whenever a public signature
changes.  This module gives them the cheapest possible coverage:

1. **Compile** every ``examples/**/*.py`` with :mod:`py_compile` — catches
   syntax errors and Python-version regressions in *all* examples,
   including the ones that depend on packages we don't install in CI.
2. **Import** the examples that only need ``lazybridge`` itself — catches
   broken imports, ``NameError`` at module load, and renamed public
   symbols.  Every example guards execution behind ``if __name__ ==
   "__main__":`` so importing them performs no live LLM call.

Examples that import an optional sibling (``lazytools`` /
``lazytoolkit``) or a third-party framework (``crewai`` / ``langgraph`` /
``langchain``) that we don't declare as a dependency are *compiled but not
imported* — they are skipped at the import step with a clear reason so a
missing optional extra can never turn into a red suite.

Mirrors the cheap-guard intent of ``test_doc_examples.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import py_compile
from pathlib import Path

import pytest

# Repo root is three levels up: tests/unit/<this file>.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# Top-level package names whose import means the example needs a dependency we
# don't install for the unit suite.  Such examples are compiled (syntax-checked)
# but not imported (we can't satisfy the import).  Detected via AST so that the
# same names appearing inside a docstring (e.g. "Original (LangGraph): ...") do
# not cause a false skip.
_OPTIONAL_TOP_LEVEL_PACKAGES = frozenset(
    {
        "lazytools",  # sibling lazytoolkit package
        "crewai",
        "langgraph",
        "langgraph_supervisor",
        "langchain",
        "langchain_openai",
    }
)


def _required_optional_package(source: str) -> str | None:
    """Return the first optional top-level package an example imports, if any.

    Uses the AST so only *real* import statements count — names mentioned in
    docstrings or comments are ignored.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _OPTIONAL_TOP_LEVEL_PACKAGES:
                    return top
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            if top in _OPTIONAL_TOP_LEVEL_PACKAGES:
                return top
    return None


def _all_example_files() -> list[Path]:
    return sorted(_EXAMPLES_DIR.rglob("*.py"))


def _example_id(path: Path) -> str:
    return str(path.relative_to(_EXAMPLES_DIR))


_EXAMPLE_FILES = _all_example_files()


def test_examples_directory_is_present() -> None:
    """Guard against the glob silently matching nothing (e.g. a moved tree)."""
    assert _EXAMPLE_FILES, f"No example files found under {_EXAMPLES_DIR}"


@pytest.mark.parametrize("path", _EXAMPLE_FILES, ids=[_example_id(p) for p in _EXAMPLE_FILES])
def test_example_compiles(path: Path) -> None:
    """Every example must byte-compile (syntax + version sanity)."""
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{_example_id(path)} failed to compile:\n{exc}")


@pytest.mark.parametrize("path", _EXAMPLE_FILES, ids=[_example_id(p) for p in _EXAMPLE_FILES])
def test_example_imports(path: Path) -> None:
    """Import lazybridge-only examples to catch load-time regressions.

    Examples that pull an optional sibling or third-party framework are
    skipped (they're still covered by ``test_example_compiles``).  Every
    example guards real work behind ``__main__`` so import is side-effect
    free with respect to live API calls.
    """
    if path.name == "__init__.py":
        pytest.skip("package marker — nothing to import standalone")

    source = path.read_text(encoding="utf-8")
    optional_pkg = _required_optional_package(source)
    if optional_pkg is not None:
        pytest.skip(f"requires optional dependency {optional_pkg!r}")

    module_name = f"_examples_import_check_{_example_id(path).replace('/', '_').replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"{_example_id(path)} failed to import: {exc!r}")
