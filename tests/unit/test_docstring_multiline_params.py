"""Multi-line-aware docstring parameter parsing (griffe) vs the regex fallback.

Before this, ``_parse_docstring_params`` captured only a single physical line
per parameter — a description wrapped across two lines silently lost its
continuation. ``griffe`` (extra: ``lazybridge[docparse]``) fixes this; the
regex parser remains as a fallback when griffe isn't installed, with the
same single-line limitation it always had.
"""

from __future__ import annotations

import builtins

import pytest

from lazybridge.core.tool_schema import (
    ToolSchemaBuilder,
    ToolSchemaMode,
    _parse_docstring_params,
    _parse_docstring_params_griffe,
    _parse_docstring_params_regex,
)

pytest.importorskip("griffe", reason="multi-line parsing needs lazybridge[docparse]")

_builder = ToolSchemaBuilder()


def _params_from(func) -> dict[str, str]:
    defn = _builder.build(func, name=func.__name__, description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
    return {name: prop.get("description") for name, prop in defn.parameters["properties"].items()}


def test_wrapped_google_style_param_is_fully_captured() -> None:
    def fn(a: int) -> None:
        """Summary.

        Args:
            a: this description wraps across
                two physical lines in the source.
        """

    assert _parse_docstring_params(fn.__doc__)["a"] == (
        "this description wraps across two physical lines in the source."
    )


def test_wrapped_sphinx_style_param_is_fully_captured() -> None:
    doc = ":param a: this description wraps across\n    two physical lines too\n"
    assert _parse_docstring_params(doc)["a"] == "this description wraps across two physical lines too"


def test_wrapped_numpy_style_param_is_fully_captured() -> None:
    doc = "Summary.\n\nParameters\n----------\na : int\n    this description wraps across\n    two physical lines\n"
    assert _parse_docstring_params(doc)["a"] == "this description wraps across two physical lines"


def test_end_to_end_schema_carries_the_full_wrapped_description() -> None:
    """The fix must reach the actual compiled tool schema, not just the parser."""

    def regression_tool(dependent: str, regressors: str) -> str:
        """Fit a regression.

        Args:
            dependent: exactly one instrument spec, comma-separated ids with
                an optional pipe-separated transform.
            regressors: comma-separated instrument specs, 1 to 10 total.
        """

    props = _params_from(regression_tool)
    assert props["dependent"] == (
        "exactly one instrument spec, comma-separated ids with an optional pipe-separated transform."
    )
    assert props["regressors"] == "comma-separated instrument specs, 1 to 10 total."


def test_regex_fallback_only_captures_the_first_physical_line() -> None:
    """Documents the known, unchanged limitation of the non-griffe fallback."""
    doc = "Summary.\n\nArgs:\n    a: wraps across\n        two lines.\n"
    assert _parse_docstring_params_regex(doc)["a"] == "wraps across"


def test_griffe_helper_returns_none_when_griffe_is_not_importable(monkeypatch) -> None:
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "griffe":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert _parse_docstring_params_griffe("Summary.\n\nArgs:\n    a: desc\n") is None


def test_parse_docstring_params_falls_back_to_regex_without_griffe(monkeypatch) -> None:
    """End-to-end: without griffe, wrapped continuations are lost (old behaviour),
    but parsing still succeeds via the regex fallback rather than raising."""
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "griffe":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    doc = "Summary.\n\nArgs:\n    a: wraps across\n        two lines.\n"
    assert _parse_docstring_params(doc)["a"] == "wraps across"


def test_empty_parameters_section_is_a_real_empty_dict_not_a_fallback_trigger() -> None:
    """A docstring with no Args: section legitimately has zero parameters —
    griffe returning {} must not be confused with "griffe unavailable"."""
    result = _parse_docstring_params_griffe("Just a summary, no parameters documented.\n")
    assert result == {}


def test_short_sphinx_docstring_defaults_correctly_when_style_detection_is_ambiguous() -> None:
    """Regression guard: infer_docstring_style returns None on a doc this
    short, so the ':param' heuristic must pick sphinx, not the google default."""
    doc = ":param int x: An integer\n"
    assert _parse_docstring_params(doc)["x"] == "An integer"
