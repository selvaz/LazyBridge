"""Tool-level description derivation from a docstring (no explicit description=).

Covers the change from "first physical line only" to "first paragraph, with
wrapped lines collapsed to single spaces" in ToolSchemaBuilder.build_artifact.
A well-formed Google-style docstring (one-line summary, blank line, then
Args:/extended prose) is unaffected: the behaviour only differs for a summary
that itself wraps across multiple physical lines.
"""

from __future__ import annotations

from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

_builder = ToolSchemaBuilder()


def _describe(func) -> str:
    defn = _builder.build(func, name=func.__name__, description=None, strict=False, mode=ToolSchemaMode.SIGNATURE)
    return defn.description


def test_single_line_summary_is_unaffected() -> None:
    def fn() -> None:
        """Add two numbers.

        Args:
            a: first
        """

    assert _describe(fn) == "Add two numbers."


def test_wrapped_summary_paragraph_is_joined_with_spaces() -> None:
    def fn() -> None:
        """Fit an OLS regression across
        multiple regressors, computing
        standard errors.

        Args:
            a: first
        """

    assert _describe(fn) == "Fit an OLS regression across multiple regressors, computing standard errors."


def test_text_after_the_first_blank_line_is_excluded() -> None:
    def fn() -> None:
        """One-line summary.

        Extended prose for humans that must never reach the model: internal
        migration notes, audit refs, TODOs.

        Args:
            a: first
        """

    description = _describe(fn)
    assert description == "One-line summary."
    assert "Extended prose" not in description
    assert "migration" not in description


def test_no_docstring_yields_empty_description() -> None:
    def fn() -> None:
        pass

    assert _describe(fn) == ""


def test_explicit_description_always_wins_over_the_docstring() -> None:
    def fn() -> None:
        """Wrapped across
        two lines.
        """

    defn = _builder.build(fn, name="fn", description="Explicit override.", strict=False, mode=ToolSchemaMode.SIGNATURE)
    assert defn.description == "Explicit override."
