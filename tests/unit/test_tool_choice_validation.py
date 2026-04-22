"""Tests for ``CompletionRequest.tool_choice`` pre-validation.

Closes audit finding #4 — a misspelled tool name in ``tool_choice`` used
to pass silently through to the provider API, which either swallowed it
or returned a cryptic server error.  The dataclass now validates on
construction so typos fail locally before any RTT is spent.
"""

from __future__ import annotations

import pytest

from lazybridge.core.types import (
    CompletionRequest,
    Message,
    Role,
    ToolDefinition,
)


def _req(
    tool_choice: str | None,
    tools: list[ToolDefinition] | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role=Role.USER, content="hi")],
        tools=tools or [],
        tool_choice=tool_choice,
    )


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"desc for {name}",
        parameters={"type": "object", "properties": {}, "required": []},
    )


# ---------------------------------------------------------------------------
# Happy paths — every accepted shape
# ---------------------------------------------------------------------------


def test_tool_choice_none_is_valid() -> None:
    _req(None, [_tool("search")])


@pytest.mark.parametrize("kw", ["auto", "required", "none", "any"])
def test_tool_choice_meta_keywords_are_valid(kw: str) -> None:
    _req(kw, [_tool("search")])


def test_tool_choice_matching_tool_name_is_valid() -> None:
    _req("search", [_tool("search"), _tool("summarize")])


# ---------------------------------------------------------------------------
# Rejected paths — every misspelling / inconsistency
# ---------------------------------------------------------------------------


def test_tool_choice_unknown_name_with_tools_raises() -> None:
    with pytest.raises(ValueError) as exc:
        _req("serach", [_tool("search")])  # typo
    # Error message lists known tools so the user can see what to pick.
    assert "serach" in str(exc.value)
    assert "search" in str(exc.value)
    assert "Known tools" in str(exc.value)


def test_tool_choice_name_with_empty_tools_raises_clear_message() -> None:
    with pytest.raises(ValueError) as exc:
        _req("search", [])
    msg = str(exc.value)
    assert "no tools" in msg
    # Meta keywords are listed as the alternative, so the fix is obvious.
    for kw in ("auto", "required", "none", "any"):
        assert kw in msg


def test_tool_choice_empty_string_does_not_silently_pass() -> None:
    """Empty string is neither a keyword nor a valid tool name — catch it."""
    with pytest.raises(ValueError):
        _req("", [_tool("search")])


def test_tool_choice_case_sensitive_names() -> None:
    """Tool names are case-sensitive — 'Search' != 'search'."""
    with pytest.raises(ValueError):
        _req("Search", [_tool("search")])
