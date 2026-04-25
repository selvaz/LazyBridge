"""Unit tests for core/structured.py — T10.xx series."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lazybridge.core.structured import (
    StructuredOutputError,
    build_repair_messages,
    normalize_json_schema,
    parse_structured_output,
)
from lazybridge.core.types import Message

# ---------------------------------------------------------------------------
# T10.01 — parse valid JSON against a dict schema
# ---------------------------------------------------------------------------


def test_parse_valid_json_dict_schema():
    # T10.01
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    result = parse_structured_output('{"name": "Alice", "age": 30}', schema)
    assert result == {"name": "Alice", "age": 30}


# ---------------------------------------------------------------------------
# T10.02 — parse valid JSON against a Pydantic model
# ---------------------------------------------------------------------------


class _City(BaseModel):
    name: str
    country: str


def test_parse_valid_json_pydantic():
    # T10.02
    result = parse_structured_output('{"name": "Rome", "country": "Italy"}', _City)
    assert isinstance(result, _City)
    assert result.name == "Rome"
    assert result.country == "Italy"


# ---------------------------------------------------------------------------
# T10.03 — markdown code fence stripped before parsing
# ---------------------------------------------------------------------------


def test_parse_strips_markdown_fence():
    # T10.03
    raw = '```json\n{"name": "Berlin", "country": "Germany"}\n```'
    result = parse_structured_output(raw, _City)
    assert result.name == "Berlin"


# ---------------------------------------------------------------------------
# T10.04 — markdown fence without language tag stripped
# ---------------------------------------------------------------------------


def test_parse_strips_fence_no_lang():
    # T10.04
    raw = '```\n{"name": "Paris", "country": "France"}\n```'
    result = parse_structured_output(raw, _City)
    assert result.name == "Paris"


# ---------------------------------------------------------------------------
# T10.05 — invalid JSON raises StructuredOutputError
# ---------------------------------------------------------------------------


def test_parse_invalid_json_raises():
    # T10.05
    with pytest.raises(StructuredOutputError, match="JSON parse error"):
        parse_structured_output("this is not json", _City)


# ---------------------------------------------------------------------------
# T10.06 — dict schema validation failure raises StructuredOutputError
# ---------------------------------------------------------------------------


def test_parse_schema_validation_failure():
    # T10.06
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }
    with pytest.raises(StructuredOutputError, match="Validation error"):
        parse_structured_output('{"score": "not_an_int"}', schema)


# ---------------------------------------------------------------------------
# T10.07 — missing required field raises StructuredOutputError
# ---------------------------------------------------------------------------


def test_parse_missing_required_raises():
    # T10.07
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    # Match either format: the in-tree subset validator says "missing
    # required fields: ..."; jsonschema says "'name' is a required
    # property".  Both indicate the same failure; the test must not
    # break depending on whether jsonschema happens to be installed.
    with pytest.raises(
        StructuredOutputError, match=r"missing required|is a required property"
    ):
        parse_structured_output('{"other": "value"}', schema)


# ---------------------------------------------------------------------------
# T10.08 — Pydantic validation failure raises StructuredOutputError
# ---------------------------------------------------------------------------


def test_parse_pydantic_validation_failure():
    # T10.08
    class Strict(BaseModel):
        value: int  # must be int

    with pytest.raises(StructuredOutputError, match="Validation error"):
        parse_structured_output('{"value": "not_an_int"}', Strict)


# ---------------------------------------------------------------------------
# T10.09 — build_repair_messages produces correct structure
# ---------------------------------------------------------------------------


def test_build_repair_messages_structure():
    # T10.09
    original = [Message.user("give me JSON"), Message.assistant("not json")]
    msgs = build_repair_messages(
        original_messages=original,
        invalid_content="not json",
        schema=_City,
        error="JSON parse error",
    )
    assert len(msgs) == 3  # original 2 + 1 repair prompt
    last = msgs[-1]
    assert last.role == "user"
    assert "JSON parse error" in last.content
    assert "not json" in last.content


# ---------------------------------------------------------------------------
# T10.10 — normalize_json_schema closes objects
# ---------------------------------------------------------------------------


def test_normalize_json_schema_closes_objects():
    # T10.10
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    result = normalize_json_schema(schema)
    assert result["additionalProperties"] is False


# ---------------------------------------------------------------------------
# T10.11 — normalize_json_schema does not overwrite existing additionalProperties
# ---------------------------------------------------------------------------


def test_normalize_json_schema_preserves_existing():
    # T10.11
    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }
    result = normalize_json_schema(schema)
    assert result["additionalProperties"] is True


# ---------------------------------------------------------------------------
# apply_structured_validation helper
# ---------------------------------------------------------------------------


def test_apply_structured_validation_success():
    """apply_structured_validation sets parsed/validated on success."""
    from lazybridge.core.structured import apply_structured_validation
    from lazybridge.core.types import CompletionResponse, UsageStats

    resp = CompletionResponse(content='{"a": 1}', usage=UsageStats())
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    apply_structured_validation(resp, '{"a": 1}', schema)
    assert resp.validated is True
    assert resp.parsed == {"a": 1}
    assert resp.validation_error is None


def test_apply_structured_validation_failure():
    """apply_structured_validation sets validation_error on failure."""
    from lazybridge.core.structured import apply_structured_validation
    from lazybridge.core.types import CompletionResponse, UsageStats

    resp = CompletionResponse(content="not json", usage=UsageStats())
    schema = {"type": "object", "properties": {}}
    apply_structured_validation(resp, "not json", schema)
    assert resp.validated is False
    assert resp.validation_error is not None
    assert resp.parsed is None
