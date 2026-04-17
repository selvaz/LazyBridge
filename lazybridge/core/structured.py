"""Shared structured output parsing and validation.

All providers call parse_structured_output() instead of doing their own
JSON parsing + Pydantic validation. This ensures consistent behaviour:
- returns the parsed result on success
- raises StructuredOutputError on any parse or validation failure
- handles both raw JSON schema dicts and Pydantic model classes
"""

from __future__ import annotations

import json
import logging
from typing import Any

from lazybridge.core.types import Message

_logger = logging.getLogger(__name__)


class StructuredOutputError(ValueError):
    """Raised when the model output cannot be parsed or validated against the schema."""


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def normalize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a provider-safe JSON schema with closed objects by default."""
    if not isinstance(schema, dict):
        return schema

    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            normalized[key] = {
                prop_name: normalize_json_schema(prop_schema) if isinstance(prop_schema, dict) else prop_schema
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            normalized[key] = normalize_json_schema(value)
        else:
            normalized[key] = value

    if normalized.get("type") == "object" and "additionalProperties" not in normalized:
        normalized["additionalProperties"] = False

    return normalized


def _enum_match(data: Any, value: Any) -> bool:
    """Return True iff data equals value, treating bool and int as distinct types."""
    if isinstance(data, bool) or isinstance(value, bool):
        return type(data) is type(value) and data == value
    return data == value


def _validate_schema(data: Any, schema: dict[str, Any]) -> str | None:
    """Minimal JSON Schema subset validator. Returns an error string or None.

    Supported keywords: type, required, enum, additionalProperties,
    properties (recursive), items (array elements).
    """
    schema_type = schema.get("type")
    if schema_type:
        expected = _TYPE_MAP.get(schema_type)
        if expected:
            # bool is a subclass of int in Python, but JSON treats them as distinct types.
            # A boolean must never pass type: integer or type: number checks.
            if isinstance(data, bool) and schema_type in ("integer", "number"):
                return f"expected {schema_type}, got boolean"
            if not isinstance(data, expected):
                return f"expected {schema_type}, got {type(data).__name__}"

    enum_vals = schema.get("enum")
    if enum_vals is not None and not any(_enum_match(data, v) for v in enum_vals):
        return f"value not in enum {enum_vals!r}: got {data!r}"

    if isinstance(data, dict):
        required = schema.get("required", [])
        missing = [k for k in required if k not in data]
        if missing:
            return f"missing required fields: {missing}"

        if schema.get("additionalProperties") is False:
            allowed = set(schema.get("properties", {}).keys())
            extra = [k for k in data if k not in allowed]
            if extra:
                return f"additional properties not allowed: {extra}"

        for key, prop_schema in schema.get("properties", {}).items():
            if key in data and isinstance(prop_schema, dict):
                err = _validate_schema(data[key], prop_schema)
                if err:
                    return f"field '{key}': {err}"

    if isinstance(data, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(data):
                err = _validate_schema(item, items_schema)
                if err:
                    return f"item {i}: {err}"

    return None


def parse_structured_output(
    content: str,
    schema: type | dict[str, Any],
) -> Any:
    """Parse *content* as JSON and validate it against *schema*.

    Parameters
    ----------
    content:
        The raw text returned by the model.
    schema:
        Either a Pydantic model class or a raw JSON schema dict.

    Returns
    -------
    parsed_result
        A Pydantic model instance (if schema is a class) or a plain dict
        (if schema is a dict).

    Raises
    ------
    StructuredOutputError
        On any parse or validation failure.
    """
    # Step 1 — strip markdown code fences that some models insert
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()

    # Step 2 — JSON parse
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise StructuredOutputError(f"JSON parse error: {exc}") from exc

    # Step 3 — validate
    if isinstance(schema, dict):
        err = _validate_schema(data, schema)
        if err:
            raise StructuredOutputError(f"Validation error: {err}")
        return data
    else:
        # Pydantic model class
        try:
            return schema.model_validate(data)  # type: ignore[attr-defined]
        except Exception as exc:
            raise StructuredOutputError(f"Validation error: {exc}") from exc


def apply_structured_validation(
    resp: Any,
    content: str,
    schema: type | dict[str, Any],
) -> None:
    """Parse *content* and set ``parsed``/``validated``/``validation_error`` on *resp*.

    Centralised helper that replaces the identical try/except blocks duplicated
    across every provider.  Mutates *resp* in place.

    Parameters
    ----------
    resp:
        A ``CompletionResponse`` or ``StreamChunk`` instance.
    content:
        The accumulated text to parse (may differ from ``resp.content`` in
        streaming scenarios where text is collected separately).
    schema:
        Pydantic model class or raw JSON schema dict.
    """
    try:
        resp.parsed = parse_structured_output(content, schema)
        resp.validated = True
    except StructuredOutputError as exc:
        resp.validation_error = str(exc)
        resp.validated = False


def build_repair_messages(
    original_messages: list[Message],
    invalid_content: str,
    schema: type | dict[str, Any],
    error: str,
) -> list[Message]:
    """Build a message list that asks the model to fix its invalid output.

    Appends a user message containing:
    - the validation error
    - the expected schema (JSON schema dict or Pydantic model_json_schema)
    - the invalid output produced

    The caller should send this new list as the next request with
    ``max_retries=0`` to avoid infinite loops.
    """
    # Build a compact schema representation
    if isinstance(schema, dict):
        schema_repr = json.dumps(schema, indent=2)
    else:
        try:
            schema_repr = json.dumps(schema.model_json_schema(), indent=2)  # type: ignore[attr-defined]
        except Exception as exc:
            _logger.warning("model_json_schema() failed for %r, falling back to str(): %s", schema, exc)
            schema_repr = str(schema)

    repair_prompt = (
        "Your previous response could not be parsed or validated.\n\n"
        f"Error: {error}\n\n"
        f"Expected JSON schema:\n```json\n{schema_repr}\n```\n\n"
        f"Your invalid output:\n```\n{invalid_content}\n```\n\n"
        "Please respond with valid JSON only, matching the schema exactly."
    )

    return list(original_messages) + [Message.user(repair_prompt)]
