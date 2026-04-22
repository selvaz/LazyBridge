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
from typing import Any, get_origin

from lazybridge.core.types import Message

_logger = logging.getLogger(__name__)


def validate_payload_against_output_type(payload: Any, output_type: Any) -> Any:
    """Validate / coerce ``payload`` to match ``output_type``.

    Supports three cases that ``LLMEngine._loop``'s naive
    ``isinstance(output_type, type)`` check missed:

    * plain Pydantic model classes (existing behaviour) — passed through
      if already an instance, otherwise validated via ``model_validate``;
    * generic collection types like ``list[MyModel]`` /
      ``dict[str, MyModel]`` — validated via
      ``pydantic.TypeAdapter(output_type).validate_python(...)``;
    * ``str`` output type — returned as-is (no validation).

    Raises on mismatch so ``Agent._validate_and_retry`` can feed the
    error back to the model as retry context.
    """
    if output_type is str or output_type is Any:
        return payload

    # Lazy import — keeps this module light when Pydantic is not present.
    from pydantic import BaseModel, TypeAdapter, ValidationError

    # Bare Pydantic model class.
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        if isinstance(payload, output_type):
            return payload
        if isinstance(payload, dict):
            return output_type.model_validate(payload)
        if isinstance(payload, str):
            return output_type.model_validate_json(payload)
        # Last resort: let Pydantic try to coerce.
        return output_type.model_validate(payload)

    # Generic type (``list[Model]``, ``dict[str, Model]``, ``Optional[X]``,
    # unions).  TypeAdapter handles everything Pydantic understands.
    origin = get_origin(output_type)
    if origin is not None or isinstance(output_type, type):
        try:
            adapter = TypeAdapter(output_type)
        except Exception:
            # Couldn't build an adapter (weirdly shaped output_type) —
            # return the payload verbatim rather than raise here; the
            # caller's validator can still reject it.
            return payload
        if isinstance(payload, str):
            return adapter.validate_json(payload)
        return adapter.validate_python(payload)

    # Fallthrough: unknown output shape, leave payload alone.
    return payload


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
    """Return a provider-safe JSON schema with closed objects by default.

    Recursively sets ``additionalProperties: false`` on every object node,
    including those inside ``$defs`` / ``definitions`` (Pydantic nested models),
    ``properties`` values, ``items``, ``additionalProperties`` (when a schema
    dict, not a bool), ``prefixItems`` (JSON Schema 2020-12 tuple arrays), and
    ``anyOf`` / ``allOf`` / ``oneOf`` sub-schemas.

    ``$ref`` nodes are not resolved inline — the definitions they point to in
    ``$defs`` are already recursively normalized when that key is encountered.
    """
    if not isinstance(schema, dict):
        return schema

    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            normalized[key] = {
                prop_name: normalize_json_schema(prop_schema) if isinstance(prop_schema, dict) else prop_schema
                for prop_name, prop_schema in value.items()
            }
        elif key in ("$defs", "definitions") and isinstance(value, dict):
            # Pydantic v2 puts nested model schemas here — must also be closed.
            normalized[key] = {
                def_name: normalize_json_schema(def_schema) if isinstance(def_schema, dict) else def_schema
                for def_name, def_schema in value.items()
            }
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            normalized[key] = normalize_json_schema(value)
        elif key == "prefixItems" and isinstance(value, list):
            normalized[key] = [
                normalize_json_schema(sub) if isinstance(sub, dict) else sub
                for sub in value
            ]
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
            normalized[key] = [
                normalize_json_schema(sub) if isinstance(sub, dict) else sub
                for sub in value
            ]
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
    """Validate ``data`` against a JSON Schema dict.

    If the optional ``jsonschema`` library is installed, delegate to it
    for full Draft-2020 coverage — ``pattern``, ``minLength``, ``maximum``,
    ``format``, etc. are all enforced.  Otherwise we fall back to a
    minimal subset validator (audit M9).

    Fallback-validator supported keywords: ``type``, ``required``,
    ``enum``, ``additionalProperties``, ``properties`` (recursive),
    ``items`` (array elements).  All others are silently ignored.
    """
    try:
        import jsonschema as _jsonschema  # type: ignore
    except Exception:
        _jsonschema = None

    if _jsonschema is not None:
        try:
            _jsonschema.validate(instance=data, schema=schema)
            return None
        except _jsonschema.ValidationError as exc:
            # Mirror the "human" messages the fallback validator produces.
            path = ".".join(str(p) for p in exc.absolute_path)
            return f"field '{path}': {exc.message}" if path else exc.message
        except _jsonschema.SchemaError as exc:
            # Malformed schema — fall through to the subset validator so
            # the caller still gets a best-effort check.
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "jsonschema rejected schema; falling back to subset validator: %s",
                exc,
            )

    return _validate_schema_subset(data, schema)


def _validate_schema_subset(data: Any, schema: dict[str, Any]) -> str | None:
    """Minimal JSON Schema subset validator (see :func:`_validate_schema`)."""
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
        if len(lines) == 1:
            # Single-line: ```json {"a":1}``` or ```{"a":1}```
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        else:
            # Multi-line: drop first (```json) and last (```) lines
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
    original_messages: list[Message | dict],
    invalid_content: str,
    schema: type | dict[str, Any],
    error: str,
) -> list[Message | dict]:
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
