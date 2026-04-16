"""Unit tests for tool schema validation and building."""

from __future__ import annotations

import pytest

from lazybridge.core.tool_schema import (
    ToolArgumentValidationError,
    _validate_and_coerce_arguments,
)
from lazybridge.lazy_tool import _params_to_schema

# ── Helper functions ──────────────────────────────────────────────────────────


def typed_func(a: int, b: float) -> float:
    return a + b


def str_func(name: str, greeting: str = "Hello") -> str:
    return f"{greeting} {name}"


def no_annotations(x, y):
    return x + y


def with_kwargs(a: int, **kwargs) -> int:
    return a


def required_and_optional(x: int, y: int = 0) -> int:
    return x + y


# ── T5.01 — int coerced to float ─────────────────────────────────────────────


def test_coerce_int_to_float():
    result = _validate_and_coerce_arguments(typed_func, {"a": 3, "b": 2})
    # b=2 (int) should be coerced to 2.0 (float)
    assert result["a"] == 3
    assert isinstance(result["b"], float)
    assert result["b"] == 2.0


# ── T5.02 — missing required field → ToolArgumentValidationError ─────────────


def test_missing_required_field():
    with pytest.raises(ToolArgumentValidationError, match="typed_func"):
        _validate_and_coerce_arguments(typed_func, {"a": 1})  # b missing


def test_missing_all_required():
    with pytest.raises(ToolArgumentValidationError):
        _validate_and_coerce_arguments(typed_func, {})


# ── T5.03 — function without annotations → pass-through ──────────────────────


def test_no_annotations_passthrough():
    args = {"x": "hello", "y": [1, 2, 3]}
    result = _validate_and_coerce_arguments(no_annotations, args)
    assert result == args


# ── T5.04 — **kwargs in signature → extra fields accepted ────────────────────


def test_kwargs_accepts_extra_fields():
    result = _validate_and_coerce_arguments(with_kwargs, {"a": 5, "extra": "ignored"})
    assert result["a"] == 5


# ── T5.05 — _params_to_schema: correct JSON Schema output ────────────────────


def test_params_to_schema_basic():
    schema = _params_to_schema({"name": str, "count": int, "ratio": float})
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["count"] == {"type": "integer"}
    assert schema["properties"]["ratio"] == {"type": "number"}
    assert set(schema["required"]) == {"name", "count", "ratio"}


def test_params_to_schema_dict_passthrough():
    """Dict values are used as-is in the schema."""
    custom = {"description": "a city", "type": "string", "enum": ["Rome", "Paris"]}
    schema = _params_to_schema({"city": custom})
    assert schema["properties"]["city"] == custom


def test_params_to_schema_empty():
    schema = _params_to_schema({})
    assert schema["properties"] == {}
    assert schema["required"] == []


# ── T5.06 — InMemoryArtifactStore: thread-safe concurrent put/get ─────────────
#
# Audit finding: InMemoryArtifactStore had no lock, making concurrent put/get
# unsafe under free-threaded Python and theoretically unsafe under the GIL.


def test_artifact_store_thread_safe():
    import threading

    from lazybridge.core.tool_schema import (
        InMemoryArtifactStore,
        ToolCompileArtifact,
        ToolSchemaMode,
        ToolSourceStatus,
    )
    from lazybridge.core.types import ToolDefinition

    store = InMemoryArtifactStore()
    errors: list[Exception] = []

    def _make_artifact(i: int) -> ToolCompileArtifact:
        defn = ToolDefinition(
            name=f"tool_{i}",
            description=f"desc {i}",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        return ToolCompileArtifact(
            fingerprint=f"fp{i:06d}",
            compiler_version="1",
            prompt_version="1",
            mode=ToolSchemaMode.SIGNATURE,
            source_status=ToolSourceStatus.BASELINE_ONLY,
            definition=defn,
            baseline_definition=None,
            llm_enriched_fields=frozenset(),
            warnings=(),
        )

    def worker(i: int) -> None:
        try:
            art = _make_artifact(i)
            store.put(art)
            store.get(art.fingerprint)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(store) == 50
