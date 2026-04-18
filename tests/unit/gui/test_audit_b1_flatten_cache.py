"""Regression tests for audit L1 — cache flattened $defs in tool_schema."""

from __future__ import annotations

import copy

import pytest

from lazybridge.core import tool_schema as ts


@pytest.fixture(autouse=True)
def _reset_cache():
    ts._flatten_cache_clear()
    yield
    ts._flatten_cache_clear()


def _recursive_schema() -> dict:
    """A schema that references the same $def several times — exercises
    the deepcopy-per-inline cost pattern the cache is meant to fix."""
    return {
        "type": "object",
        "properties": {
            "first":  {"$ref": "#/$defs/Node"},
            "second": {"$ref": "#/$defs/Node"},
            "nested": {
                "type": "object",
                "properties": {"inner": {"$ref": "#/$defs/Node"}},
            },
        },
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "child": {"$ref": "#/$defs/Leaf"},
                },
            },
            "Leaf": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }


def test_flatten_refs_caches_repeat_calls():
    s = _recursive_schema()
    first = ts._flatten_refs(s)
    second = ts._flatten_refs(s)
    # Same value, different object identity (cache returns a copy).
    assert first == second
    assert first is not second, "cache must return a copy, not the stored value"
    size, hits = ts._flatten_cache_stats()
    assert size == 1
    assert hits == 1, f"expected 1 cache hit after second call, got {hits}"


def test_flatten_refs_cache_value_is_correct():
    """The cached flattened schema must not contain $defs any more."""
    out = ts._flatten_refs(_recursive_schema())
    assert "$defs" not in out
    # The two Node references should both be fully inlined.
    props = out["properties"]
    assert "properties" in props["first"]
    assert "properties" in props["second"]
    assert props["first"] is not props["second"]  # deepcopy isolation


def test_flatten_refs_fifo_eviction_bounded():
    """Exceeding the cap evicts the oldest entries (FIFO)."""
    # Shrink the cap for the duration of this test so we don't have to
    # allocate 130+ distinct schemas.
    original = ts._FLATTEN_CACHE_MAX
    ts._FLATTEN_CACHE_MAX = 3  # type: ignore[assignment]
    try:
        for i in range(5):
            ts._flatten_refs({
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/N"}},
                "$defs": {"N": {"type": "integer", "const": i}},
            })
        size, _ = ts._flatten_cache_stats()
        assert size == 3, f"cache grew past cap: {size}"
    finally:
        ts._FLATTEN_CACHE_MAX = original  # type: ignore[assignment]


def test_flatten_refs_noop_when_no_defs():
    """Schemas without $defs still pass through unchanged, no cache use."""
    s = {"type": "object", "properties": {"x": {"type": "integer"}}}
    out = ts._flatten_refs(s)
    assert out is s  # original reference returned
    size, _ = ts._flatten_cache_stats()
    assert size == 0


def test_flatten_refs_mutation_of_result_does_not_poison_cache():
    """Because we deepcopy on get AND on put, a caller mutating the
    returned schema must not affect future flatten_refs() callers."""
    s = _recursive_schema()
    out = ts._flatten_refs(s)
    # Mutate the result aggressively.
    out["properties"]["first"]["POISONED"] = True
    # Next call must return a clean copy.
    out2 = ts._flatten_refs(s)
    assert "POISONED" not in out2["properties"]["first"]
