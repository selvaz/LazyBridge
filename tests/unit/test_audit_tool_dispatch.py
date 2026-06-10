"""Regression tests for core-audit fixes on tool dispatch and schema caching.

* Tool.run / Tool.run_sync now validate and coerce LLM-provided arguments
  against the function signature (the validation helper existed but was
  never wired into the dispatch path).
* ``**kwargs`` extras survive validation (they were silently dropped).
* ArtifactStore fingerprints include ``flatten_refs`` — a flattening and a
  non-flattening builder sharing one store used to exchange artifacts.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from lazybridge.core.tool_schema import (
    InMemoryArtifactStore,
    ToolArgumentValidationError,
    ToolSchemaBuilder,
    _validate_and_coerce_arguments,
)
from lazybridge.tools import Tool


# ---------------------------------------------------------------------------
# Dispatch-time argument validation
# ---------------------------------------------------------------------------


def _typed_tool(count: int, scale: float = 1.0) -> float:
    return count * scale


def test_run_sync_coerces_string_numbers():
    tool = Tool(_typed_tool)
    # LLMs frequently send numbers as strings — coercion must happen
    # before the function body runs.
    assert tool.run_sync(count="3", scale="2.5") == 7.5


def test_run_sync_rejects_bad_arguments_with_readable_error():
    tool = Tool(_typed_tool)
    with pytest.raises(ToolArgumentValidationError, match="count"):
        tool.run_sync(count="not-a-number")


def test_async_run_validates_too():
    tool = Tool(_typed_tool)

    async def _go():
        return await tool.run(count="4")

    assert asyncio.run(_go()) == 4.0


def test_validate_args_opt_out():
    tool = Tool(_typed_tool)
    tool.validate_args = False
    # Opt-out restores raw dispatch: the un-coerced string reaches the
    # function body ("3" * 2 == "33" instead of the validated 6.0).
    assert tool.run_sync(count="3", scale=2) == "33"


def test_kwargs_extras_survive_validation():
    """extra='allow' extras live in __pydantic_extra__ — they were dropped."""

    def kw_tool(a: int, **kwargs):
        return a, kwargs

    out = _validate_and_coerce_arguments(kw_tool, {"a": 1, "extra": "x", "n": 2})
    assert out == {"a": 1, "extra": "x", "n": 2}

    tool = Tool(kw_tool)
    assert tool.run_sync(a="1", extra="x") == (1, {"extra": "x"})


class _Point(BaseModel):
    x: int
    y: int


def _move(p: _Point) -> int:
    return p.x + p.y


def test_pydantic_model_params_stay_models():
    tool = Tool(_move)
    assert tool.run_sync(p={"x": 1, "y": 2}) == 3


# ---------------------------------------------------------------------------
# ArtifactStore fingerprint includes flatten_refs
# ---------------------------------------------------------------------------


class _Outer(BaseModel):
    inner: _Point  # nested model → schema carries $defs/$ref


def _nested_tool(p: _Outer) -> str:
    """Does something with a nested model."""
    return ""


def test_flatten_refs_builders_do_not_share_cache_entries():
    store = InMemoryArtifactStore()
    flat = ToolSchemaBuilder(store, flatten_refs=True).build_artifact(_nested_tool)
    raw = ToolSchemaBuilder(store, flatten_refs=False).build_artifact(_nested_tool)

    assert flat.fingerprint != raw.fingerprint
    assert raw.cache_hit is False
    # The non-flattening builder keeps its $refs intact; the flattening one
    # must resolve them even when $defs sits nested inside a property schema
    # (it was a silent no-op for that common case before the fix).
    import json

    assert "$ref" in json.dumps(raw.definition.parameters)
    flat_json = json.dumps(flat.definition.parameters)
    assert "$ref" not in flat_json
    assert "$defs" not in flat_json


def test_local_return_type_does_not_break_validation():
    """A return annotation referencing a locally-defined class raises
    NameError in get_type_hints; parameter validation must survive it
    (regression: Plan steps with such tools silently failed)."""

    class LocalOut:  # not importable from module globals
        pass

    def tool(task: str) -> LocalOut:
        assert isinstance(task, str)
        return LocalOut()

    t = Tool(tool)
    assert isinstance(t.run_sync(task="hello"), LocalOut)
    with pytest.raises(ToolArgumentValidationError):
        t.run_sync(task="x", unexpected=1)
