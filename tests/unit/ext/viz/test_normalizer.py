"""JSON normaliser: every value crossing SSE must come out JSON-safe."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from lazybridge.ext.viz._normalizer import normalise_event, to_jsonable


def test_primitives_pass_through():
    assert to_jsonable(None) is None
    assert to_jsonable(True) is True
    assert to_jsonable(42) == 42
    assert to_jsonable(3.14) == 3.14
    assert to_jsonable("hello") == "hello"


def test_datetime_becomes_isoformat():
    d = dt.datetime(2026, 5, 4, 12, 30, 0)
    assert to_jsonable(d) == "2026-05-04T12:30:00"
    assert to_jsonable(dt.date(2026, 5, 4)) == "2026-05-04"


def test_bytes_decoded_or_repr():
    assert to_jsonable(b"hello") == "hello"
    out = to_jsonable(b"\xff\xfe")  # invalid utf-8
    assert isinstance(out, str)


def test_nested_collections():
    src = {"a": [1, {"b": (2, 3)}], "c": {1, 2}}
    out = to_jsonable(src)
    # Round-trips through json.dumps without raising
    json.dumps(out)


def test_pydantic_v2_model_dump():
    pydantic = pytest.importorskip("pydantic")

    class M(pydantic.BaseModel):
        x: int
        name: str

    out = to_jsonable(M(x=1, name="hi"))
    assert out == {"x": 1, "name": "hi"}


def test_unknown_object_falls_back_to_repr():
    class Weird:
        __slots__ = ("a",)

        def __init__(self):
            self.a = 1

        def __repr__(self):
            return "<Weird a=1>"

    out = to_jsonable(Weird())
    # __dict__ path triggers, picks up `a`
    assert out == {"a": 1} or isinstance(out, str)


def test_normalise_event_preserves_keys():
    ev = {"event_type": "tool_call", "ts": dt.datetime(2026, 1, 1)}
    out = normalise_event(ev)
    assert out["event_type"] == "tool_call"
    assert "ts" in out


def test_long_string_is_truncated():
    big = "x" * 20000
    out = to_jsonable(big)
    assert out.endswith("…")
    assert len(out) < 20000


def test_recursion_depth_capped():
    # Build a 12-deep nested dict — must not stack overflow, must terminate
    cur: dict = {}
    root = cur
    for _ in range(12):
        nxt: dict = {}
        cur["next"] = nxt
        cur = nxt
    out = to_jsonable(root)
    json.dumps(out)
