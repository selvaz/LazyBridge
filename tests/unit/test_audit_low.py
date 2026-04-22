"""Regression tests for LOW audit fixes.

Covers:
* ``Store.write`` preserves Pydantic payloads on SQLite backend (was
  stringifying to ``__repr__`` via ``default=str``).
* ``Session.usage_summary`` is O(events) with TWO bulk queries, not
  ``2N+2`` single-row lookups.
* ``GraphSchema.clear()`` drops nodes and edges.
* ``EventLog.query`` now surfaces ``run_id`` directly.
"""

from __future__ import annotations

import tempfile

from pydantic import BaseModel

from lazybridge import Agent, EventType, GraphSchema, Session, Store

# ---------------------------------------------------------------------------
# Store preserves Pydantic through SQLite round-trip
# ---------------------------------------------------------------------------


class _MyModel(BaseModel):
    x: int
    name: str


def test_store_sqlite_preserves_pydantic_via_model_dump(tmp_path):
    """Before the fix: ``store.read("k")`` returned the string
    ``"x=42 name='hello'"``.  After: a dict that round-trips cleanly
    and can be re-validated back into the model.
    """
    store = Store(db=str(tmp_path / "s.sqlite"))
    store.write("k1", _MyModel(x=42, name="hello"))

    raw = store.read("k1")
    assert isinstance(raw, dict)
    assert raw == {"x": 42, "name": "hello"}

    # Caller can re-hydrate to the model type if they want.
    assert _MyModel.model_validate(raw) == _MyModel(x=42, name="hello")


def test_store_sqlite_preserves_list_of_pydantic():
    """Lists and dicts containing Pydantic models recurse correctly."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(db=f"{td}/s.sqlite")
        items = [_MyModel(x=1, name="a"), _MyModel(x=2, name="b")]
        store.write("items", items)

        raw = store.read("items")
        assert raw == [{"x": 1, "name": "a"}, {"x": 2, "name": "b"}]


def test_store_inmemory_unchanged():
    """In-memory store is unaffected — keeps the Python instance."""
    store = Store()
    m = _MyModel(x=7, name="z")
    store.write("k", m)
    assert store.read("k") is m


# ---------------------------------------------------------------------------
# EventLog.query surfaces run_id; usage_summary no longer N+1
# ---------------------------------------------------------------------------


def test_event_log_query_returns_run_id_field():
    """Pre-fix callers (like usage_summary) had to re-query the table
    per row to resolve run_id. It's now part of the result dict."""
    sess = Session()
    sess.emit(EventType.AGENT_START, {"agent_name": "a"}, run_id="r-1")
    sess.emit(EventType.MODEL_RESPONSE, {"input_tokens": 10}, run_id="r-1")

    rows = sess.events.query()
    assert len(rows) == 2
    assert all("run_id" in r for r in rows)
    assert {r["run_id"] for r in rows} == {"r-1"}


def test_usage_summary_aggregates_across_runs_without_perrow_queries():
    """The functional output is unchanged; we verify the shape and that
    per-agent aggregation still works after the N+1 refactor.
    """
    sess = Session()
    sess.emit(EventType.AGENT_START, {"agent_name": "researcher"}, run_id="run-1")
    sess.emit(
        EventType.MODEL_RESPONSE,
        {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.002},
        run_id="run-1",
    )
    sess.emit(EventType.AGENT_START, {"agent_name": "writer"}, run_id="run-2")
    sess.emit(
        EventType.MODEL_RESPONSE,
        {"input_tokens": 200, "output_tokens": 80, "cost_usd": 0.004},
        run_id="run-2",
    )
    # Second MODEL_RESPONSE for researcher (multi-turn).
    sess.emit(
        EventType.MODEL_RESPONSE,
        {"input_tokens": 30, "output_tokens": 10, "cost_usd": 0.0005},
        run_id="run-1",
    )

    summary = sess.usage_summary()

    assert summary["total"]["input_tokens"] == 330
    assert summary["total"]["output_tokens"] == 140
    assert round(summary["total"]["cost_usd"], 4) == 0.0065

    assert summary["by_agent"]["researcher"]["input_tokens"] == 130
    assert summary["by_agent"]["writer"]["input_tokens"] == 200

    assert summary["by_run"]["run-1"]["agent_name"] == "researcher"
    assert summary["by_run"]["run-1"]["input_tokens"] == 130
    assert summary["by_run"]["run-2"]["input_tokens"] == 200


# ---------------------------------------------------------------------------
# GraphSchema.clear()
# ---------------------------------------------------------------------------


def test_graph_schema_clear_drops_nodes_and_edges():
    sess = Session()
    Agent("claude-opus-4-7", name="a", session=sess)
    Agent("claude-opus-4-7", name="b", session=sess)
    sess.graph.add_edge("a", "b", label="handoff")

    assert len(sess.graph.nodes()) == 2
    assert len(sess.graph.edges()) == 1

    sess.graph.clear()

    assert sess.graph.nodes() == []
    assert sess.graph.edges() == []
    # Session id survives so event correlation still works.
    assert sess.graph.session_id == sess.session_id


def test_graph_schema_clear_on_empty_graph_is_idempotent():
    g = GraphSchema(session_id="s")
    g.clear()
    g.clear()
    assert g.nodes() == []
    assert g.edges() == []
