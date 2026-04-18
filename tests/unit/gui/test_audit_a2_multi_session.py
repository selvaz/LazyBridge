"""Regression tests for audit L5 — pipeline panel captures events from all sessions."""

from __future__ import annotations

import time
from types import SimpleNamespace

from lazybridge.gui.pipeline import PipelinePanel
from lazybridge.lazy_session import LazySession


def _fake_agent(name: str, session) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"id-{name}",
        name=name,
        session=session,
        _provider_name="anthropic",
        _model_name="m",
    )


def _pipeline_tool(participants, emit_fn):
    tool = SimpleNamespace()
    tool.name = "multi_sess_pipe"
    tool.description = "test"
    tool._is_pipeline_tool = True
    tool._pipeline = SimpleNamespace(
        mode="chain",
        participants=tuple(participants),
        combiner=None,
        concurrency_limit=None,
        step_timeout=None,
        guidance=None,
    )
    tool.run = emit_fn
    return tool


def test_pipeline_panel_captures_events_from_multiple_sessions():
    sess_a = LazySession()
    sess_b = LazySession()
    a1 = _fake_agent("alpha", sess_a)
    a2 = _fake_agent("beta", sess_b)

    def _run(args):
        # alpha's events go to sess_a; beta's to sess_b.
        sess_a.events.log("agent_start", agent_id=a1.id, agent_name=a1.name)
        sess_a.events.log("agent_finish", agent_id=a1.id, agent_name=a1.name,
                          method="chat", stop_reason="end_turn", n_steps=1)
        sess_b.events.log("agent_start", agent_id=a2.id, agent_name=a2.name)
        sess_b.events.log("agent_finish", agent_id=a2.id, agent_name=a2.name,
                          method="chat", stop_reason="end_turn", n_steps=1)
        return "multi-session done"

    tool = _pipeline_tool([a1, a2], _run)
    panel = PipelinePanel(tool)
    panel.handle_action("run", {"task": "go"})
    assert panel._run_done.wait(timeout=2.0)

    lr = panel.render_state()["last_run"]
    assert lr["status"] == "done"
    assert lr["captured_from_session"] is True
    assert len(lr["events"]) == 4, lr["events"]
    assert {e["agent_name"] for e in lr["events"]} == {"alpha", "beta"}


def test_pipeline_panel_deduplicates_sessions():
    """If two participants share one session, the exporter must be
    attached exactly once — not once per participant."""
    sess = LazySession()
    a1 = _fake_agent("alpha", sess)
    a2 = _fake_agent("beta", sess)

    # Count exporter additions.
    original_add = sess.events.add_exporter
    added: list = []

    def _tracking_add(ex):
        added.append(ex)
        return original_add(ex)

    sess.events.add_exporter = _tracking_add  # type: ignore[assignment]

    def _run(args):
        return "ok"

    tool = _pipeline_tool([a1, a2], _run)
    panel = PipelinePanel(tool)
    panel.handle_action("run", {"task": "hi"})
    assert panel._run_done.wait(timeout=2.0)
    assert len(added) == 1, (
        f"exporter attached {len(added)} times for one session — dedup failed"
    )
