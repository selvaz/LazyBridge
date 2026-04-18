"""Tests for PipelinePanel live per-step progress capture.

Uses a fake-pipeline that advertises itself as a pipeline tool and
publishes synthetic ``AGENT_START`` / ``AGENT_FINISH`` events on a
session's :class:`EventLog` — so we exercise the capture pipeline without
needing real provider credentials.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lazybridge.gui.pipeline import PipelinePanel
from lazybridge.lazy_session import LazySession


def _fake_pipeline_tool(participants, session):
    """Build a LazyTool-shaped stub whose .run() emits synthetic events."""
    tool = SimpleNamespace()
    tool.name = "fake_pipe"
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

    def _run(args):
        # Emit AGENT_START / AGENT_FINISH for each participant.
        for p in participants:
            session.events.log(
                "agent_start",
                agent_id=p.id,
                agent_name=p.name,
                task=args.get("task", ""),
            )
            time.sleep(0.01)
            session.events.log(
                "agent_finish",
                agent_id=p.id,
                agent_name=p.name,
                method="chat",
                stop_reason="end_turn",
                n_steps=1,
            )
        return "combined output"

    tool.run = _run
    return tool


def _fake_agent(name: str, session):
    agent = SimpleNamespace(
        id=f"id-{name}",
        name=name,
        session=session,
        _provider_name="anthropic",
        _model_name="m",
    )
    return agent


def test_run_captures_session_events_and_finishes_done():
    sess = LazySession()
    a1 = _fake_agent("alpha", sess)
    a2 = _fake_agent("beta", sess)
    tool = _fake_pipeline_tool([a1, a2], sess)
    panel = PipelinePanel(tool)

    out = panel.handle_action("run", {"task": "go"})
    assert out == {"started": True, "run_id": out["run_id"]}

    # Wait for the background thread to finish — Event-based, not polled.
    assert panel._run_done.wait(timeout=2.0)

    state = panel.render_state()
    lr = state["last_run"]
    assert lr["status"] == "done"
    assert lr["result"] == "combined output"
    assert lr["captured_from_session"] is True
    # Two participants × 2 events each = 4 lifecycle events captured.
    assert len(lr["events"]) == 4
    event_types = [e["event_type"] for e in lr["events"]]
    assert event_types == ["agent_start", "agent_finish", "agent_start", "agent_finish"]
    assert {e["agent_name"] for e in lr["events"]} == {"alpha", "beta"}


def test_run_without_session_skips_capture_but_still_returns_result():
    tool = SimpleNamespace()
    tool.name = "no_sess_pipe"
    tool.description = "test"
    tool._is_pipeline_tool = True
    tool._pipeline = SimpleNamespace(
        mode="chain",
        participants=(SimpleNamespace(id="x", name="x", session=None, _provider_name="a", _model_name="m"),),
        combiner=None,
        concurrency_limit=None,
        step_timeout=None,
        guidance=None,
    )
    tool.run = MagicMock(return_value="only result")

    panel = PipelinePanel(tool)
    panel.handle_action("run", {"task": "hi"})
    # Wait for thread — Event-based, not polled.
    assert panel._run_done.wait(timeout=2.0)
    state = panel.render_state()
    lr = state["last_run"]
    assert lr["status"] == "done"
    assert lr["result"] == "only result"
    assert lr["captured_from_session"] is False
    assert lr["events"] == []
    tool.run.assert_called_once_with({"task": "hi"})


def test_run_propagates_exception_as_error_state():
    sess = LazySession()
    agent = _fake_agent("boomer", sess)
    tool = SimpleNamespace()
    tool.name = "boom_pipe"
    tool.description = "test"
    tool._is_pipeline_tool = True
    tool._pipeline = SimpleNamespace(
        mode="chain",
        participants=(agent,),
        combiner=None,
        concurrency_limit=None,
        step_timeout=None,
        guidance=None,
    )

    def _run(args):
        raise RuntimeError("kaboom")

    tool.run = _run

    panel = PipelinePanel(tool)
    panel.handle_action("run", {"task": "x"})
    assert panel._run_done.wait(timeout=2.0)
    lr = panel.render_state()["last_run"]
    assert lr["status"] == "error"
    assert "kaboom" in lr["error"]


def test_running_label_marker():
    """The sidebar label should include '· running' while a run is in flight."""
    sess = LazySession()
    agent = _fake_agent("slow", sess)
    tool = SimpleNamespace()
    tool.name = "slow_pipe"
    tool.description = "test"
    tool._is_pipeline_tool = True
    tool._pipeline = SimpleNamespace(
        mode="chain",
        participants=(agent,),
        combiner=None,
        concurrency_limit=None,
        step_timeout=None,
        guidance=None,
    )

    started = threading_event_bool = {"running": True}

    def _run(args):
        while threading_event_bool["running"]:
            time.sleep(0.005)
        return "done"

    tool.run = _run

    panel = PipelinePanel(tool)
    panel.handle_action("run", {"task": "go"})
    time.sleep(0.02)
    assert panel.label.endswith("· running")
    threading_event_bool["running"] = False
    assert panel._run_done.wait(timeout=2.0)


def test_second_run_rejected_while_first_running():
    sess = LazySession()
    agent = _fake_agent("a", sess)
    tool = SimpleNamespace()
    tool.name = "p"
    tool.description = "test"
    tool._is_pipeline_tool = True
    tool._pipeline = SimpleNamespace(
        mode="chain",
        participants=(agent,),
        combiner=None,
        concurrency_limit=None,
        step_timeout=None,
        guidance=None,
    )

    flag = {"running": True}
    def _run(args):
        while flag["running"]:
            time.sleep(0.005)
        return "ok"
    tool.run = _run

    panel = PipelinePanel(tool)
    panel.handle_action("run", {"task": "1"})
    time.sleep(0.02)
    with pytest.raises(ValueError):
        panel.handle_action("run", {"task": "2"})
    flag["running"] = False


def test_clear_run_drops_last_run():
    sess = LazySession()
    agent = _fake_agent("a", sess)
    tool = _fake_pipeline_tool([agent], sess)
    panel = PipelinePanel(tool)
    panel.handle_action("run", {"task": "x"})
    assert panel._run_done.wait(timeout=2.0)
    assert "last_run" in panel.render_state()
    panel.handle_action("clear_run", {})
    assert "last_run" not in panel.render_state()
