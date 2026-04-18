"""Regression tests for Wave 5 of the deep audit (M10, L9, L2, L11)."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# L2 — run_async emits a DEBUG log when it hits the thread-offload path
# ---------------------------------------------------------------------------


def test_run_async_logs_debug_when_offloading(caplog):
    """Hitting the thread-offload path should leave a DEBUG breadcrumb so
    power users can spot the overhead and prefer ``await coro`` directly."""
    import asyncio as _asyncio

    from lazybridge.lazy_run import run_async

    async def _noop():
        return 42

    async def _driver():
        # Calling run_async from inside a running loop exercises the
        # offload path.
        return run_async(_noop())

    with caplog.at_level(logging.DEBUG, logger="lazybridge.lazy_run"):
        result = _asyncio.run(_driver())
    assert result == 42
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("offloading to a thread pool" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# L11 — llm_judge errors now raise JudgeError; EvalReport.errors counts them
# ---------------------------------------------------------------------------


def test_llm_judge_crash_surfaces_as_error_not_fail():
    from lazybridge.evals import EvalCase, EvalSuite, JudgeError, llm_judge

    # Judge agent whose .text() always raises.
    crashing_judge = MagicMock()
    crashing_judge.text.side_effect = RuntimeError("judge exploded")
    check = llm_judge(crashing_judge, criteria="always pass")

    import pytest
    with pytest.raises(JudgeError):
        check("anything")

    # And the suite-level behaviour: a case whose check raises goes into
    # EvalResult.error AND bumps EvalReport.errors.
    test_agent = MagicMock()
    test_agent.text = MagicMock(return_value="the output")
    suite = EvalSuite(cases=[EvalCase(prompt="hi", check=check, name="c1")])
    report = suite.run(test_agent)
    assert report.total == 1
    assert report.passed == 0
    assert report.failed == 1
    assert report.errors == 1  # <<< new field
    assert report.results[0].error is not None
    assert "judge exploded" in report.results[0].error


# ---------------------------------------------------------------------------
# L9 — PipelinePanel exposes a _run_done Event for Event-based waiting
# ---------------------------------------------------------------------------


def test_pipeline_panel_run_done_event_is_threading_event():
    from types import SimpleNamespace

    from lazybridge.gui.pipeline import PipelinePanel

    agent = SimpleNamespace(id="a", name="x", session=None, _provider_name="p", _model_name="m")
    tool = SimpleNamespace(
        name="p",
        description="",
        _is_pipeline_tool=True,
        _pipeline=SimpleNamespace(
            mode="chain", participants=(agent,), combiner=None,
            concurrency_limit=None, step_timeout=None, guidance=None,
        ),
        run=lambda args: "ok",
    )
    panel = PipelinePanel(tool)
    # Is set-like from threading?
    assert isinstance(panel._run_done, threading.Event)
    # Starts cleared.
    assert not panel._run_done.is_set()
    # After a quick run it gets set.
    panel.handle_action("run", {"task": "t"})
    assert panel._run_done.wait(timeout=2.0)
    assert panel.render_state()["last_run"]["status"] == "done"


# ---------------------------------------------------------------------------
# M10 — provider adapters are no longer omitted from coverage
# ---------------------------------------------------------------------------


def test_coverage_config_includes_providers():
    """pyproject.toml must not silently exclude the provider adapters
    from coverage.run.omit — audit M10."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    omit = (pyproject.get("tool", {})
                       .get("coverage", {})
                       .get("run", {})
                       .get("omit", []))
    bad = [p for p in omit if "core/providers/" in p or "core\\providers\\" in p]
    assert not bad, f"coverage.run.omit still hides provider adapters: {bad}"
