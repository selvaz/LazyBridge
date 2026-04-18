"""Tests for PipelinePanel + the LazyTool.gui dispatch."""

from __future__ import annotations

from lazybridge.gui.pipeline import PipelinePanel, is_pipeline_tool
from lazybridge.gui.tool import ToolPanel
from lazybridge.lazy_tool import LazyTool


def _echo(x: str) -> str:
    """Echo."""
    return x


def _upper(x: str) -> str:
    """Upper."""
    return x.upper()


def test_is_pipeline_tool_detects_chain_and_parallel():
    plain = LazyTool.from_function(_echo)
    assert is_pipeline_tool(plain) is False

    chain = LazyTool.chain(
        plain, LazyTool.from_function(_upper),
        name="pipe", description="echo then upper",
    )
    assert is_pipeline_tool(chain) is True

    par = LazyTool.parallel(
        plain, LazyTool.from_function(_upper),
        name="par", description="parallel demo",
    )
    assert is_pipeline_tool(par) is True


def test_pipeline_panel_render_state_chain():
    chain = LazyTool.chain(
        LazyTool.from_function(_echo),
        LazyTool.from_function(_upper),
        name="pipe", description="echo then upper",
        step_timeout=5.0,
    )
    panel = PipelinePanel(chain)
    state = panel.render_state()
    assert state["name"] == "pipe"
    assert state["mode"] == "chain"
    assert state["step_timeout"] == 5.0
    assert [p["name"] for p in state["participants"]] == ["_echo", "_upper"]
    assert all(p["kind"] == "tool" for p in state["participants"])


def test_pipeline_panel_render_state_parallel_with_agents():
    # Mock two agent-shaped participants with the attrs PipelinePanel reads.
    class FakeAgent:
        def __init__(self, name: str) -> None:
            self.id = f"id-{name}"
            self.name = name
            self._provider_name = "anthropic"
            self._model_name = "claude-x"

    # Build a parallel tool directly from LazyTool.parallel so the config
    # is populated, then swap the participant tuple for our fakes.
    par = LazyTool.parallel(
        LazyTool.from_function(_echo),
        LazyTool.from_function(_upper),
        name="par", description="x",
        combiner="concat", concurrency_limit=2,
    )
    assert par._pipeline is not None
    par._pipeline.participants = (FakeAgent("a1"), FakeAgent("a2"))

    state = PipelinePanel(par).render_state()
    assert state["mode"] == "parallel"
    assert state["combiner"] == "concat"
    assert state["concurrency_limit"] == 2
    assert [(p["kind"], p["name"], p.get("panel_id")) for p in state["participants"]] == [
        ("agent", "a1", "agent-id-a1"),
        ("agent", "a2", "agent-id-a2"),
    ]


def test_pipeline_panel_run_action(monkeypatch):
    """Panel's 'run' action invokes tool.run({'task': ...}) and returns the result."""
    chain = LazyTool.chain(
        LazyTool.from_function(_echo),
        LazyTool.from_function(_upper),
        name="pipe", description="echo then upper",
    )
    # `LazyTool.chain` is built for agent-participants; function-tools have
    # their own `(x: str)` signature.  Here we only assert that the panel
    # forwards the task correctly — mock the underlying run() so we don't
    # exercise the whole chain resolver.
    calls = []

    def fake_run(payload):
        calls.append(payload)
        return "final result"

    monkeypatch.setattr(chain, "run", fake_run)
    panel = PipelinePanel(chain)
    out = panel.handle_action("run", {"task": "hi"})
    assert out == {"result": "final result"}
    assert calls == [{"task": "hi"}]


def test_pipeline_panel_run_rejects_empty_task():
    chain = LazyTool.chain(
        LazyTool.from_function(_echo), LazyTool.from_function(_upper),
        name="pipe", description="x",
    )
    panel = PipelinePanel(chain)
    import pytest
    with pytest.raises(ValueError):
        panel.handle_action("run", {"task": "   "})


def test_pipeline_panel_rejects_non_pipeline_tool():
    plain = LazyTool.from_function(_echo)
    import pytest
    with pytest.raises(ValueError):
        PipelinePanel(plain)


def test_tool_gui_picks_pipeline_panel_for_pipeline_tools():
    from lazybridge.gui import close_server, get_server, install_gui_methods
    from lazybridge.gui._global import _reset_for_tests

    _reset_for_tests()
    install_gui_methods()
    try:
        chain = LazyTool.chain(
            LazyTool.from_function(_echo), LazyTool.from_function(_upper),
            name="pipeX", description="x",
        )
        url = chain.gui(open_browser=False)
        assert "#panel=pipeline-pipeX" in url
        server = get_server()
        panel = server.get("pipeline-pipeX")
        assert isinstance(panel, PipelinePanel)

        # Plain tool still uses ToolPanel.
        plain = LazyTool.from_function(_echo)
        plain_url = plain.gui(open_browser=False)
        assert "#panel=tool-_echo" in plain_url
        plain_panel = server.get("tool-_echo")
        assert isinstance(plain_panel, ToolPanel)
    finally:
        close_server()
