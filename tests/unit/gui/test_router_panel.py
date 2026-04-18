"""Tests for RouterPanel."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from lazybridge.core.types import CompletionResponse, UsageStats
from lazybridge.gui.router import RouterPanel
from lazybridge.lazy_agent import LazyAgent
from lazybridge.lazy_router import LazyRouter


class AnthropicProvider:
    def get_default_max_tokens(self) -> int:  # pragma: no cover
        return 4096


def _bare_agent(name: str):
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent.__new__(LazyAgent)
    mock_exec = MagicMock()
    mock_exec.provider = AnthropicProvider()
    mock_exec.model = "m"
    agent._executor = mock_exec
    agent.id = str(uuid.uuid4())
    agent.name = name
    agent.description = None
    agent.system = ""
    agent.context = None
    agent.tools = []
    agent.native_tools = []
    agent.output_schema = None
    agent._last_output = None
    agent._last_response = None
    agent.session = None
    agent._log = None
    return agent


def _router():
    writer = _bare_agent("writer")
    reviewer = _bare_agent("reviewer")
    return (
        LazyRouter(
            condition=lambda v: "writer" if v == "ship" else "reviewer",
            routes={"writer": writer, "reviewer": reviewer},
            name="gate",
            default="reviewer",
        ),
        writer,
        reviewer,
    )


def test_router_panel_render_state():
    router, writer, reviewer = _router()
    state = RouterPanel(router).render_state()
    assert state["name"] == "gate"
    assert state["default"] == "reviewer"
    keys = [r["key"] for r in state["routes"]]
    assert keys == ["writer", "reviewer"]
    names = [r["agent_name"] for r in state["routes"]]
    assert names == ["writer", "reviewer"]
    assert all(r["panel_id"].startswith("agent-") for r in state["routes"])
    assert "condition" in state


def test_router_panel_route_picks_matched_key():
    router, writer, _ = _router()
    out = RouterPanel(router).handle_action("route", {"value": "ship"})
    assert out["matched_key"] == "writer"
    assert out["agent_name"] == "writer"
    assert out["panel_id"] == f"agent-{writer.id}"


def test_router_panel_route_and_run_invokes_agent():
    router, writer, reviewer = _router()
    resp = CompletionResponse(
        content="the writer says hi",
        usage=UsageStats(input_tokens=1, output_tokens=2),
    )
    writer.chat = MagicMock(return_value=resp)
    out = RouterPanel(router).handle_action("route_and_run", {"value": "ship", "prompt": "write something"})
    writer.chat.assert_called_once_with("write something")
    assert out["content"] == "the writer says hi"
    assert out["agent_name"] == "writer"
    assert out["usage"]["input_tokens"] == 1


def test_router_panel_route_and_run_rejects_empty_prompt():
    router, *_ = _router()
    with pytest.raises(ValueError):
        RouterPanel(router).handle_action("route_and_run", {"value": "x", "prompt": ""})


def test_router_panel_route_unknown_key_with_default_uses_default():
    router, _, reviewer = _router()
    out = RouterPanel(router).handle_action("route", {"value": "anything-else"})
    # default="reviewer" → routes to reviewer
    assert out["agent_name"] == "reviewer"
    assert out["matched_key"] == "reviewer"


def test_router_gui_registers_panel():
    from lazybridge.gui import close_server, get_server, install_gui_methods
    from lazybridge.gui._global import _reset_for_tests

    _reset_for_tests()
    install_gui_methods()
    try:
        router, *_ = _router()
        url = router.gui(open_browser=False)
        assert "#panel=router-" in url
        assert get_server().get(f"router-{router.id}") is not None
    finally:
        close_server()
