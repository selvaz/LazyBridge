"""Tests for lazybridge.gui.__init__ monkey-patching + end-to-end flow."""

from __future__ import annotations

import json
import urllib.request

import pytest

from lazybridge.gui import (
    AgentPanel,
    GuiServer,
    Panel,
    close_server,
    get_server,
    install_gui_methods,
    is_running,
    uninstall_gui_methods,
)
from lazybridge.gui._global import _reset_for_tests
from lazybridge.lazy_agent import LazyAgent
from lazybridge.lazy_session import LazySession
from lazybridge.lazy_tool import LazyTool


@pytest.fixture(autouse=True)
def _fresh_server():
    _reset_for_tests()
    install_gui_methods()
    yield
    close_server()


def test_install_attaches_gui_method_to_each_class():
    assert callable(getattr(LazyAgent, "gui", None))
    assert callable(getattr(LazyTool, "gui", None))
    assert callable(getattr(LazySession, "gui", None))


def test_install_is_idempotent():
    first = LazyAgent.gui
    install_gui_methods()
    assert LazyAgent.gui is first


def test_uninstall_removes_methods_then_reinstall_works():
    uninstall_gui_methods()
    assert not hasattr(LazyAgent, "gui")
    install_gui_methods()
    assert callable(LazyAgent.gui)


def test_tool_gui_registers_and_returns_url():
    def ping() -> str:
        """Ping."""
        return "pong"

    tool = LazyTool.from_function(ping)
    url = tool.gui(open_browser=False)
    assert is_running()
    assert "#panel=tool-ping" in url
    # And the panel appears in the API listing.
    base = url.split("?")[0]
    token = url.split("?t=")[1].split("#")[0]
    data = json.loads(urllib.request.urlopen(f"{base}api/panels?t={token}").read())
    names = [p["id"] for p in data["panels"]]
    assert "tool-ping" in names


def test_tool_gui_end_to_end_invoke_over_http():
    def echo(x: str) -> str:
        """Echo."""
        return f"echo: {x}"

    tool = LazyTool.from_function(echo)
    url = tool.gui(open_browser=False)
    base = url.split("?")[0]
    token = url.split("?t=")[1].split("#")[0]
    panel_id = url.split("#panel=")[1]

    body = json.dumps({"action": "invoke", "args": {"args": {"x": "hi"}}}).encode()
    req = urllib.request.Request(
        f"{base}api/panel/{panel_id}/action?t={token}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    payload = json.loads(urllib.request.urlopen(req, timeout=2).read())
    assert payload == {"result": "echo: hi"}


def test_session_gui_registers_child_agents_and_tools():
    import uuid
    from unittest.mock import MagicMock, patch

    def tool_a(x: str) -> str:
        """A."""
        return x

    def _bare_agent(name: str) -> LazyAgent:
        with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
            agent = LazyAgent.__new__(LazyAgent)
        mock_exec = MagicMock()
        mock_exec.provider.get_default_max_tokens.return_value = 4096
        mock_exec.provider.name = "anthropic"
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

    sess = LazySession()
    a1 = _bare_agent("first")
    a2 = _bare_agent("second")
    a1.tools = [LazyTool.from_function(tool_a)]
    sess._agents = [a1, a2]
    a1.session = a2.session = sess

    url = sess.gui(open_browser=False)
    assert f"#panel=session-{sess.id}" in url

    server = get_server()
    ids = {p.id for p in server.panels()}
    assert f"session-{sess.id}" in ids
    assert f"agent-{a1.id}" in ids
    assert f"agent-{a2.id}" in ids
    assert "tool-tool_a" in ids


def test_exports_public_api():
    # Spot-check that the public names round-trip as expected.
    assert Panel.__name__ == "Panel"
    assert AgentPanel.__name__ == "AgentPanel"
    assert isinstance(get_server(open_browser=False), GuiServer)
