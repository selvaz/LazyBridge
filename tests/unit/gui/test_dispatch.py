"""Tests for open_gui() type-dispatched helper."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from lazybridge.gui import close_server, get_server, open_gui
from lazybridge.gui._global import _reset_for_tests
from lazybridge.gui.agent import AgentPanel
from lazybridge.gui.pipeline import PipelinePanel
from lazybridge.gui.router import RouterPanel
from lazybridge.gui.session import SessionPanel
from lazybridge.gui.store import StorePanel
from lazybridge.gui.tool import ToolPanel
from lazybridge.lazy_agent import LazyAgent
from lazybridge.lazy_router import LazyRouter
from lazybridge.lazy_session import LazySession
from lazybridge.lazy_store import LazyStore
from lazybridge.lazy_tool import LazyTool


class AnthropicProvider:
    def get_default_max_tokens(self) -> int:  # pragma: no cover
        return 4096


def _bare_agent(name: str = "a"):
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


def _echo(x: str) -> str:
    """Echo."""
    return x


@pytest.fixture(autouse=True)
def _fresh():
    _reset_for_tests()
    yield
    close_server()


def test_open_gui_dispatches_lazy_agent():
    agent = _bare_agent()
    url = open_gui(agent, open_browser=False)
    panel = get_server().get(f"agent-{agent.id}")
    assert isinstance(panel, AgentPanel)
    assert f"#panel=agent-{agent.id}" in url


def test_open_gui_dispatches_lazy_tool():
    tool = LazyTool.from_function(_echo)
    url = open_gui(tool, open_browser=False)
    assert isinstance(get_server().get("tool-_echo"), ToolPanel)
    assert "#panel=tool-_echo" in url


def test_open_gui_dispatches_pipeline_tool():
    chain = LazyTool.chain(
        LazyTool.from_function(_echo),
        LazyTool.from_function(_echo),
        name="pipeD",
        description="x",
    )
    open_gui(chain, open_browser=False)
    assert isinstance(get_server().get("pipeline-pipeD"), PipelinePanel)


def test_open_gui_dispatches_router():
    router = LazyRouter(
        condition=lambda v: "x",
        routes={"x": _bare_agent("r")},
        name="rtr",
    )
    open_gui(router, open_browser=False)
    assert isinstance(get_server().get(f"router-{router.id}"), RouterPanel)


def test_open_gui_dispatches_store():
    store = LazyStore()
    url = open_gui(store, open_browser=False)
    panel_id = url.split("#panel=")[1]
    assert isinstance(get_server().get(panel_id), StorePanel)


def test_open_gui_dispatches_memory():
    from lazybridge.gui.memory import MemoryPanel
    from lazybridge.memory import Memory

    mem = Memory()
    url = open_gui(mem, open_browser=False)
    panel_id = url.split("#panel=")[1]
    assert isinstance(get_server().get(panel_id), MemoryPanel)


def test_open_gui_dispatches_session_and_pre_registers_children():
    sess = LazySession()
    a = _bare_agent("x")
    a.tools = [LazyTool.from_function(_echo)]
    sess._agents = [a]
    open_gui(sess, open_browser=False)
    srv = get_server()
    assert isinstance(srv.get(f"session-{sess.id}"), SessionPanel)
    assert isinstance(srv.get(f"agent-{a.id}"), AgentPanel)
    assert isinstance(srv.get("tool-_echo"), ToolPanel)
    # session store pre-registered
    assert any(p.kind == "store" for p in srv.panels())


def test_open_gui_rejects_unsupported_type():
    with pytest.raises(TypeError):
        open_gui(object(), open_browser=False)


def test_open_gui_returns_same_url_as_monkey_patched_gui():
    """Parity check: open_gui(obj) ≡ obj.gui() after install_gui_methods()."""
    from lazybridge.gui import install_gui_methods

    install_gui_methods()
    agent = _bare_agent("parity")
    url_method = agent.gui(open_browser=False)
    url_helper = open_gui(agent, open_browser=False)
    assert url_method == url_helper
