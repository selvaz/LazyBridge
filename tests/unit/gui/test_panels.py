"""AgentPanel / ToolPanel / SessionPanel behaviour.

Uses the same low-level LazyAgent construction trick as
``tests/unit/test_lazy_agent.py`` to avoid instantiating real providers.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from lazybridge.core.types import CompletionResponse, ToolCall, UsageStats
from lazybridge.gui.agent import AgentPanel
from lazybridge.gui.session import SessionPanel
from lazybridge.gui.tool import ToolPanel
from lazybridge.lazy_agent import LazyAgent
from lazybridge.lazy_tool import LazyTool


class _PydanticOut(BaseModel):
    value: int


def _make_it(n: int) -> _PydanticOut:
    """Make."""
    return _PydanticOut(value=n)


_PydanticPanel = ToolPanel(LazyTool.from_function(_make_it))


class AnthropicProvider:
    """Stub whose type name satisfies LazyAgent._provider_name."""

    def get_default_max_tokens(self) -> int:  # pragma: no cover - signature only
        return 4096


def _bare_agent(*, name="alpha", model="test-model", provider="anthropic"):
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent.__new__(LazyAgent)
    mock_exec = MagicMock()
    mock_exec.provider = AnthropicProvider()
    mock_exec.model = model
    agent._executor = mock_exec
    agent.id = str(uuid.uuid4())
    agent.name = name
    agent.description = None
    agent.system = "you are helpful"
    agent.context = None
    agent.tools = []
    agent.native_tools = []
    agent.output_schema = None
    agent._last_output = None
    agent._last_response = None
    agent.session = None
    agent._log = None
    return agent


# ---------------------------------------------------------------------------
# AgentPanel
# ---------------------------------------------------------------------------


def test_agent_panel_render_state_basic():
    agent = _bare_agent(name="r", model="m1")
    panel = AgentPanel(agent)
    state = panel.render_state()
    assert state["name"] == "r"
    assert state["model"] == "m1"
    assert state["system"] == "you are helpful"
    assert state["tools"] == []
    assert state["available_tools"] == []
    assert state["has_native_tools"] is False


def test_agent_panel_update_system_mutates_agent():
    agent = _bare_agent()
    panel = AgentPanel(agent)
    panel.handle_action("update_system", {"value": "new instructions"})
    assert agent.system == "new instructions"


def test_agent_panel_available_tools_uses_override():
    def foo(x: str) -> str:
        """Foo."""
        return x

    tool = LazyTool.from_function(foo)
    agent = _bare_agent()
    panel = AgentPanel(agent, available_tools=[tool])
    state = panel.render_state()
    assert [t["name"] for t in state["available_tools"]] == ["foo"]


def test_agent_panel_available_tools_from_session():
    """When no override is given, panel pulls tools from the session's agents."""
    def search(q: str) -> str:
        """Search."""
        return q

    def calc(x: int) -> int:
        """Calc."""
        return x

    search_tool = LazyTool.from_function(search)
    calc_tool = LazyTool.from_function(calc)

    a1 = _bare_agent(name="a1")
    a2 = _bare_agent(name="a2")
    a1.tools = [search_tool]
    a2.tools = [calc_tool]

    fake_session = SimpleNamespace(id="s", _agents=[a1, a2])
    a1.session = fake_session
    a2.session = fake_session

    panel = AgentPanel(a1)
    names = sorted(t["name"] for t in panel.render_state()["available_tools"])
    assert names == ["calc", "search"]


def test_agent_panel_toggle_tool_adds_from_scope():
    def greet(name: str) -> str:
        """Greet."""
        return f"hi {name}"

    greet_tool = LazyTool.from_function(greet)
    agent = _bare_agent()
    other = _bare_agent(name="other")
    other.tools = [greet_tool]
    fake_session = SimpleNamespace(id="s", _agents=[agent, other])
    agent.session = other.session = fake_session

    panel = AgentPanel(agent)
    res = panel.handle_action("toggle_tool", {"name": "greet", "enabled": True})
    assert [t["name"] for t in res["tools"]] == ["greet"]
    assert [t.name for t in agent.tools] == ["greet"]

    # Unchecking removes it again.
    res = panel.handle_action("toggle_tool", {"name": "greet", "enabled": False})
    assert res["tools"] == []
    assert agent.tools == []


def test_agent_panel_toggle_tool_out_of_scope_fails():
    agent = _bare_agent()
    panel = AgentPanel(agent)
    with pytest.raises(ValueError):
        panel.handle_action("toggle_tool", {"name": "nope", "enabled": True})


def test_agent_panel_test_mode_chat():
    agent = _bare_agent()
    resp = CompletionResponse(
        content="hello!",
        usage=UsageStats(input_tokens=3, output_tokens=2),
        model="m1",
        stop_reason="end_turn",
        tool_calls=[ToolCall(id="t1", name="fn", arguments={"a": 1})],
    )
    agent.chat = MagicMock(return_value=resp)
    panel = AgentPanel(agent)
    out = panel.handle_action("test", {"mode": "chat", "message": "hi"})
    agent.chat.assert_called_once_with("hi")
    assert out["content"] == "hello!"
    assert out["usage"] == {
        "input_tokens": 3, "output_tokens": 2, "thinking_tokens": 0, "cost_usd": None
    }
    assert out["tool_calls"] == [{"name": "fn", "arguments": {"a": 1}}]


def test_agent_panel_test_empty_message_rejected():
    panel = AgentPanel(_bare_agent())
    with pytest.raises(ValueError):
        panel.handle_action("test", {"mode": "chat", "message": "   "})


def test_agent_panel_test_invalid_mode():
    panel = AgentPanel(_bare_agent())
    with pytest.raises(ValueError):
        panel.handle_action("test", {"mode": "stream", "message": "hi"})


def test_agent_panel_test_mode_text_returns_content_only():
    agent = _bare_agent()
    agent.text = MagicMock(return_value="plain")
    panel = AgentPanel(agent)
    out = panel.handle_action("test", {"mode": "text", "message": "hi"})
    assert out == {"content": "plain"}


def test_agent_panel_test_mode_loop():
    agent = _bare_agent()
    resp = CompletionResponse(content="looped", usage=UsageStats())
    agent.loop = MagicMock(return_value=resp)
    panel = AgentPanel(agent)
    out = panel.handle_action("test", {"mode": "loop", "message": "hi"})
    agent.loop.assert_called_once_with("hi")
    assert out["content"] == "looped"


# ---------------------------------------------------------------------------
# ToolPanel
# ---------------------------------------------------------------------------


def test_tool_panel_render_state_exposes_schema():
    def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    panel = ToolPanel(LazyTool.from_function(add))
    state = panel.render_state()
    assert state["name"] == "add"
    assert state["description"].startswith("Add")
    params = state["parameters"]
    assert params["properties"]["x"]["type"] == "integer"
    assert params["properties"]["y"]["type"] == "integer"
    assert set(params["required"]) == {"x", "y"}
    assert state["is_pipeline_tool"] is False
    assert state["is_delegate"] is False


def test_tool_panel_invoke_runs_tool():
    def concat(a: str, b: str) -> str:
        """Concat."""
        return a + b

    panel = ToolPanel(LazyTool.from_function(concat))
    out = panel.handle_action("invoke", {"args": {"a": "foo", "b": "bar"}})
    assert out == {"result": "foobar"}


def test_tool_panel_invoke_rejects_non_dict_args():
    def noop() -> str:
        """No-op."""
        return "ok"

    panel = ToolPanel(LazyTool.from_function(noop))
    with pytest.raises(ValueError):
        panel.handle_action("invoke", {"args": "oops"})


def test_tool_panel_invoke_jsonifies_pydantic_result():
    out = _PydanticPanel.handle_action("invoke", {"args": {"n": 7}})
    assert out == {"result": {"value": 7}}


# ---------------------------------------------------------------------------
# SessionPanel
# ---------------------------------------------------------------------------


def test_session_panel_lists_agents_and_store_keys():
    agent = _bare_agent(name="r", provider="anthropic", model="claude-x")
    fake_store = SimpleNamespace(keys=lambda: ["foo", "bar"])
    sess = SimpleNamespace(id="s-1", tracking=None, _agents=[agent], store=fake_store)
    agent.session = sess
    state = SessionPanel(sess).render_state()
    assert state["id"] == "s-1"
    assert state["agents"] == [
        {"id": f"agent-{agent.id}", "name": "r", "provider": "anthropic", "model": "claude-x"}
    ]
    assert state["store_keys"] == ["foo", "bar"]
