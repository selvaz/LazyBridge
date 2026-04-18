"""Tests for the AgentPanel ``export_python`` action."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from lazybridge.core.types import NativeTool
from lazybridge.gui.agent import AgentPanel
from lazybridge.lazy_agent import LazyAgent


class AnthropicProvider:
    def get_default_max_tokens(self) -> int:  # pragma: no cover
        return 4096


def _bare_agent():
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent.__new__(LazyAgent)
    mock_exec = MagicMock()
    mock_exec.provider = AnthropicProvider()
    mock_exec.model = "claude-sonnet-4-6"
    agent._executor = mock_exec
    agent.id = str(uuid.uuid4())
    agent.name = "researcher"
    agent.description = None
    agent.system = "You are a terse research assistant."
    agent.context = None
    agent.tools = []
    agent.native_tools = []
    agent.output_schema = None
    agent._last_output = None
    agent._last_response = None
    agent.session = None
    agent._log = None
    return agent


def test_export_python_basic_agent():
    panel = AgentPanel(_bare_agent())
    out = panel.handle_action("export_python", {})
    snippet = out["snippet"]
    assert "from lazybridge import LazyAgent" in snippet
    assert "agent = LazyAgent(" in snippet
    assert "'anthropic'" in snippet
    assert "name='researcher'" in snippet
    assert "model='claude-sonnet-4-6'" in snippet
    assert "You are a terse research assistant." in snippet
    # Must compile as valid Python.
    compile(snippet, "<test>", "exec")


def test_export_python_omits_empty_system():
    agent = _bare_agent()
    agent.system = ""
    panel = AgentPanel(agent)
    snippet = panel.handle_action("export_python", {})["snippet"]
    assert "system=" not in snippet


def test_export_python_with_native_tools_emits_import_and_enum():
    agent = _bare_agent()
    first = next(iter(NativeTool))
    agent.native_tools = [first]
    panel = AgentPanel(agent)
    snippet = panel.handle_action("export_python", {})["snippet"]
    assert "from lazybridge.core.types import NativeTool" in snippet
    assert f"NativeTool({str(first.value)!r})" in snippet


def test_export_python_with_tools_adds_placeholder_comment():
    from lazybridge.lazy_tool import LazyTool

    def search(q: str) -> str:
        """Search."""
        return q

    agent = _bare_agent()
    agent.tools = [LazyTool.from_function(search)]
    panel = AgentPanel(agent)
    snippet = panel.handle_action("export_python", {})["snippet"]
    assert "# tools=" in snippet
    assert "'search'" in snippet


def test_export_python_roundtrips_via_exec(tmp_path):
    """The snippet should parse and assign an `agent` variable in a fresh ns."""
    agent = _bare_agent()
    agent.system = 'hello\n"world"'  # exercise quoting
    snippet = AgentPanel(agent).handle_action("export_python", {})["snippet"]
    ns: dict = {}
    # Patch LazyAgent to a lightweight stub so exec doesn't try to spin up
    # a real provider.
    stub = MagicMock()
    import lazybridge as _lb

    original = _lb.LazyAgent
    _lb.LazyAgent = stub
    try:
        exec(snippet, ns)
    finally:
        _lb.LazyAgent = original
    stub.assert_called_once()
    _, kw = stub.call_args
    assert kw["name"] == "researcher"
    assert 'hello\n"world"' in kw["system"]
