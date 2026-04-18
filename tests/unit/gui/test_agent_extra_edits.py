"""Tests for the Phase-6 agent edits: model + native_tools."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

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
    mock_exec.model = "start-model"
    agent._executor = mock_exec
    agent.id = str(uuid.uuid4())
    agent.name = "a"
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


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_render_state_includes_model_and_available_native_tools():
    panel = AgentPanel(_bare_agent())
    state = panel.render_state()
    assert state["model"] == "start-model"
    # At least WEB_SEARCH should be listed.
    names = state["available_native_tools"]
    assert any("web" in n.lower() for n in names)


def test_update_model_mutates_executor():
    agent = _bare_agent()
    panel = AgentPanel(agent)
    out = panel.handle_action("update_model", {"value": "claude-sonnet-4-6"})
    assert out["model"] == "claude-sonnet-4-6"
    assert agent._executor.model == "claude-sonnet-4-6"
    assert agent._model_name == "claude-sonnet-4-6"


def test_update_model_rejects_empty():
    panel = AgentPanel(_bare_agent())
    with pytest.raises(ValueError):
        panel.handle_action("update_model", {"value": "  "})


# ---------------------------------------------------------------------------
# native_tools
# ---------------------------------------------------------------------------


def test_toggle_native_tool_enables_and_disables():
    agent = _bare_agent()
    panel = AgentPanel(agent)

    # Pick the first NativeTool member by value.
    first = next(iter(NativeTool))
    name = str(first.value)

    out = panel.handle_action("toggle_native_tool", {"name": name, "enabled": True})
    assert name in out["native_tools"]
    assert first in agent.native_tools

    out = panel.handle_action("toggle_native_tool", {"name": name, "enabled": False})
    assert name not in out["native_tools"]
    assert first not in agent.native_tools


def test_toggle_native_tool_rejects_unknown():
    panel = AgentPanel(_bare_agent())
    with pytest.raises(ValueError):
        panel.handle_action("toggle_native_tool", {"name": "no-such", "enabled": True})


def test_toggle_native_tool_is_idempotent():
    agent = _bare_agent()
    panel = AgentPanel(agent)
    first = next(iter(NativeTool))
    name = str(first.value)

    panel.handle_action("toggle_native_tool", {"name": name, "enabled": True})
    panel.handle_action("toggle_native_tool", {"name": name, "enabled": True})
    # Should still only appear once.
    assert agent.native_tools.count(first) == 1
