"""Unit tests for LazyTool — no API calls, no provider required."""
from __future__ import annotations

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from lazybridge.lazy_tool import (
    LazyTool,
    NormalizedToolSet,
    ToolArgumentValidationError,
)
from lazybridge.core.types import ToolDefinition


# ── Helper functions used as tools ───────────────────────────────────────────

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello {name}"


def scale(x: float) -> float:
    """Scale a value."""
    return x * 2.0


def no_annotations(a, b):
    return a + b


def with_kwargs(a: int, **kwargs) -> int:
    return a


async def async_double(x: int) -> int:
    return x * 2


# ── T1.01 — from_function: default name and description ──────────────────────

def test_from_function_default_name_desc():
    tool = LazyTool.from_function(add)
    assert tool.name == "add"
    assert tool.description == "Add two numbers."


# ── T1.02 — from_function: explicit name/description override ────────────────

def test_from_function_override_name_desc():
    tool = LazyTool.from_function(add, name="my_add", description="Custom desc")
    assert tool.name == "my_add"
    assert tool.description == "Custom desc"


# ── T1.03 — run: int coerced to float ────────────────────────────────────────

def test_run_coercion():
    tool = LazyTool.from_function(scale)
    result = tool.run({"x": 3})      # int 3 → float 3.0
    assert result == 6.0


# ── T1.04 — run: missing required argument → ToolArgumentValidationError ─────

def test_run_missing_argument():
    tool = LazyTool.from_function(add)
    with pytest.raises(ToolArgumentValidationError):
        tool.run({"a": 1})           # b is missing


# ── T1.05 — run: wrong type that can't be coerced → ToolArgumentValidationError

def test_run_bad_type():
    tool = LazyTool.from_function(add)
    with pytest.raises(ToolArgumentValidationError):
        tool.run({"a": "not_an_int", "b": 2})


# ── T1.06 — from_agent: fixed {"task": str} schema, _delegate set ─────────────

def test_from_agent_schema():
    agent = MagicMock()
    agent.name = "my_agent"
    agent.description = "Does stuff"
    tool = LazyTool.from_agent(agent)
    defn = tool.definition()
    assert defn.parameters["properties"] == {"task": {"type": "string"}}
    assert tool._delegate is not None
    assert tool._delegate.agent is agent


# ── T1.07 — specialize: delegate tool with name override updates _compiled ────

def test_specialize_delegate_updates_compiled():
    agent = MagicMock()
    agent.name = "agent"
    agent.description = "desc"
    tool = LazyTool.from_agent(agent)
    specialized = tool.specialize(name="new_name", description="new_desc")
    assert specialized.name == "new_name"
    assert specialized.description == "new_desc"
    assert specialized._compiled is not None
    assert specialized._compiled.name == "new_name"
    assert specialized._compiled.description == "new_desc"
    # Parameters unchanged
    assert specialized._compiled.parameters == tool._compiled.parameters


# ── T1.08 — specialize: function tool resets _compiled ───────────────────────

def test_specialize_function_resets_compiled():
    tool = LazyTool.from_function(add)
    _ = tool.definition()            # cache it
    assert tool._compiled is not None
    specialized = tool.specialize(name="add_v2")
    assert specialized.name == "add_v2"
    assert specialized._compiled is None   # cleared so it rebuilds


# ── T1.09 — definition: result is cached after first call ─────────────────────

def test_definition_cached():
    tool = LazyTool.from_function(add)
    defn1 = tool.definition()
    defn2 = tool.definition()
    assert defn1 is defn2


# ── T1.10 — arun: sync function wrapped as awaitable ─────────────────────────

async def test_arun_sync_function():
    tool = LazyTool.from_function(add)
    result = await tool.arun({"a": 3, "b": 4})
    assert result == 7


# ── T1.11 — arun: async function awaited correctly ───────────────────────────

async def test_arun_async_function():
    tool = LazyTool.from_function(async_double)
    result = await tool.arun({"x": 5})
    assert result == 10


# ── T1.12 — NormalizedToolSet: duplicate name raises ValueError ───────────────

def test_normalized_toolset_duplicate_name():
    tool_a = LazyTool.from_function(add)
    tool_b = LazyTool.from_function(add)   # same name "add"
    with pytest.raises(ValueError, match="Duplicate tool name"):
        NormalizedToolSet.from_list([tool_a, tool_b])


# ── T1.13 — NormalizedToolSet: mix of LazyTool, ToolDefinition, dict ─────────

def test_normalized_toolset_mixed_types():
    lazy = LazyTool.from_function(greet)
    defn = ToolDefinition(
        name="explicit_defn",
        description="A ToolDefinition",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    raw = {
        "name": "raw_dict",
        "description": "A dict tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    ts = NormalizedToolSet.from_list([lazy, defn, raw])
    assert len(ts.definitions) == 3
    assert len(ts.bridges) == 1          # only LazyTool
    assert "greet" in ts.registry
    assert "explicit_defn" not in ts.registry
    assert "raw_dict" not in ts.registry


# ═══════════════════════════════════════════════════════════════════════════════
# T3.x — save() / load()
# ═══════════════════════════════════════════════════════════════════════════════

# Helper defined at module level so inspect.getsource() works
def multiply(x: int, y: int) -> int:
    """Multiply two integers."""
    return x * y


# ── T3.1 — round-trip: save + load produces a working tool ───────────────────

def test_save_load_roundtrip(tmp_path):
    tool = LazyTool.from_function(multiply, guidance="Use for multiplication.")
    dest = str(tmp_path / "multiply_tool.py")
    tool.save(dest)
    loaded = LazyTool.load(dest)
    assert loaded.name == "multiply"
    assert loaded.description == "Multiply two integers."
    assert loaded.guidance == "Use for multiplication."
    assert loaded.run({"x": 3, "y": 4}) == 12


# ── T3.2 — generated file contains function source ───────────────────────────

def test_save_contains_function_source(tmp_path):
    tool = LazyTool.from_function(multiply)
    dest = str(tmp_path / "tool.py")
    tool.save(dest)
    content = Path(dest).read_text()
    assert "def multiply" in content
    assert "return x * y" in content


# ── T3.3 — generated file contains lazybridge import and sentinel ─────────────

def test_save_contains_sentinel_and_import(tmp_path):
    tool = LazyTool.from_function(multiply)
    dest = str(tmp_path / "tool.py")
    tool.save(dest)
    content = Path(dest).read_text()
    assert content.startswith("# LAZYBRIDGE_GENERATED_TOOL v1")
    assert "from lazybridge import LazyTool" in content


# ── T3.4 — save on lambda raises ValueError ───────────────────────────────────

def test_save_lambda_raises(tmp_path):
    tool = LazyTool.from_function(lambda x: x, name="identity", description="identity")
    with pytest.raises(ValueError, match="lambda"):
        tool.save(str(tmp_path / "lambda_tool.py"))


# ── T3.5 — save + load for from_agent tool ────────────────────────────────────

def test_save_load_agent_tool(tmp_path):
    agent = MagicMock()
    agent.name = "my_agent"
    agent.description = "Does research"
    agent.system = "You are a researcher."
    # Simulate a real executor with a provider
    mock_provider = MagicMock()
    mock_provider.__class__.__name__ = "AnthropicProvider"
    agent._executor = MagicMock()
    agent._executor.provider = mock_provider
    agent._executor.model = "claude-sonnet-4-6"
    agent.as_tool = LazyTool.from_agent.__func__  # not needed; test file content

    tool = LazyTool.from_agent(agent, name="research_tool", description="Research task")
    dest = str(tmp_path / "agent_tool.py")
    tool.save(dest)
    content = Path(dest).read_text()
    assert content.startswith("# LAZYBRIDGE_GENERATED_TOOL v1")
    assert "LazyAgent" in content
    assert "anthropic" in content
    assert "API keys are not serialized" in content
    # api_key must not appear as a kwarg assignment (only allowed in comments)
    import re
    assert not re.search(r'api_key\s*=\s*["\']', content)


# ── T3.6 — load on file without sentinel raises ValueError ───────────────────

def test_load_missing_sentinel_raises(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("tool = 'not a LazyTool'\n")
    with pytest.raises(ValueError, match="sentinel"):
        LazyTool.load(str(bad))


# ── T3.7 — save creates parent directories automatically ──────────────────────

def test_save_creates_parent_dirs(tmp_path):
    dest = str(tmp_path / "a" / "b" / "c" / "tool.py")
    tool = LazyTool.from_function(multiply)
    tool.save(dest)
    assert Path(dest).exists()


# ── T3.8 — api_key not in agent-tool file ────────────────────────────────────

def test_save_agent_no_api_key(tmp_path):
    agent = MagicMock()
    agent.name = "secure_agent"
    agent.description = None
    agent.system = None
    mock_provider = MagicMock()
    mock_provider.__class__.__name__ = "OpenAIProvider"
    agent._executor = MagicMock()
    agent._executor.provider = mock_provider
    agent._executor.model = "gpt-5.4"

    tool = LazyTool.from_agent(agent, name="t", description="d")
    dest = str(tmp_path / "secure.py")
    tool.save(dest)
    content = Path(dest).read_text()
    assert "sk-" not in content
    # "api_key" should not appear as a kwarg being set
    import re
    assert not re.search(r'api_key\s*=\s*["\']', content)


# ── T3.9 — load on file without sentinel (explicit check) ────────────────────

def test_load_no_sentinel_error_message(tmp_path):
    f = tmp_path / "tool.py"
    f.write_text("# some other header\ntool = None\n")
    with pytest.raises(ValueError, match="sentinel"):
        LazyTool.load(str(f))


# ── T3.10 — load with '..' in path raises ValueError ─────────────────────────

def test_load_dotdot_raises():
    with pytest.raises(ValueError, match="'\\.\\.'"):
        LazyTool.load("auto_tool/../etc/passwd.py")


# ── T3.11 — save with '..' in path raises ValueError ─────────────────────────

def test_save_dotdot_raises():
    tool = LazyTool.from_function(multiply)
    with pytest.raises(ValueError, match="'\\.\\.'"):
        tool.save("auto_tool/../../etc/evil.py")


# =============================================================================
# PR-B — LazyTool.parallel(), LazyTool.chain(), cloning, discriminators
# N1–N22
# =============================================================================

from lazybridge.pipeline_builders import (
    _is_agent_instance,
    _is_delegate_tool,
    _clone_for_invocation,
    _resolve_participant,
)
from lazybridge.core.types import CompletionResponse, UsageStats


def _mock_agent(name: str, response_content: str) -> MagicMock:
    """Minimal mock LazyAgent-like object with all relevant attributes set."""
    agent = MagicMock()
    agent.name = name
    agent._last_output = None
    agent._last_response = None
    agent.output_schema = None
    agent.tools = []
    agent.native_tools = []
    agent.session = None
    agent._log = None
    agent._executor = MagicMock()
    resp = CompletionResponse(content=response_content, usage=UsageStats())

    # Side effects return resp without touching _last_output.
    # The chain/parallel builders read resp.content from the return value;
    # real LazyAgent.chat() sets _last_output internally, but the mock doesn't
    # need to so that N13/N14 can verify the original is not mutated.
    agent.chat = MagicMock(return_value=resp)
    agent.achat = AsyncMock(return_value=resp)
    agent.json = MagicMock(return_value={"mocked": True})
    agent.ajson = AsyncMock(return_value={"mocked": True})
    return agent


# ── N1 — parallel() returns LazyTool with _is_pipeline_tool=True ─────────────

def test_n1_parallel_returns_lazytool():
    a = _mock_agent("a", "out_a")
    b = _mock_agent("b", "out_b")
    t = LazyTool.parallel(a, b, name="par", description="parallel test")
    assert isinstance(t, LazyTool)
    assert t._is_pipeline_tool is True


# ── N2 — chain() returns LazyTool with _is_pipeline_tool=True ────────────────

def test_n2_chain_returns_lazytool():
    a = _mock_agent("a", "out_a")
    t = LazyTool.chain(a, name="ch", description="chain test")
    assert isinstance(t, LazyTool)
    assert t._is_pipeline_tool is True


# ── N3 — parallel() with no participants raises ValueError ───────────────────

def test_n3_parallel_no_participants():
    with pytest.raises(ValueError):
        LazyTool.parallel(name="t", description="t")


# ── N4 — chain() with no participants raises ValueError ──────────────────────

def test_n4_chain_no_participants():
    with pytest.raises(ValueError):
        LazyTool.chain(name="t", description="t")


# ── N5 — save() on pipeline tool raises ValueError ───────────────────────────

def test_n5_save_pipeline_tool_raises(tmp_path):
    a = _mock_agent("a", "out_a")
    t = LazyTool.parallel(a, name="par", description="t")
    with pytest.raises(ValueError, match="pipeline tool"):
        t.save(str(tmp_path / "out.py"))
    t2 = LazyTool.chain(a, name="ch", description="t")
    with pytest.raises(ValueError, match="cannot be serialized"):
        t2.save(str(tmp_path / "out2.py"))


# ── N6 — _clone_for_invocation resets _last_output ───────────────────────────

def test_n6_clone_resets_last_output():
    agent = _mock_agent("a", "something")
    agent._last_output = "previous result"
    clone = _clone_for_invocation(agent)
    assert clone._last_output is None
    assert agent._last_output == "previous result"  # original unchanged


# ── N7 — _clone_for_invocation shares _executor ──────────────────────────────

def test_n7_clone_shares_executor():
    agent = _mock_agent("a", "x")
    clone = _clone_for_invocation(agent)
    assert clone._executor is agent._executor


# ── N8 — _clone_for_invocation has new id ────────────────────────────────────

def test_n8_clone_new_id():
    import uuid
    agent = _mock_agent("a", "x")
    agent.id = str(uuid.uuid4())
    clone = _clone_for_invocation(agent)
    assert clone.id != agent.id


# ── N9 — _clone_for_invocation has session=None ──────────────────────────────

def test_n9_clone_session_none():
    agent = _mock_agent("a", "x")
    agent.session = MagicMock()  # agent had a session
    clone = _clone_for_invocation(agent)
    assert clone.session is None


# ── N10 — _clone_for_invocation gets _log when original has session ──────────

def test_n10_clone_log_when_session():
    agent = _mock_agent("a", "x")
    mock_session = MagicMock()
    mock_log = MagicMock()
    mock_session.events.agent_log.return_value = mock_log
    agent.session = mock_session
    agent.name = "a"
    clone = _clone_for_invocation(agent)
    assert clone._log is mock_log
    mock_session.events.agent_log.assert_called_once_with(clone.id, "a")


# ── N11 — _clone_for_invocation gets _log=None when no session ───────────────

def test_n11_clone_log_none_no_session():
    agent = _mock_agent("a", "x")
    agent.session = None
    clone = _clone_for_invocation(agent)
    assert clone._log is None


# ── N12 — _clone_delegate_tool_for_invocation clones inner agent ─────────────

def test_n12_clone_delegate_tool():
    from lazybridge.lazy_tool import _clone_delegate_tool_for_invocation
    agent = _mock_agent("delegate", "response")
    agent.id = "original-id"
    dt = LazyTool.from_agent(agent, name="dt", description="delegate tool")
    dt_clone = _clone_delegate_tool_for_invocation(dt)
    assert dt_clone._delegate.agent is not dt._delegate.agent


# ── N13 — original agent._last_output is None after chain run ────────────────

def test_n13_original_last_output_unchanged_after_chain():
    agent = _mock_agent("worker", "chain output")
    agent._last_output = None
    t = LazyTool.chain(agent, name="pipe", description="t")
    result = t.run({"task": "do something"})
    assert result == "chain output"
    # Original agent was NOT run — clone ran
    assert agent._last_output is None


# ── N14 — LazyContext.from_agent(original) returns "" after chain run ─────────

def test_n14_lazy_context_from_original_empty_after_chain():
    from lazybridge.lazy_context import LazyContext
    agent = _mock_agent("worker", "chain output")
    t = LazyTool.chain(agent, name="pipe", description="t")
    t.run({"task": "go"})
    ctx = LazyContext.from_agent(agent)
    assert ctx() == ""   # original _last_output is None → empty


# ── N15 — specialize() preserves _is_pipeline_tool ───────────────────────────

def test_n15_specialize_preserves_pipeline_flag():
    a = _mock_agent("a", "out")
    t = LazyTool.parallel(a, name="par", description="t")
    t2 = t.specialize(name="par_v2")
    assert t2._is_pipeline_tool is True


# ── N16 — session= with conflicting session raises ValueError ─────────────────

def test_n16_session_cross_session_conflict():
    a = _mock_agent("a", "out")
    session_a = MagicMock()
    session_b = MagicMock()
    a.session = session_a   # agent bound to session_a
    with pytest.raises(ValueError, match="different session"):
        LazyTool.parallel(a, name="t", description="t", session=session_b)


# ── N17 — concurrent parallel calls: original _last_output unchanged ─────────

def test_n17_parallel_original_last_output_unchanged():
    a = _mock_agent("a", "result_a")
    b = _mock_agent("b", "result_b")
    t = LazyTool.parallel(a, b, name="par", description="t")
    t.run({"task": "go"})
    # Originals were NOT called — clones ran
    assert a._last_output is None
    assert b._last_output is None


# ── N18 — _is_agent_instance returns True for mock agent ─────────────────────

def test_n18_is_agent_instance_true():
    a = _mock_agent("a", "x")
    assert _is_agent_instance(a) is True


# ── N19 — _is_agent_instance returns False for LazyTool ──────────────────────

def test_n19_is_agent_instance_false_for_tool():
    def f(task: str) -> str: return task
    t = LazyTool.from_function(f, name="t", description="t")
    assert _is_agent_instance(t) is False


# ── N20 — _is_delegate_tool returns True for from_agent() tool ───────────────

def test_n20_is_delegate_tool_true():
    agent = _mock_agent("a", "x")
    dt = LazyTool.from_agent(agent, name="dt", description="t")
    assert _is_delegate_tool(dt) is True


# ── N21 — _is_delegate_tool returns False for from_function() tool ───────────

def test_n21_is_delegate_tool_false():
    def f(task: str) -> str: return task
    t = LazyTool.from_function(f, name="t", description="t")
    assert _is_delegate_tool(t) is False


# ── N22 — _resolve_participant raises TypeError for unknown type ──────────────

def test_n22_resolve_participant_unknown_type_raises():
    class Unknown:
        pass
    with pytest.raises(TypeError, match="LazyAgent or LazyTool"):
        _resolve_participant(Unknown())


def test_n23_parallel_raises_cross_session_for_delegate_tool():
    """N23: _validate_session_compatibility raises ValueError for a delegate tool
    whose inner agent is bound to a different session."""
    from unittest.mock import MagicMock

    sess_a = MagicMock()
    sess_b = MagicMock()

    inner_agent = MagicMock()
    inner_agent._last_output = None   # _is_agent_instance checks hasattr(_last_output)
    inner_agent.session = sess_b
    inner_agent.name = "foreign_agent"

    class _FakeDelegate:
        agent = inner_agent

    class _FakeDelegateTool:
        _delegate = _FakeDelegate()
        def run(self, task): return "ok"
        def arun(self, task): return "ok"

    participant = _FakeDelegateTool()

    with pytest.raises(ValueError, match="different session"):
        LazyTool.parallel(participant, name="p", description="d", session=sess_a)
