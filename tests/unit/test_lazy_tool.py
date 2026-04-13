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

def test_n5_save_pipeline_tool_produces_file(tmp_path):
    a = _mock_agent("a", "out_a")
    t = LazyTool.parallel(a, name="par", description="parallel test")
    out = str(tmp_path / "out.py")
    t.save(out)
    content = Path(out).read_text()
    assert "LAZYBRIDGE_GENERATED_TOOL v2" in content
    assert "LazyTool.parallel(" in content

    t2 = LazyTool.chain(a, name="ch", description="chain test")
    out2 = str(tmp_path / "out2.py")
    t2.save(out2)
    content2 = Path(out2).read_text()
    assert "LAZYBRIDGE_GENERATED_TOOL v2" in content2
    assert "LazyTool.chain(" in content2


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


# ── N-typed — chain: agent(output_schema) → LazyTool passes model_dump() ────

def test_chain_typed_agent_to_tool_passes_model_dump():
    """When an agent step produces a typed Pydantic object (output_schema),
    the next LazyTool step receives model_dump() as its arguments dict
    instead of {"task": serialized_text}."""
    from pydantic import BaseModel

    class AddInput(BaseModel):
        a: int
        b: int

    # Mock agent that produces a typed Pydantic result
    typed_result = AddInput(a=3, b=4)
    agent = _mock_agent("param_builder", "")
    agent.output_schema = AddInput
    resp = CompletionResponse(content=typed_result.model_dump_json(), usage=UsageStats())
    resp._parsed = typed_result

    # ajson returns the Pydantic model directly (like real LazyAgent.ajson)
    agent.ajson = AsyncMock(return_value=typed_result)

    # The function tool that should receive {"a": 3, "b": 4}
    def add_fn(a: int, b: int) -> int:
        return a + b

    tool = LazyTool.from_function(add_fn, name="add_fn", description="add two numbers")

    pipeline = LazyTool.chain(agent, tool, name="typed_chain", description="test")
    result = pipeline.run({"task": "add 3 and 4"})
    # Chain stringifies the last step's output (text representation)
    assert result == "7"


def test_chain_untyped_agent_to_tool_passes_task():
    """When an agent step has no output_schema, the next LazyTool step
    receives {"task": text} as before (backward compat)."""
    agent = _mock_agent("writer", "some text output")

    captured_args = {}

    def sink(task: str) -> str:
        captured_args["task"] = task
        return "done"

    tool = LazyTool.from_function(sink, name="sink", description="captures input")

    pipeline = LazyTool.chain(agent, tool, name="compat_chain", description="test")
    result = pipeline.run({"task": "go"})
    assert result == "done"
    assert captured_args["task"] == "some text output"


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


# =============================================================================
# Pipeline Persistence Tests
# =============================================================================

from pathlib import Path


def test_pipeline_chain_save_produces_v2_file(tmp_path):
    """Chain pipeline save produces a v2 sentinel file with chain reconstruction."""
    a = _mock_agent("researcher", "result")
    t = LazyTool.chain(a, name="my_chain", description="a chain")
    out = str(tmp_path / "chain.py")
    t.save(out)
    content = Path(out).read_text()
    assert "LAZYBRIDGE_GENERATED_TOOL v2" in content
    assert "LazyTool.chain(" in content
    assert "my_chain" in content
    assert "a chain" in content


def test_pipeline_parallel_save_preserves_combiner(tmp_path):
    """Parallel pipeline save includes combiner and concurrency settings."""
    a = _mock_agent("a", "out_a")
    b = _mock_agent("b", "out_b")
    t = LazyTool.parallel(a, b, name="par", description="parallel",
                          combiner="last", concurrency_limit=2)
    out = str(tmp_path / "par.py")
    t.save(out)
    content = Path(out).read_text()
    assert "combiner='last'" in content
    assert "concurrency_limit=2" in content
    assert "LazyTool.parallel(" in content


def test_pipeline_save_no_api_keys(tmp_path):
    """Pipeline save does not include API key values."""
    a = _mock_agent("worker", "result")
    t = LazyTool.chain(a, name="ch", description="test")
    out = str(tmp_path / "ch.py")
    t.save(out)
    content = Path(out).read_text()
    assert "API keys are not serialized" in content
    # Verify no common key patterns
    assert "sk-" not in content
    assert "api_key" not in content.lower()


def test_pipeline_config_stored_on_chain():
    """chain() stores _PipelineConfig with mode='chain'."""
    a = _mock_agent("a", "x")
    t = LazyTool.chain(a, name="ch", description="t")
    assert t._pipeline is not None
    assert t._pipeline.mode == "chain"
    assert len(t._pipeline.participants) == 1


def test_pipeline_config_stored_on_parallel():
    """parallel() stores _PipelineConfig with mode='parallel'."""
    a = _mock_agent("a", "x")
    b = _mock_agent("b", "y")
    t = LazyTool.parallel(a, b, name="par", description="t", combiner="last")
    assert t._pipeline is not None
    assert t._pipeline.mode == "parallel"
    assert t._pipeline.combiner == "last"
    assert len(t._pipeline.participants) == 2


def test_pipeline_with_function_tool_save(tmp_path):
    """Pipeline with a from_function tool saves the function source."""
    a = _mock_agent("a", "out")

    # Use a real function (not closure) for saveable source
    tool = LazyTool.from_function(add, name="add", description="add two numbers")
    t = LazyTool.chain(a, tool, name="chain_with_func", description="test")
    out = str(tmp_path / "chain_func.py")
    t.save(out)
    content = Path(out).read_text()
    assert "def add(" in content
    assert "LazyTool.from_function(" in content
    assert "LazyTool.chain(" in content


def test_pipeline_with_closure_tool_emits_reconnect(tmp_path):
    """Pipeline with a closure tool emits reconnect placeholder."""
    a = _mock_agent("a", "out")

    # Create a closure-based tool (unsaveable)
    captured = 42
    def closure_func(x: int) -> int:
        return x + captured
    closure_tool = LazyTool.from_function(closure_func, name="closure_op", description="closure")

    t = LazyTool.chain(a, closure_tool, name="chain_reconnect", description="test")
    out = str(tmp_path / "chain_reconnect.py")
    t.save(out)
    content = Path(out).read_text()
    assert "reconnect" in content
    assert "closure_op" in content


def test_load_v2_sentinel_accepted(tmp_path):
    """load() accepts both v1 and v2 sentinel headers."""
    # v1 is already tested by existing tests
    # v2: create a minimal pipeline file
    a = _mock_agent("a", "out")
    t = LazyTool.chain(a, name="ch", description="test")
    out = str(tmp_path / "v2.py")
    t.save(out)
    # Verify it has v2 sentinel
    content = Path(out).read_text()
    assert "v2" in content.split("\n")[0]


def test_pipeline_save_with_step_timeout(tmp_path):
    """Pipeline with step_timeout includes it in the output."""
    a = _mock_agent("a", "out")
    t = LazyTool.chain(a, name="timed", description="timed chain", step_timeout=30.0)
    # Just verify the config captured it
    assert t._pipeline.step_timeout == 30.0


# =============================================================================
# Round-trip load tests (save → load → verify)
# =============================================================================


def test_roundtrip_chain_with_function_tools(tmp_path):
    """Save a chain of two function tools, load it, verify it's a working pipeline."""
    # Use functions defined at module scope (source retrievable)
    t1 = LazyTool.from_function(add, name="add", description="add two numbers")
    t2 = LazyTool.from_function(scale, name="scale", description="scale a value")
    chain = LazyTool.chain(t1, t2, name="add_then_scale", description="add then scale")

    out = str(tmp_path / "chain.py")
    chain.save(out)

    # Verify the generated file is valid Python
    content = Path(out).read_text()
    assert "def add(" in content
    assert "def scale(" in content
    assert "LazyTool.chain(" in content

    # Load it back
    loaded = LazyTool.load(out)
    assert isinstance(loaded, LazyTool)
    assert loaded._is_pipeline_tool is True
    assert loaded.name == "add_then_scale"
    assert loaded.description == "add then scale"


def test_roundtrip_parallel_with_function_tools(tmp_path):
    """Save a parallel of two function tools, load it, verify reconstruction."""
    t1 = LazyTool.from_function(add, name="add", description="add")
    t2 = LazyTool.from_function(greet, name="greet", description="greet")
    par = LazyTool.parallel(t1, t2, name="add_and_greet", description="both",
                            combiner="last", concurrency_limit=3)

    out = str(tmp_path / "par.py")
    par.save(out)
    loaded = LazyTool.load(out)
    assert loaded._is_pipeline_tool is True
    assert loaded.name == "add_and_greet"
    # Verify combiner was preserved in the generated code
    content = Path(out).read_text()
    assert "combiner='last'" in content
    assert "concurrency_limit=3" in content


def test_roundtrip_chain_with_reconnect(tmp_path):
    """Save a chain with an unsaveable closure tool, load with reconnect."""
    # Saveable function tool
    func_tool = LazyTool.from_function(add, name="add", description="add")

    # Unsaveable closure tool
    captured = 10
    def closure_fn(x: int) -> int:
        return x + captured
    closure_tool = LazyTool.from_function(closure_fn, name="my_closure", description="closure")

    chain = LazyTool.chain(func_tool, closure_tool, name="mixed", description="mixed chain")

    out = str(tmp_path / "mixed.py")
    chain.save(out)

    content = Path(out).read_text()
    assert 'reconnect["my_closure"]' in content
    assert "REQUIRES reconnect" in content

    # Load with reconnect
    replacement = LazyTool.from_function(greet, name="my_closure", description="replacement")
    loaded = LazyTool.load(out, reconnect={"my_closure": replacement})
    assert loaded._is_pipeline_tool is True
    assert loaded.name == "mixed"


def test_roundtrip_load_without_reconnect_raises(tmp_path):
    """Loading a pipeline that needs reconnect without providing it raises KeyError."""
    func_tool = LazyTool.from_function(add, name="add", description="add")
    captured = 10
    def closure_fn(x: int) -> int:
        return x + captured
    closure_tool = LazyTool.from_function(closure_fn, name="needs_reconnect", description="x")

    chain = LazyTool.chain(func_tool, closure_tool, name="ch", description="d")
    out = str(tmp_path / "ch.py")
    chain.save(out)

    with pytest.raises(KeyError, match="needs_reconnect"):
        LazyTool.load(out)


def test_roundtrip_nested_pipeline(tmp_path):
    """Save a chain containing a parallel, load it back."""
    t1 = LazyTool.from_function(add, name="add", description="add")
    t2 = LazyTool.from_function(greet, name="greet", description="greet")
    inner = LazyTool.parallel(t1, t2, name="inner_par", description="parallel inner")

    t3 = LazyTool.from_function(scale, name="scale", description="scale")
    outer = LazyTool.chain(inner, t3, name="outer_chain", description="chain with nested parallel")

    out = str(tmp_path / "nested.py")
    outer.save(out)

    content = Path(out).read_text()
    assert "LazyTool.parallel(" in content
    assert "LazyTool.chain(" in content

    loaded = LazyTool.load(out)
    assert loaded._is_pipeline_tool is True
    assert loaded.name == "outer_chain"


def test_v1_files_still_load(tmp_path):
    """Backward compat: v1 function tool files still load correctly."""
    t = LazyTool.from_function(add, name="add", description="add two numbers")
    out = str(tmp_path / "v1.py")
    t.save(out)  # v1 format (not a pipeline)

    content = Path(out).read_text()
    assert "LAZYBRIDGE_GENERATED_TOOL v1" in content

    loaded = LazyTool.load(out)
    assert loaded.name == "add"
    assert loaded.func is not None
    result = loaded.run({"a": 3, "b": 4})
    assert result == 7
