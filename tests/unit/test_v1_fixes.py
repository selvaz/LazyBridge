"""Tests for v1.0 fix set: as_tool, usage_summary, native_tools, stream, sources."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lazybridge.agent import Agent, _ParallelAgent
from lazybridge.envelope import Envelope, EnvelopeMetadata
from lazybridge.session import EventType, Session
from lazybridge.store import Store
from lazybridge.memory import Memory
from lazybridge.tools import Tool, build_tool_map


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_agent(response: str = "hello", *, name: str = "agent") -> Agent:
    """Agent whose engine is mocked to return a fixed response."""
    from lazybridge.engines.llm import LLMEngine
    engine = LLMEngine.__new__(LLMEngine)
    engine.model = "claude-opus-4-7"
    engine.thinking = False
    engine.max_turns = 10
    engine.tool_choice = "auto"
    engine.temperature = None
    engine.system = None
    engine.native_tools = []
    engine._agent_name = name

    async def _fake_run(env, *, tools, output_type, memory, session):
        return Envelope(task=env.task, payload=response,
                        metadata=EnvelopeMetadata(input_tokens=10, output_tokens=20,
                                                  cost_usd=0.001))

    async def _fake_stream(env, *, tools, output_type, memory, session):
        for ch in response:
            yield ch

    engine.run = _fake_run
    engine.stream = _fake_stream

    a = Agent.__new__(Agent)
    a.engine = engine
    a._tools_raw = []
    a._tool_map = {}
    a.output = str
    a.memory = None
    a.sources = []
    a.guard = None
    a.verify = None
    a.max_verify = 3
    a.name = name
    a.description = None
    a.session = None
    return a


# ── 1. agent.as_tool() ────────────────────────────────────────────────────────

class TestAsToolMethod:
    def test_returns_tool(self):
        a = _make_agent("result")
        t = a.as_tool()
        assert isinstance(t, Tool)

    def test_name_defaults_to_agent_name(self):
        a = _make_agent(name="researcher")
        t = a.as_tool()
        assert t.name == "researcher"

    def test_name_override(self):
        a = _make_agent()
        t = a.as_tool(name="my_tool")
        assert t.name == "my_tool"

    def test_description_override(self):
        a = _make_agent()
        t = a.as_tool(description="Does a thing")
        assert "Does a thing" in t.definition().description

    def test_tool_runs_agent(self):
        # Post-Envelope-through-Tool change: Agent.as_tool returns the
        # inner agent's full Envelope rather than a flattened string.
        # The string payload is still reachable via .text() / str(env).
        from lazybridge import Envelope

        a = _make_agent("the answer")
        t = a.as_tool()
        result = asyncio.run(t.run(task="question"))
        assert isinstance(result, Envelope)
        assert result.text() == "the answer"
        assert str(result) == "the answer"    # __str__ falls through to text()
        assert t.returns_envelope is True

    def test_tool_definition_has_task_param(self):
        a = _make_agent()
        defn = a.as_tool().definition()
        assert "task" in defn.parameters.get("properties", {})

    def test_agent_usable_as_tool_in_another_agent(self):
        inner = _make_agent("inner result", name="inner")
        t = inner.as_tool("inner", "runs inner")
        tm = build_tool_map([t])
        assert "inner" in tm


# ── 2. Session.usage_summary() ────────────────────────────────────────────────

class TestUsageSummary:
    def test_empty_session(self):
        sess = Session()
        summary = sess.usage_summary()
        assert summary["total"]["input_tokens"] == 0
        assert summary["total"]["output_tokens"] == 0
        assert summary["total"]["cost_usd"] == 0.0
        assert summary["by_agent"] == {}

    def test_accumulates_model_responses(self):
        sess = Session()
        run_id = "run-1"
        sess.emit(EventType.AGENT_START, {"agent_name": "researcher"}, run_id=run_id)
        sess.emit(EventType.MODEL_RESPONSE,
                  {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01},
                  run_id=run_id)
        sess.emit(EventType.MODEL_RESPONSE,
                  {"input_tokens": 200, "output_tokens": 80, "cost_usd": 0.02},
                  run_id=run_id)

        s = sess.usage_summary()
        assert s["total"]["input_tokens"] == 300
        assert s["total"]["output_tokens"] == 130
        assert abs(s["total"]["cost_usd"] - 0.03) < 1e-9

    def test_by_agent_breakdown(self):
        sess = Session()
        for i, name in enumerate(["agent_a", "agent_b"]):
            rid = f"run-{i}"
            sess.emit(EventType.AGENT_START, {"agent_name": name}, run_id=rid)
            sess.emit(EventType.MODEL_RESPONSE,
                      {"input_tokens": 10 * (i + 1), "output_tokens": 5, "cost_usd": 0.001},
                      run_id=rid)

        s = sess.usage_summary()
        assert "agent_a" in s["by_agent"]
        assert "agent_b" in s["by_agent"]
        assert s["by_agent"]["agent_a"]["input_tokens"] == 10
        assert s["by_agent"]["agent_b"]["input_tokens"] == 20

    def test_by_run_breakdown(self):
        sess = Session()
        sess.emit(EventType.AGENT_START, {"agent_name": "x"}, run_id="r1")
        sess.emit(EventType.MODEL_RESPONSE,
                  {"input_tokens": 5, "output_tokens": 3, "cost_usd": 0.0},
                  run_id="r1")
        s = sess.usage_summary()
        assert "r1" in s["by_run"]
        assert s["by_run"]["r1"]["input_tokens"] == 5

    def test_null_cost_treated_as_zero(self):
        sess = Session()
        sess.emit(EventType.AGENT_START, {"agent_name": "x"}, run_id="r")
        sess.emit(EventType.MODEL_RESPONSE,
                  {"input_tokens": 1, "output_tokens": 1, "cost_usd": None},
                  run_id="r")
        s = sess.usage_summary()
        assert s["total"]["cost_usd"] == 0.0


# ── 3. LLMEngine native_tools ─────────────────────────────────────────────────

class TestNativeTools:
    def test_native_tools_stored(self):
        from lazybridge.engines.llm import LLMEngine
        from lazybridge.core.types import NativeTool
        engine = LLMEngine("claude-opus-4-7", native_tools=[NativeTool.WEB_SEARCH])
        assert NativeTool.WEB_SEARCH in engine.native_tools

    def test_native_tools_from_str(self):
        from lazybridge.engines.llm import LLMEngine
        from lazybridge.core.types import NativeTool
        engine = LLMEngine("claude-opus-4-7", native_tools=["web_search"])
        assert engine.native_tools[0] == NativeTool.WEB_SEARCH

    def test_native_tools_empty_default(self):
        from lazybridge.engines.llm import LLMEngine
        engine = LLMEngine("claude-opus-4-7")
        assert engine.native_tools == []

    def test_native_tools_attribute_set(self):
        from lazybridge.engines.llm import LLMEngine
        from lazybridge.core.types import NativeTool
        engine = LLMEngine("claude-opus-4-7", native_tools=[NativeTool.WEB_SEARCH, "code_execution"])
        assert len(engine.native_tools) == 2
        assert NativeTool.WEB_SEARCH in engine.native_tools
        assert NativeTool.CODE_EXECUTION in engine.native_tools


# ── 4. stream() full loop ─────────────────────────────────────────────────────

class TestStreamLoop:
    def test_stream_yields_tokens(self):
        a = _make_agent("hello world")

        async def _collect():
            parts = []
            async for tok in a.stream("say hello"):
                parts.append(tok)
            return parts

        tokens = asyncio.run(_collect())
        assert "".join(tokens) == "hello world"

    def test_stream_multiple_chars(self):
        a = _make_agent("abc")

        async def _collect():
            return [t async for t in a.stream("task")]

        tokens = asyncio.run(_collect())
        assert len(tokens) == 3  # one char per yield in mock
        assert "".join(tokens) == "abc"

    def test_llmengine_stream_method_signature(self):
        from lazybridge.engines.llm import LLMEngine
        import inspect
        sig = inspect.signature(LLMEngine.stream)
        params = list(sig.parameters.keys())
        assert "env" in params
        assert "tools" in params
        assert "memory" in params
        assert "session" in params


# ── 5. sources= robustification ──────────────────────────────────────────────

class TestSourcesInjection:
    def test_store_as_source(self):
        store = Store()
        store.write("key", "stored value")
        a = _make_agent("ok")
        a.sources = [store]

        env = Envelope.from_task("task")
        result = a._inject_sources(env)
        assert "stored value" in (result.context or "")

    def test_memory_as_source(self):
        mem = Memory()
        mem.add("previous question", "previous answer")
        a = _make_agent("ok")
        a.sources = [mem]

        env = Envelope.from_task("task")
        result = a._inject_sources(env)
        assert "previous" in (result.context or "")

    def test_callable_as_source(self):
        a = _make_agent("ok")
        a.sources = [lambda: "dynamic context"]

        env = Envelope.from_task("task")
        result = a._inject_sources(env)
        assert "dynamic context" in (result.context or "")

    def test_string_as_source(self):
        a = _make_agent("ok")
        a.sources = ["static context text"]

        env = Envelope.from_task("task")
        result = a._inject_sources(env)
        assert "static context text" in (result.context or "")

    def test_existing_context_preserved(self):
        a = _make_agent("ok")
        a.sources = ["extra"]

        env = Envelope(task="t", context="original ctx")
        result = a._inject_sources(env)
        assert "original ctx" in (result.context or "")
        assert "extra" in (result.context or "")

    def test_empty_sources_no_change(self):
        a = _make_agent("ok")
        a.sources = []

        env = Envelope(task="t", context="ctx")
        result = a._inject_sources(env)
        assert result is env  # same object returned

    def test_multiple_sources_combined(self):
        a = _make_agent("ok")
        a.sources = [lambda: "part1", lambda: "part2"]

        env = Envelope.from_task("task")
        result = a._inject_sources(env)
        assert "part1" in (result.context or "")
        assert "part2" in (result.context or "")

    def test_live_view_reads_current_state(self):
        store = Store()
        store.write("k", "v1")
        a = _make_agent("ok")
        a.sources = [store]

        env = Envelope.from_task("t")
        r1 = a._inject_sources(env)
        assert "v1" in (r1.context or "")

        store.write("k", "v2")
        r2 = a._inject_sources(env)
        assert "v2" in (r2.context or "")
        assert "v1" not in (r2.context or "")


# ── 6. _ParallelAgent semaphore fix ──────────────────────────────────────────

class TestParallelAgent:
    def test_parallel_returns_list(self):
        a1 = _make_agent("result1", name="a1")
        a2 = _make_agent("result2", name="a2")
        pa = _ParallelAgent([a1, a2])
        results = asyncio.run(pa.run("task"))
        assert len(results) == 2
        texts = {r.text() for r in results}
        assert "result1" in texts
        assert "result2" in texts

    def test_parallel_with_concurrency_limit(self):
        a1 = _make_agent("r1", name="a1")
        a2 = _make_agent("r2", name="a2")
        pa = _ParallelAgent([a1, a2], concurrency_limit=1)
        results = asyncio.run(pa.run("task"))
        assert len(results) == 2

    def test_parallel_error_captured_not_raised(self):
        good = _make_agent("ok", name="good")

        class _BoomEngine:
            _agent_name = "bad"
            async def run(self, env, *, tools, output_type, memory, session):
                raise RuntimeError("boom")
            async def stream(self, *a, **k):
                raise NotImplementedError

        bad = Agent.__new__(Agent)
        bad.engine = _BoomEngine()
        bad._tool_map = {}
        bad._tools_raw = []
        bad.output = str
        bad.memory = None
        bad.sources = []
        bad.guard = None
        bad.verify = None
        bad.max_verify = 3
        bad.name = "bad"
        bad.description = None
        bad.session = None

        pa = _ParallelAgent([good, bad])
        results = asyncio.run(pa.run("task"))
        assert len(results) == 2
        oks = [r for r in results if r.ok]
        errs = [r for r in results if not r.ok]
        assert len(oks) == 1
        assert len(errs) == 1
