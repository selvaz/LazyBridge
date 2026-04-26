"""v1.0 core verification tests — Steps 1-8 of the rewrite plan."""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel

from lazybridge.engines.plan import Plan, PlanCompileError, PlanCompiler, Step
from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo
from lazybridge.ext.evals import EvalCase, EvalSuite, contains, exact_match
from lazybridge.guardrails import ContentGuard, GuardAction, GuardChain
from lazybridge.memory import Memory
from lazybridge.sentinels import (
    _FromParallel,
    _FromPrev,
    _FromStart,
    _FromStep,
    from_parallel,
    from_prev,
    from_start,
    from_step,
)
from lazybridge.session import EventType, Session
from lazybridge.store import Store
from lazybridge.tools import Tool, build_tool_map, wrap_tool

# ============================================================
# Step 1: Envelope
# ============================================================


class TestEnvelope:
    def test_text_from_str_payload(self):
        env = Envelope(task="hi", payload="world")
        assert env.text() == "world"

    def test_text_from_none(self):
        env = Envelope(task="hi")
        assert env.text() == ""

    def test_text_from_pydantic(self):
        class M(BaseModel):
            x: int

        env = Envelope(payload=M(x=1))
        assert '"x"' in env.text()

    def test_ok_no_error(self):
        env = Envelope(task="t", payload="p")
        assert env.ok is True

    def test_ok_with_error(self):
        env = Envelope(error=ErrorInfo(type="E", message="oops"))
        assert env.ok is False

    def test_from_task(self):
        env = Envelope.from_task("hello", context="ctx")
        assert env.task == "hello"
        assert env.context == "ctx"
        assert env.payload == "hello"

    def test_error_envelope(self):
        env = Envelope.error_envelope(ValueError("bad"))
        assert not env.ok
        assert env.error.type == "ValueError"
        assert env.error.message == "bad"

    def test_metadata_defaults(self):
        meta = EnvelopeMetadata()
        assert meta.input_tokens == 0
        assert meta.cost_usd == 0.0


# ============================================================
# Step 2: Sentinels
# ============================================================


class TestSentinels:
    def test_from_prev_singleton(self):
        assert isinstance(from_prev, _FromPrev)

    def test_from_start_singleton(self):
        assert isinstance(from_start, _FromStart)

    def test_from_step_factory(self):
        s = from_step("my_step")
        assert isinstance(s, _FromStep)
        assert s.name == "my_step"

    def test_from_parallel_factory(self):
        p = from_parallel("branch_a")
        assert isinstance(p, _FromParallel)
        assert p.name == "branch_a"

    def test_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            from_prev.something = "x"  # type: ignore[attr-defined]


# ============================================================
# Step 1b: Tool
# ============================================================


class TestTool:
    def test_tool_name_from_func(self):
        def my_func(x: str) -> str:
            return x

        t = Tool(my_func)
        assert t.name == "my_func"

    def test_tool_name_override(self):
        def my_func(x: str) -> str:
            return x

        t = Tool(my_func, name="override")
        assert t.name == "override"

    def test_definition_returns_tool_definition(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        t = Tool(add)
        defn = t.definition()
        assert defn.name == "add"
        assert "a" in defn.parameters.get("properties", {})

    def test_definition_cached(self):
        def f(x: str) -> str:
            return x

        t = Tool(f)
        d1 = t.definition()
        d2 = t.definition()
        assert d1 is d2

    def test_run_sync_callable(self):
        def double(n: int) -> int:
            return n * 2

        t = Tool(double)
        assert t.run_sync(n=3) == 6

    def test_run_async_callable(self):
        async def async_double(n: int) -> int:
            return n * 2

        t = Tool(async_double)
        result = asyncio.run(t.run(n=5))
        assert result == 10

    def test_wrap_raw_callable(self):
        def fn(x: str) -> str:
            return x

        wrapped = wrap_tool(fn)
        assert isinstance(wrapped, Tool)
        assert wrapped.name == "fn"

    def test_wrap_tool_passthrough(self):
        def fn(x: str) -> str:
            return x

        t = Tool(fn)
        assert wrap_tool(t) is t

    def test_build_tool_map(self):
        def alpha(x: str) -> str:
            return x

        def beta(y: int) -> int:
            return y

        tm = build_tool_map([alpha, beta])
        assert "alpha" in tm
        assert "beta" in tm
        assert isinstance(tm["alpha"], Tool)


# ============================================================
# Step 6: Memory
# ============================================================


class TestMemory:
    def test_add_and_text(self):
        m = Memory()
        m.add("hello", "world")
        txt = m.text()
        assert "hello" in txt
        assert "world" in txt

    def test_clear(self):
        m = Memory()
        m.add("a", "b")
        m.clear()
        assert m.text() == ""

    def test_messages_returns_list(self):
        m = Memory()
        m.add("q", "a")
        msgs = m.messages()
        assert len(msgs) >= 2

    def test_compression_sliding(self):
        m = Memory(strategy="sliding", max_tokens=1)
        for i in range(15):
            m.add(f"q{i}", f"a{i}", tokens=100)
        assert len(m._turns) <= 10

    def test_strategy_none_no_compress(self):
        m = Memory(strategy="none")
        for i in range(20):
            m.add(f"q{i}", f"a{i}", tokens=1000)
        assert len(m._turns) == 20


# ============================================================
# Step 6b: Store
# ============================================================


class TestStore:
    def test_write_read(self):
        s = Store()
        s.write("k", "v")
        assert s.read("k") == "v"

    def test_read_default(self):
        s = Store()
        assert s.read("missing", "default") == "default"

    def test_read_all(self):
        s = Store()
        s.write("a", 1)
        s.write("b", 2)
        data = s.read_all()
        assert data == {"a": 1, "b": 2}

    def test_delete(self):
        s = Store()
        s.write("x", 42)
        s.delete("x")
        assert s.read("x") is None

    def test_clear(self):
        s = Store()
        s.write("x", 1)
        s.clear()
        assert s.read_all() == {}

    def test_keys(self):
        s = Store()
        s.write("a", 1)
        s.write("b", 2)
        assert set(s.keys()) == {"a", "b"}

    def test_to_text(self):
        s = Store()
        s.write("name", "Alice")
        txt = s.to_text()
        assert "name" in txt
        assert "Alice" in txt

    def test_read_entry(self):
        s = Store()
        s.write("k", "val", agent_id="agent1")
        entry = s.read_entry("k")
        assert entry is not None
        assert entry.value == "val"
        assert entry.agent_id == "agent1"


# ============================================================
# Step 5: Session
# ============================================================


class TestSession:
    def test_emit_and_query(self):
        sess = Session()
        sess.emit(EventType.AGENT_START, {"agent_name": "test"}, run_id="r1")
        events = sess.events.query(run_id="r1")
        assert len(events) == 1
        assert events[0]["event_type"] == "agent_start"

    def test_exporter_called(self):
        received = []

        class _Exp:
            def export(self, event):
                received.append(event)

        sess = Session(exporters=[_Exp()])
        sess.emit(EventType.TOOL_CALL, {"tool": "search"})
        assert len(received) == 1
        assert received[0]["event_type"] == "tool_call"

    def test_redact(self):
        received = []

        class _Exp:
            def export(self, event):
                received.append(event)

        def redact(payload):
            return {k: "***" if k == "secret" else v for k, v in payload.items()}

        sess = Session(exporters=[_Exp()], redact=redact)
        sess.emit(EventType.MODEL_REQUEST, {"secret": "key123", "model": "claude"})
        assert received[0]["secret"] == "***"
        assert received[0]["model"] == "claude"

    def test_add_remove_exporter(self):
        received = []

        class _Exp:
            def export(self, e):
                received.append(e)

        sess = Session()
        exp = _Exp()
        sess.add_exporter(exp)
        sess.emit(EventType.AGENT_START, {})
        assert len(received) == 1
        sess.remove_exporter(exp)
        sess.emit(EventType.AGENT_START, {})
        assert len(received) == 1  # not incremented


# ============================================================
# Step 9: Guardrails
# ============================================================


class TestGuardrails:
    def test_allow(self):
        g = ContentGuard(input_fn=lambda t: GuardAction.allow())
        assert g.check_input("any").allowed is True

    def test_block(self):
        g = ContentGuard(input_fn=lambda t: GuardAction.block("no"))
        action = g.check_input("bad")
        assert action.allowed is False
        assert action.message == "no"

    def test_modify(self):
        g = ContentGuard(input_fn=lambda t: GuardAction.modify("clean " + t))
        action = g.check_input("dirty")
        assert action.allowed is True
        assert action.modified_text == "clean dirty"

    def test_guard_chain_first_block(self):
        chain = GuardChain(
            ContentGuard(input_fn=lambda t: GuardAction.allow()),
            ContentGuard(input_fn=lambda t: GuardAction.block("chain_block")),
        )
        action = chain.check_input("x")
        assert not action.allowed
        assert "chain_block" in action.message

    def test_guard_chain_all_allow(self):
        chain = GuardChain(
            ContentGuard(input_fn=lambda t: GuardAction.allow()),
            ContentGuard(input_fn=lambda t: GuardAction.allow()),
        )
        assert chain.check_input("x").allowed


# ============================================================
# Step 9b: Evals
# ============================================================


class TestEvals:
    def test_exact_match(self):
        fn = exact_match("hello")
        assert fn("hello", "hello") is True
        assert fn("world", "hello") is False

    def test_contains(self):
        fn = contains("foo")
        assert fn("foobar") is True
        assert fn("baz") is False

    def test_eval_suite_pass(self):
        class _MockAgent:
            def __call__(self, task):
                return Envelope(payload="yes it contains foo")

        suite = EvalSuite(EvalCase(input="test", check=contains("foo")))
        report = suite.run(_MockAgent())
        assert report.passed == 1
        assert str(report) == "1/1 passed (100%)"

    def test_eval_suite_fail(self):
        class _MockAgent:
            def __call__(self, task):
                return Envelope(payload="nothing here")

        suite = EvalSuite(EvalCase(input="test", check=contains("foo")))
        report = suite.run(_MockAgent())
        assert report.passed == 0
        assert report.failed == 1


# ============================================================
# Step 8: PlanCompiler
# ============================================================


class TestPlanCompiler:
    def test_valid_plan_passes(self):
        def search(task: str) -> str:
            return "results"

        def rank(task: str) -> str:
            return "ranked"

        steps = [Step("search", name="search"), Step("rank", name="rank")]
        tools = build_tool_map([search, rank])
        compiler = PlanCompiler()
        compiler.validate(steps, tools)  # no exception

    def test_missing_tool_raises(self):
        steps = [Step("nonexistent", name="nonexistent")]
        compiler = PlanCompiler()
        with pytest.raises(PlanCompileError, match="nonexistent"):
            compiler.validate(steps, {})

    def test_invalid_from_step_reference(self):
        def a(task: str) -> str:
            return "a"

        tools = build_tool_map([a])
        steps = [Step("a", task=from_step("missing"), name="a")]
        compiler = PlanCompiler()
        with pytest.raises(PlanCompileError, match="missing"):
            compiler.validate(steps, tools)


# ============================================================
# Step 8b: Plan execution (mock tools)
# ============================================================


class TestPlanExecution:
    def test_linear_plan_runs(self):
        results: list[str] = []

        async def step_a(task: str) -> str:
            results.append("a")
            return "from_a"

        async def step_b(task: str) -> str:
            results.append("b")
            return "from_b"

        tools = build_tool_map([step_a, step_b])
        plan = Plan(
            Step("step_a", name="step_a"),
            Step("step_b", name="step_b"),
        )
        env = Envelope.from_task("start")

        async def run():
            return await plan.run(env, tools=list(tools.values()), output_type=str, memory=None, session=None)

        result = asyncio.run(run())
        assert results == ["a", "b"]
        assert result.ok

    def test_plan_with_routing(self):
        chosen: list[str] = []

        class RouteOut(BaseModel):
            result: str
            next: Literal["branch_a", "branch_b"] = "branch_a"

        async def router(task: str) -> RouteOut:
            return RouteOut(result="go_a", next="branch_a")

        async def branch_a(task: str) -> str:
            chosen.append("a")
            return "done_a"

        async def branch_b(task: str) -> str:
            chosen.append("b")
            return "done_b"

        tools = build_tool_map([router, branch_a, branch_b])
        plan = Plan(
            Step("router", name="router", output=RouteOut),
            Step("branch_a", name="branch_a"),
            Step("branch_b", name="branch_b"),
        )
        env = Envelope.from_task("start")

        async def run():
            return await plan.run(env, tools=list(tools.values()), output_type=str, memory=None, session=None)

        asyncio.run(run())
        assert "a" in chosen
        assert "b" not in chosen
