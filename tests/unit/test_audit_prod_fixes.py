"""Regression tests for the production-readiness audit fix set.

Covers:
  * B1 — LLMEngine passes retry config to its Executor.
  * B2 — No calls to deprecated ``asyncio.get_event_loop`` in hot paths.
  * M1 — Store and Session EventLog close thread-local SQLite conns.
  * M2 — Memory caps turns at ``max_turns`` even under ``strategy="none"``.
  * M3 — Bare-except no longer swallows Pydantic coercion errors silently.
  * M4 — Agent.run(timeout=) surfaces a TimeoutError envelope.
"""

from __future__ import annotations

import asyncio
import sqlite3
import warnings

import pytest

from lazybridge.agent import Agent
from lazybridge.engines.llm import LLMEngine
from lazybridge.envelope import Envelope
from lazybridge.ext.hil.human import HumanEngine
from lazybridge.memory import Memory
from lazybridge.session import EventType, Session
from lazybridge.store import Store

# ── B1: LLMEngine retry defaults ──────────────────────────────────────────────


def test_llm_engine_defaults_retries_enabled():
    eng = LLMEngine("claude-opus-4-7")
    assert eng.max_retries == 3
    assert eng.retry_delay == 1.0
    assert eng.request_timeout == 120.0


def test_llm_engine_executor_receives_retry_config():
    eng = LLMEngine("claude-opus-4-7", max_retries=5, retry_delay=0.25)
    # Bypass real provider resolution by patching _resolve_provider on the
    # executor path — easier to just inspect the Executor factory directly.
    from lazybridge.core.executor import Executor

    captured: dict = {}
    orig_init = Executor.__init__

    def capture(self, provider, **kwargs):
        captured.update(kwargs)
        captured["provider"] = provider
        # Don't actually construct the real provider — just record.
        self._provider = object()
        self._max_retries = kwargs.get("max_retries", 0)
        self._retry_delay = kwargs.get("retry_delay", 1.0)

    Executor.__init__ = capture  # type: ignore[assignment]
    try:
        eng._make_executor()
    finally:
        Executor.__init__ = orig_init
    assert captured["max_retries"] == 5
    assert captured["retry_delay"] == 0.25


# ── B2: no deprecated get_event_loop in Tool.run / HumanEngine ────────────────


def test_tools_module_uses_get_running_loop():
    import lazybridge.tools as tools_mod

    path = tools_mod.__file__ or ""
    with open(path) as fh:  # type: ignore[arg-type]
        src = path and fh.read()
    assert "get_event_loop()" not in src, "get_event_loop is deprecated on 3.10+"


def test_human_module_uses_get_running_loop():
    import lazybridge.ext.hil.human as human_mod

    with open(human_mod.__file__) as fh:  # type: ignore[arg-type]
        src = fh.read()
    assert "get_event_loop()" not in src


# ── M1: SQLite connection lifecycle ───────────────────────────────────────────


def test_store_close_releases_connections(tmp_path):
    db_path = str(tmp_path / "store.sqlite")
    s = Store(db=db_path)
    s.write("k", "v")
    conns = list(s._all_conns)
    assert conns, "thread-local connection should be registered"
    s.close()
    assert s._closed is True
    # Closed conn raises on use
    for c in conns:
        with pytest.raises(sqlite3.ProgrammingError):
            c.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="closed"):
        s.write("k2", "v2")


def test_store_context_manager_closes(tmp_path):
    db_path = str(tmp_path / "store.sqlite")
    with Store(db=db_path) as s:
        s.write("a", 1)
    assert s._closed is True


def test_session_close_releases_eventlog(tmp_path):
    db_path = str(tmp_path / "events.sqlite")
    with Session(db=db_path) as sess:
        sess.emit(EventType.AGENT_START, {"agent_name": "t"}, run_id="r1")
        assert sess.events._all_conns
    assert sess.events._closed is True


# ── M2: Memory turn cap ───────────────────────────────────────────────────────


def test_memory_caps_turns_under_strategy_none():
    m = Memory(strategy="none", max_turns=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(10):
            m.add(f"u{i}", f"a{i}")
        assert len(m._turns) == 3
        assert any("max_turns" in str(w.message) for w in caught)
    # Latest turns win (FIFO drop of oldest)
    assert m._turns[-1].user == "u9"
    assert m._turns[0].user == "u7"


def test_memory_unbounded_when_max_turns_none():
    m = Memory(strategy="none", max_turns=None)
    for i in range(50):
        m.add(f"u{i}", f"a{i}")
    assert len(m._turns) == 50  # opt-in unbounded still works


# ── M3: structured-output coercion failures emit an event ─────────────────────


@pytest.mark.asyncio
async def test_human_engine_records_coercion_failure(monkeypatch):
    """HumanEngine: when the human's input can't be coerced to the
    requested ``output_type``, the engine must return an **error
    envelope** — not a silently-downgraded string payload the caller
    can't tell apart from a successful answer (audit finding #6).
    The raw string is still exposed via ``.payload`` for debug.
    """
    from pydantic import BaseModel

    class Answer(BaseModel):
        n: int

    # Stub UI that returns a non-JSON string the Answer model can't coerce.
    class StubUI:
        async def prompt(self, task, *, tools, output_type):
            return "not-an-int"

    eng = HumanEngine(ui=StubUI())  # type: ignore[arg-type]
    with Session() as sess:
        env = Envelope(task="give me n")
        out = await eng.run(env, tools=[], output_type=Answer, memory=None, session=sess)
        # Raw string remains on payload for diagnostic inspection.
        assert out.payload == "not-an-int"
        # Error envelope surfaces the coercion failure clearly.
        assert not out.ok
        assert out.error is not None
        assert out.error.type == "StructuredOutputCoercionError"
        assert "Answer" in out.error.message
        # Audit trail still emits the structured-output event.
        tool_errors = sess.events.query(event_type=EventType.TOOL_ERROR)
        kinds = [e["payload"].get("kind") for e in tool_errors]
        assert "structured_output_coercion" in kinds


@pytest.mark.asyncio
async def test_human_engine_success_path_unchanged() -> None:
    """Sanity: valid input still produces an ok envelope with the
    parsed model as payload — the error-envelope path is only taken
    on coercion failure."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        n: int

    class StubUI:
        async def prompt(self, task, *, tools, output_type):
            return '{"n": 42}'

    eng = HumanEngine(ui=StubUI())  # type: ignore[arg-type]
    env = Envelope(task="give me n")
    out = await eng.run(env, tools=[], output_type=Answer, memory=None, session=None)
    assert out.ok
    assert isinstance(out.payload, Answer)
    assert out.payload.n == 42


# ── M4: Agent.run(timeout=) ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_run_honours_timeout():
    """An engine that never returns is cut off at ``timeout`` and yields
    a TimeoutError envelope rather than blocking indefinitely.
    """

    class _SlowEngine:
        model = "fake"
        _agent_name = "t"
        native_tools: list = []

        async def run(self, env, *, tools, output_type, memory, session):
            await asyncio.sleep(5)
            return Envelope(task=env.task, payload="never")

    a = Agent.__new__(Agent)
    a.engine = _SlowEngine()
    a._tools_raw = []
    a._tool_map = {}
    a.output = str
    a.memory = None
    a.sources = []
    a.guard = None
    a.verify = None
    a.max_verify = 3
    a.name = "t"
    a.description = None
    a.session = None
    a.timeout = 0.05

    result = await a.run("hi")
    assert result.error is not None
    assert "timeout" in result.error.message.lower()
