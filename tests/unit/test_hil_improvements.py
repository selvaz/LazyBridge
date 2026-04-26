"""HIL improvements: audit events, async input_fn, re-prompt on
validation error, public scripted_inputs helper.
"""

from __future__ import annotations

import asyncio

import pytest

from lazybridge import (
    Agent,
    EventType,
    Session,
)
from lazybridge.ext.hil import HumanEngine, SupervisorEngine
from lazybridge.ext.hil.human import _TerminalUI
from lazybridge.testing import scripted_ainputs, scripted_inputs

# ---------------------------------------------------------------------------
# Public scripted_inputs / scripted_ainputs
# ---------------------------------------------------------------------------


def test_scripted_inputs_is_public_and_callable():
    fn = scripted_inputs(["a", "b", "c"])
    assert fn("prompt") == "a"
    assert fn("prompt") == "b"
    assert fn("prompt") == "c"
    with pytest.raises(StopIteration):
        fn("prompt")


@pytest.mark.asyncio
async def test_scripted_ainputs_awaitable():
    fn = scripted_ainputs(["x", "y"])
    assert (await fn("p")) == "x"
    assert (await fn("p")) == "y"


def test_scripted_inputs_drives_supervisor_repl():
    sess = Session()
    sup = Agent(
        engine=SupervisorEngine(input_fn=scripted_inputs(["continue"])),
        name="sup",
        session=sess,
    )
    env = sup("task-x")
    assert env.text() == "task-x"


# ---------------------------------------------------------------------------
# HIL_DECISION audit events
# ---------------------------------------------------------------------------


def test_supervisor_emits_hil_decision_on_continue():
    sess = Session()
    sup = Agent(
        engine=SupervisorEngine(input_fn=scripted_inputs(["continue"])),
        name="sup",
        session=sess,
    )
    sup("hello")

    events = sess.events.query(event_type=EventType.HIL_DECISION)
    assert len(events) == 1
    assert events[0]["payload"]["kind"] == "continue"
    assert events[0]["payload"]["command"] == "continue"


def test_supervisor_emits_one_hil_decision_per_command():
    def echo(q: str) -> str:
        """Echo."""
        return f"echo:{q}"

    sess = Session()
    sup = Agent(
        engine=SupervisorEngine(
            tools=[echo],
            input_fn=scripted_inputs(["echo(hi)", "store missing", "continue"]),
        ),
        name="sup",
        session=sess,
    )
    sup("start")

    events = sess.events.query(event_type=EventType.HIL_DECISION)
    kinds = [e["payload"]["kind"] for e in events]
    # tool call, store miss (no store configured), continue
    assert kinds == ["tool", "store", "continue"]


def test_supervisor_emits_hil_decision_for_unknown_command():
    sess = Session()
    sup = Agent(
        engine=SupervisorEngine(input_fn=scripted_inputs(["wat", "continue"])),
        name="sup",
        session=sess,
    )
    sup("x")

    events = sess.events.query(event_type=EventType.HIL_DECISION)
    assert any(e["payload"]["kind"] == "unknown" for e in events)


def test_human_engine_emits_hil_decision_input_kind():
    """Even a basic HumanEngine logs an audit event distinct from
    AGENT_FINISH, so downstream tooling can tell human answers apart
    from LLM answers.
    """
    # A terminal UI whose ``prompt`` just returns a canned string.
    class _FakeUI:
        async def prompt(self, task, *, tools, output_type):
            return "the answer"

    eng = HumanEngine()
    eng._ui = _FakeUI()
    sess = Session()

    ag = Agent(engine=eng, name="h", session=sess)
    ag("hello")

    decisions = sess.events.query(event_type=EventType.HIL_DECISION)
    assert len(decisions) == 1
    assert decisions[0]["payload"]["kind"] == "input"
    assert decisions[0]["payload"]["result"] == "the answer"


# ---------------------------------------------------------------------------
# Async REPL path via ainput_fn
# ---------------------------------------------------------------------------


def test_supervisor_uses_async_repl_when_ainput_fn_supplied():
    """Supplying ``ainput_fn`` routes through the event-loop-native REPL
    instead of the thread-pool fallback.  We verify via a counter inside
    the ainput_fn (scripted versions are trivially async).
    """
    calls: list[str] = []

    async def ainput(prompt):
        calls.append(prompt)
        return "continue"

    sup = Agent(
        engine=SupervisorEngine(ainput_fn=ainput),
        name="sup",
    )
    sup("task")

    assert len(calls) == 1
    assert calls[0].startswith("[sup]")


@pytest.mark.asyncio
async def test_supervisor_async_repl_with_tool_call():
    """Full async REPL including a scripted tool invocation."""
    def echo(q: str) -> str:
        """Echo."""
        return f"e:{q}"

    sup = Agent(
        engine=SupervisorEngine(
            tools=[echo],
            ainput_fn=scripted_ainputs(["echo(hi)", "continue"]),
        ),
        name="sup",
    )
    env = await sup.run("start")
    assert env.text() == "e:hi"


# ---------------------------------------------------------------------------
# HumanEngine re-prompt on ValidationError
# ---------------------------------------------------------------------------


def test_human_engine_coerce_field_strict_raises_on_bad_int(monkeypatch):
    """The strict coercion used by the re-prompt loop must raise so the
    loop can show the error and re-prompt, instead of falling back to
    string and failing Pydantic at the very end of the form.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _TerminalUI._coerce_field_strict(int, "not-a-number")


def test_human_engine_coerce_field_strict_accepts_valid_int():
    assert _TerminalUI._coerce_field_strict(int, "42") == 42


@pytest.mark.asyncio
async def test_human_engine_reprompts_until_valid(monkeypatch):
    """_prompt_field loops: first two inputs are garbage for int, third
    is valid.  Verify the field ends up with the valid int and only the
    third call survived.
    """
    answers = iter(["abc", "def", "7"])

    async def fake_run_in_executor(loop, _func, *args):  # type: ignore[no-untyped-def]
        return next(answers)

    class _FakeLoop:
        def run_in_executor(self, *a, **kw):
            fut = asyncio.Future()
            try:
                fut.set_result(next(answers))
            except StopIteration:
                fut.set_exception(RuntimeError("no more scripted inputs"))
            return fut

    def fake_get_running_loop():
        return _FakeLoop()

    ui = _TerminalUI()
    # HumanEngine now uses ``asyncio.get_running_loop`` (Python 3.13+
    # safe); ``get_event_loop`` is deprecated and no longer called.
    monkeypatch.setattr(asyncio, "get_running_loop", fake_get_running_loop)

    result = await ui._prompt_field("n", int, "int")
    assert result == 7
