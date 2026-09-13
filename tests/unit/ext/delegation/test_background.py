from __future__ import annotations

import asyncio
import gc
import inspect
from typing import Any

import pytest

from lazybridge import Store
from lazybridge.ext.delegation.background import (
    _safe_notify,
    _track,
    make_background_delegate,
    make_parallel_delegate,
    make_persistent_consultant,
    make_plan_delegate,
)
from lazybridge.ext.delegation.jobs import JobRegistry
from lazybridge.ext.planners.durable_blackboard import DurableBlackboard


async def _drain() -> None:
    await asyncio.sleep(0.05)


def _jobs(registry: JobRegistry, store: Store) -> list[dict[str, Any]]:
    return [raw for _key, raw in store.items(prefix=registry._prefix) if isinstance(raw, dict)]


class _SuccessEnvelope:
    ok = True

    def __init__(self, text: str = "delegate finished") -> None:
        self._text = text

    def text(self) -> str:
        return self._text


def _install_fake_agent(monkeypatch: pytest.MonkeyPatch, *, engines: list[Any] | None = None) -> None:
    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            self.engine = engine
            if engines is not None:
                engines.append(engine)

        async def run(self, objective: str) -> _SuccessEnvelope:
            return _SuccessEnvelope(f"did: {objective}")

    monkeypatch.setattr("lazybridge.Agent", FakeAgent)


@pytest.mark.asyncio
async def test_track_keeps_a_strong_reference_until_completion() -> None:
    background_tasks: set[asyncio.Task[Any]] = set()
    ran = False

    async def slow() -> None:
        nonlocal ran
        await asyncio.sleep(0.05)
        ran = True

    _track(background_tasks, slow())
    assert len(background_tasks) == 1
    gc.collect()
    await asyncio.sleep(0.1)
    assert ran is True
    assert background_tasks == set()


def test_safe_notify_swallows_exceptions_and_none_is_noop() -> None:
    def broken(_text: str) -> None:
        raise RuntimeError("notification transport is down")

    _safe_notify(broken, "hello")
    _safe_notify(None, "hello")


class _FakeConsultantTool:
    def __init__(self, id_field: str) -> None:
        self.calls: list[dict[str, Any]] = []
        if id_field == "thread_id":

            async def ask(question: str, thread_id: str | None = None) -> str:
                self.calls.append({"question": question, "thread_id": thread_id})
                return f"[label] answer thread_id=handle-{len(self.calls)}"

        else:

            async def ask(question: str, session_id: str | None = None) -> str:
                self.calls.append({"question": question, "session_id": session_id})
                return f"[label] answer session_id=handle-{len(self.calls)}"

        self.func = ask


@pytest.mark.asyncio
async def test_persistent_consultant_reuses_handle_and_notifies_twice() -> None:
    store = Store()
    registry = JobRegistry(store)
    fake = _FakeConsultantTool("thread_id")
    notified: list[str] = []
    tool = make_persistent_consultant(
        lambda: fake,
        tool_name="ask_codex",
        registry=registry,
        background_tasks=set(),
        notify=notified.append,
        doc_suffix="Runs in the background.",
    )

    started = await tool.func("first question")
    assert "Started job" in started
    assert len(notified) == 1
    await _drain()
    assert fake.calls[0]["thread_id"] is None
    assert len(notified) == 2

    await tool.func("second question")
    await _drain()
    assert fake.calls[1]["thread_id"] == "handle-1"


@pytest.mark.asyncio
async def test_persistent_consultant_detects_session_id() -> None:
    fake = _FakeConsultantTool("session_id")
    tool = make_persistent_consultant(
        lambda: fake,
        tool_name="ask_claude",
        registry=JobRegistry(Store()),
        background_tasks=set(),
    )
    await tool.func("question")
    await _drain()
    assert fake.calls == [{"question": "question", "session_id": None}]


def test_persistent_consultant_rejects_unknown_handle_field() -> None:
    class Broken:
        async def ask(self, question: str) -> str:
            return question

    broken = Broken()
    broken.func = broken.ask
    with pytest.raises(TypeError, match="neither a thread_id nor a session_id"):
        make_persistent_consultant(
            lambda: broken,
            tool_name="broken",
            registry=JobRegistry(Store()),
            background_tasks=set(),
        )


@pytest.mark.asyncio
async def test_background_delegate_approval_and_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)

    async def approve(_objective: str) -> bool:
        return True

    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        pre_confirm=approve,
    )
    immediate = await tool.func("do it")
    assert "Requested approval" in immediate
    assert _jobs(registry, store)[0]["status"] == "awaiting_approval"
    await _drain()
    assert _jobs(registry, store)[0]["status"] == "done"

    async def deny(_objective: str) -> bool:
        return False

    denied_store = Store()
    denied_registry = JobRegistry(denied_store)
    denied = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=lambda: pytest.fail("engine must not be constructed"),
        registry=denied_registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        pre_confirm=deny,
    )
    await denied.func("do not do it")
    await _drain()
    assert _jobs(denied_registry, denied_store)[0]["status"] == "denied"


@pytest.mark.asyncio
async def test_parallel_delegate_uses_fresh_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    engines: list[Any] = []
    _install_fake_agent(monkeypatch, engines=engines)
    store = Store()
    registry = JobRegistry(store)
    built: list[object] = []

    def engine_factory() -> object:
        engine = object()
        built.append(engine)
        return engine

    tool = make_parallel_delegate(
        engine_factory=engine_factory,
        registry=registry,
        background_tasks=set(),
        doc="parallel",
    )
    result = await tool.func(["one", "two", "three"])
    assert "Started 3 job(s)" in result and "not blocking" in result
    await _drain()

    assert len(built) == 3
    assert len({id(engine) for engine in built}) == 3
    assert engines == built
    jobs = _jobs(registry, store)
    assert {job["result"] for job in jobs} == {"did: one", "did: two", "did: three"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("objectives", "tasks", "per_call", "in_flight"),
    [([], set(), 8, 12), (["a", "b", "c"], set(), 2, 12), (["a"], {object()}, 8, 1)],
)
async def test_parallel_delegate_rejects_limits(
    objectives: list[str], tasks: set[Any], per_call: int, in_flight: int
) -> None:
    store = Store()
    tool = make_parallel_delegate(
        engine_factory=object,
        registry=JobRegistry(store),
        background_tasks=tasks,
        doc="parallel",
        max_parallel_objectives=per_call,
        max_in_flight_delegate_tasks=in_flight,
    )
    result = await tool.func(objectives)
    assert result.startswith("REJECTED")
    assert store.items(prefix="delegation:job:") == []


def test_default_engine_factory_requires_workspace_and_gate() -> None:
    with pytest.raises(ValueError, match="workspace_root and gate"):
        make_parallel_delegate(registry=JobRegistry(Store()), background_tasks=set(), doc="parallel")


def test_parallel_and_plan_delegate_tools_are_coroutine_functions() -> None:
    """Tool dispatches a synchronous function's body through
    loop.run_in_executor -- a worker thread with no running event loop,
    where asyncio.create_task (called by _track) would raise. Calling
    tool.func(...) directly in a test (as the other tests here do) runs
    it on the SAME thread as the test itself, which masks this: only the
    real Tool.run()/executor dispatch path exposes it. Asserting
    iscoroutinefunction directly is what actually pins the fix, since it
    checks the same property Tool's own dispatch branches on. Found by
    Codex review before this ever shipped."""
    parallel_tool = make_parallel_delegate(
        engine_factory=object, registry=JobRegistry(Store()), background_tasks=set(), doc="parallel"
    )
    assert inspect.iscoroutinefunction(parallel_tool.func)

    board = DurableBlackboard(Store(), "plan-a")
    board.set_plan("work", ["task one"])
    plan_tool = make_plan_delegate(
        engine_factory=object,
        registry=JobRegistry(Store()),
        background_tasks=set(),
        board=board,
        owner="agent:test",
    )
    assert inspect.iscoroutinefunction(plan_tool.func)


@pytest.mark.asyncio
async def test_background_delegate_pre_confirm_failure_terminates_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception raised by pre_confirm (approval channel disconnected,
    timed out, ...) must still leave the job at a TERMINAL status -- the
    initial write already left it "awaiting_approval", and nothing else
    ever revisits a background job outside _run_job, so an uncaught
    exception here would show it stuck "awaiting_approval" forever even
    though no approval is actually pending anymore. Found by Codex review
    before this ever shipped."""
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    notified: list[str] = []

    async def broken_confirm(_objective: str) -> bool:
        raise RuntimeError("approval channel disconnected")

    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=notified.append,
        doc="delegate",
        pre_confirm=broken_confirm,
    )
    await tool.func("do it")
    await _drain()

    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert "approval channel disconnected" in job["error"]
    assert any("FAILED before it started" in n for n in notified)


@pytest.mark.asyncio
async def test_plan_delegate_claims_starts_and_preserves_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["task one"])
    owner = "agent:test"
    tool = make_plan_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        board=board,
        owner=owner,
    )

    result = await tool.func([{"task_index": 0, "expected_text": "task one", "objective": "implement it"}])
    assert "started job" in result
    claimed = board.snapshot().tasks[0]
    assert claimed["status"] == "claimed" and claimed["owner"] == owner
    running = _jobs(registry, store)[0]
    assert running["plan_id"] == "plan-a"
    assert running["task_index"] == 0
    assert running["plan_task_text"] == "task one"

    await _drain()
    finished = _jobs(registry, store)[0]
    assert finished["status"] == "done"
    closed = board.mark_done(0, "reviewed delegate result", owner=owner)
    assert not closed.startswith("REJECTED")
    assert board.snapshot().tasks[0]["status"] == "done"


@pytest.mark.asyncio
async def test_plan_delegate_mixed_batch_continues_after_rejections(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["task one", "task two"])
    tool = make_plan_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        board=board,
        owner="agent:test",
    )
    result = await tool.func(
        [
            {"task_index": True, "expected_text": "task one", "objective": "invalid"},
            {"task_index": 0, "expected_text": "stale text", "objective": "stale"},
            {"task_index": 1, "expected_text": "task two", "objective": "valid"},
        ]
    )
    assert "bool is not accepted" in result
    assert "does not match expected_text" in result
    assert "started job" in result and "task 1" in result
    assert board.snapshot().tasks[0]["status"] == "todo"
    assert board.snapshot().tasks[1]["status"] == "claimed"
    await _drain()


@pytest.mark.asyncio
async def test_plan_delegate_capacity_rejection_makes_no_partial_claims() -> None:
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["one", "two"])
    tool = make_plan_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks={object()},
        board=board,
        owner="agent:test",
        max_in_flight_delegate_tasks=2,
    )
    result = await tool.func(
        [
            {"task_index": 0, "expected_text": "one", "objective": "first"},
            {"task_index": 1, "expected_text": "two", "objective": "second"},
        ]
    )
    assert result.startswith("REJECTED")
    assert all(task["status"] == "todo" for task in board.snapshot().tasks)
    assert _jobs(registry, store) == []


@pytest.mark.asyncio
async def test_plan_delegate_setup_failure_releases_claim_and_fails_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["task one"])

    def broken_track(_tasks: set[Any], _coroutine: Any) -> None:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("lazybridge.ext.delegation.background._track", broken_track)
    tool = make_plan_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        board=board,
        owner="agent:test",
    )
    result = await tool.func([{"task_index": 0, "expected_text": "task one", "objective": "do it"}])

    assert "setup failed" in result and "scheduler unavailable" in result
    task = board.snapshot().tasks[0]
    assert task["status"] == "todo" and task["owner"] is None
    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert "scheduler unavailable" in job["error"]
