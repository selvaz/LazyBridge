from __future__ import annotations

import asyncio
import contextlib
import gc
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from lazybridge import Store, Tool
from lazybridge.ext.delegation.background import (
    ExtraParam,
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


class _Metadata:
    def __init__(self, cost_usd: float = 0.0, nested_cost_usd: float = 0.0) -> None:
        self.cost_usd = cost_usd
        self.nested_cost_usd = nested_cost_usd


class _CostedEnvelope:
    ok = True

    def __init__(self, text: str = "done", cost_usd: float = 0.0, nested_cost_usd: float = 0.0) -> None:
        self._text = text
        self.metadata = _Metadata(cost_usd, nested_cost_usd)

    def text(self) -> str:
        return self._text


class _EngineStub:
    """Stand-in for ClaudeCodeEngine/CodexEngine -- both real engines expose
    plain ``model``/``reasoning_effort`` attributes that ``_engine_identity``
    reads off whatever ``engine_factory()`` built."""

    def __init__(self, model: str | None = "sonnet", effort: str | None = None) -> None:
        self.model = model
        self.reasoning_effort = effort


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


@pytest.mark.asyncio
async def test_run_delegate_job_records_engine_model_cost_and_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            self.engine = engine

        async def run(self, objective: str) -> _CostedEnvelope:
            return _CostedEnvelope(f"did: {objective}", cost_usd=0.12, nested_cost_usd=0.03)

    monkeypatch.setattr("lazybridge.Agent", FakeAgent)
    store = Store()
    registry = JobRegistry(store)
    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=lambda: _EngineStub(model="sonnet", effort="high"),
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
    )
    await tool.func("do it")
    await _drain()

    [job] = _jobs(registry, store)
    assert job["status"] == "done"
    assert job["engine"] == "_EngineStub"
    assert job["model"] == "sonnet"
    assert job["effort"] == "high"
    assert job["execution_started"] is True
    assert job["cost_usd"] == pytest.approx(0.15)
    assert job["cost_unknown"] is False
    assert job["created_at"]
    assert job["finished_at"]


@pytest.mark.asyncio
async def test_run_delegate_job_keeps_cost_unknown_on_a_worker_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class RaisingAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            pass

        async def run(self, objective: str) -> Any:
            raise RuntimeError("worker crashed")

    monkeypatch.setattr("lazybridge.Agent", RaisingAgent)
    store = Store()
    registry = JobRegistry(store)
    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=_EngineStub,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
    )
    await tool.func("do it")
    await _drain()

    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert "worker crashed" in job["error"]
    assert job["cost_unknown"] is True
    assert "cost_usd" not in job
    assert job["execution_started"] is True


@pytest.mark.asyncio
async def test_validate_model_runs_before_anything_is_recorded_or_spawned() -> None:
    store = Store()
    registry = JobRegistry(store)

    def reject(_model: str | None) -> str | None:
        return "REJECTED: this engine cannot run that model"

    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=lambda: pytest.fail("engine must not be constructed"),
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        model="gpt-5",
        validate_model=reject,
    )
    result = await tool.func("do it")

    assert result == "REJECTED: this engine cannot run that model"
    assert _jobs(registry, store) == []


@pytest.mark.asyncio
async def test_validate_model_allows_through_when_it_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    seen: list[str | None] = []

    def allow(model: str | None) -> str | None:
        seen.append(model)
        return None

    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        model="sonnet",
        validate_model=allow,
    )
    await tool.func("do it")
    await _drain()

    assert seen == ["sonnet"]
    assert _jobs(registry, store)[0]["status"] == "done"


@pytest.mark.asyncio
async def test_admission_gate_refusal_after_approval_fails_the_job_without_starting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_built = False

    def engine_factory() -> Any:
        nonlocal engine_built
        engine_built = True
        return _EngineStub()

    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)

    async def approve(_objective: str) -> bool:
        return True

    async def admission_gate() -> Any:
        return SimpleNamespace(allowed=False, reason="quota exhausted")

    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=engine_factory,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        pre_confirm=approve,
        admission_gate=admission_gate,
    )
    await tool.func("do it")
    await _drain()

    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert "quota exhausted" in job["error"]
    assert job["execution_started"] is False
    assert engine_built is False  # refused before the engine was ever built


@pytest.mark.asyncio
async def test_admission_gate_allows_the_job_to_proceed_after_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    calls = 0

    async def approve(_objective: str) -> bool:
        return True

    async def admission_gate() -> Any:
        nonlocal calls
        calls += 1
        return SimpleNamespace(allowed=True)

    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        pre_confirm=approve,
        admission_gate=admission_gate,
    )
    await tool.func("do it")
    await _drain()

    assert calls == 1
    assert _jobs(registry, store)[0]["status"] == "done"


@pytest.mark.asyncio
async def test_admission_gate_none_is_treated_as_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)

    async def approve(_objective: str) -> bool:
        return True

    async def admission_gate() -> Any:
        return None

    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        pre_confirm=approve,
        admission_gate=admission_gate,
    )
    await tool.func("do it")
    await _drain()

    assert _jobs(registry, store)[0]["status"] == "done"


@pytest.mark.asyncio
async def test_admission_gate_is_not_consulted_without_pre_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """admission_gate only ever re-checks admission AFTER a human approval
    lands -- claude_write's own shape (no pre_confirm) has no such wait to
    re-check anything across, so an admission_gate passed anyway must be a
    silent no-op, never consulted."""
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    calls = 0

    async def admission_gate() -> Any:
        nonlocal calls
        calls += 1
        return SimpleNamespace(allowed=False, reason="should never be asked")

    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        admission_gate=admission_gate,
    )
    await tool.func("do it")
    await _drain()

    assert calls == 0
    assert _jobs(registry, store)[0]["status"] == "done"


@pytest.mark.asyncio
async def test_admission_gate_failure_terminates_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)

    async def approve(_objective: str) -> bool:
        return True

    async def broken_admission_gate() -> Any:
        raise RuntimeError("admission service unreachable")

    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        pre_confirm=approve,
        admission_gate=broken_admission_gate,
    )
    await tool.func("do it")
    await _drain()

    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert "admission service unreachable" in job["error"]
    assert job["execution_started"] is False


@pytest.mark.asyncio
async def test_parallel_delegate_admission_gate_checks_every_objective_before_starting_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    checked: list[int] = []

    async def admission_gate() -> Any:
        checked.append(len(checked))
        # Refuse the SECOND objective -- the whole batch must roll back,
        # nothing started for either objective.
        if len(checked) == 2:
            return SimpleNamespace(allowed=False, reason="quota exhausted")
        return SimpleNamespace(allowed=True)

    tool = make_parallel_delegate(
        engine_factory=lambda: pytest.fail("no engine must be built when the batch is refused"),
        registry=registry,
        background_tasks=set(),
        doc="parallel",
        admission_gate=admission_gate,
    )
    result = await tool.func(["one", "two", "three"])

    assert result.startswith("REJECTED: quota exhausted")
    assert "objective 2 of 3" in result
    assert len(checked) == 2  # stopped at the first refusal, never checked the third
    assert _jobs(registry, store) == []


@pytest.mark.asyncio
async def test_parallel_delegate_admission_gate_allows_every_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    checked: list[str] = []

    async def admission_gate() -> Any:
        checked.append("ok")
        return None  # None means allowed

    tool = make_parallel_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        doc="parallel",
        admission_gate=admission_gate,
    )
    result = await tool.func(["one", "two"])
    await _drain()

    assert "Started 2 job(s)" in result
    assert len(checked) == 2
    jobs = _jobs(registry, store)
    assert {job["status"] for job in jobs} == {"done"}


@pytest.mark.asyncio
async def test_parallel_delegate_admission_gate_rechecks_capacity_before_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capacity check that only ran once, at the very top, left a window
    open across admission_gate's own await for a concurrent caller to also
    pass it and jointly exceed the cap. Re-checking right before scheduling
    narrows that window. Found by Codex review before this ever shipped."""
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    background_tasks: set[asyncio.Task[Any]] = set()

    async def admission_gate() -> Any:
        # Simulate another caller filling up capacity WHILE this call is
        # suspended awaiting admission.
        background_tasks.add(asyncio.ensure_future(asyncio.sleep(10)))
        return SimpleNamespace(allowed=True)

    tool = make_parallel_delegate(
        engine_factory=lambda: pytest.fail("no engine must be built once capacity is exceeded"),
        registry=registry,
        background_tasks=background_tasks,
        doc="parallel",
        max_in_flight_delegate_tasks=1,
        admission_gate=admission_gate,
    )
    result = await tool.func(["one"])

    assert result.startswith("REJECTED")
    assert "capacity was taken by another call" in result
    assert _jobs(registry, store) == []

    for task in list(background_tasks):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            _ = await task


class _FailedEnvelopeWithZeroCost:
    """Stand-in for the shape Envelope.error_envelope() produces: ok is
    False, metadata defaults to all-zero regardless of real spend."""

    ok = False

    def __init__(self, message: str = "boom") -> None:
        self.error = SimpleNamespace(message=message)
        self.metadata = _Metadata(0.0, 0.0)


class _FailedEnvelopeWithRealCost:
    """A failure path that preserves its own envelope's real metadata
    (e.g. an output-validation failure built via model_copy)."""

    ok = False

    def __init__(self, message: str = "boom", cost_usd: float = 0.07) -> None:
        self.error = SimpleNamespace(message=message)
        self.metadata = _Metadata(cost_usd, 0.0)


@pytest.mark.asyncio
async def test_failed_result_with_ambiguous_zero_cost_is_recorded_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            pass

        async def run(self, objective: str) -> _FailedEnvelopeWithZeroCost:
            return _FailedEnvelopeWithZeroCost("engine-level failure")

    monkeypatch.setattr("lazybridge.Agent", FakeAgent)
    store = Store()
    registry = JobRegistry(store)
    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
    )
    await tool.func("do it")
    await _drain()

    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert job["cost_unknown"] is True
    assert "cost_usd" not in job


@pytest.mark.asyncio
async def test_failed_result_with_real_nonzero_cost_is_still_recorded_known(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            pass

        async def run(self, objective: str) -> _FailedEnvelopeWithRealCost:
            return _FailedEnvelopeWithRealCost("output validation failed", cost_usd=0.07)

    monkeypatch.setattr("lazybridge.Agent", FakeAgent)
    store = Store()
    registry = JobRegistry(store)
    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
    )
    await tool.func("do it")
    await _drain()

    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert job["cost_unknown"] is False
    assert job["cost_usd"] == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# Gap 1 -- per-call extra params + guard hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_rejects_before_anything_is_recorded_or_spawned(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    seen: list[tuple[str, dict[str, Any]]] = []

    def guard(objective: str, call_kwargs: dict[str, Any]) -> str | None:
        seen.append((objective, call_kwargs))
        return "REJECTED: not project work"

    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=lambda: pytest.fail("engine must not be built once guard refuses"),
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        guard=guard,
    )
    result = await tool.func("do it")

    assert result == "REJECTED: not project work"
    assert seen == [("do it", {})]
    assert _jobs(registry, store) == []


@pytest.mark.asyncio
async def test_async_guard_is_awaited(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)

    async def guard(objective: str, _call_kwargs: dict[str, Any]) -> str | None:
        await asyncio.sleep(0)
        return None

    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        guard=guard,
    )
    await tool.func("do it")
    await _drain()
    assert _jobs(registry, store)[0]["status"] == "done"


def test_extra_params_reserved_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="reserved names"):
        make_background_delegate(
            tool_name="delegate",
            label="worker",
            engine_factory=object,
            registry=JobRegistry(Store()),
            background_tasks=set(),
            notify=None,
            doc="delegate",
            extra_params={"model": ExtraParam()},
        )


@pytest.mark.asyncio
async def test_extra_params_are_forwarded_to_engine_factory_and_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    built: list[dict[str, Any]] = []
    guarded: list[dict[str, Any]] = []

    def engine_factory(*, repo: str) -> Any:
        built.append({"repo": repo})
        return _EngineStub()

    def guard(_objective: str, call_kwargs: dict[str, Any]) -> str | None:
        guarded.append(dict(call_kwargs))
        return None

    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=engine_factory,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        guard=guard,
        extra_params={"repo": ExtraParam(description="target repo", required=True)},
    )
    result = await tool.func(objective="do it", repo="market-data-hub")
    assert "Started job" in result
    await _drain()

    assert built == [{"repo": "market-data-hub"}]
    assert guarded == [{"repo": "market-data-hub"}]
    assert _jobs(registry, store)[0]["status"] == "done"
    definition = tool.definition()
    assert definition.parameters["required"] == ["objective", "repo"]
    assert definition.parameters["properties"]["repo"]["type"] == "string"
    assert definition.parameters["additionalProperties"] is False


@pytest.mark.asyncio
async def test_disabled_model_override_is_rejected_not_silently_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """With accept_session_override=True but accept_model_override=False,
    the real Python callable is `(objective, **extra)` -- Tool's own
    argument validation lets ANY keyword through, including a "model" the
    caller was never offered. Without an explicit guard, that "model"
    would reach engine_factory unvalidated (validate_model only ever runs
    against the tool's own fixed default, since the override is off).
    Found by Codex review before this ever shipped."""
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)

    def engine_factory(*, session: str | None = None) -> Any:
        pytest.fail("engine must not be built once the undeclared model= is rejected")

    def validate_model(model: str | None) -> str | None:
        if model == "gpt-5":
            return "REJECTED: gpt-5 is not an Anthropic model"
        return None

    tool = make_background_delegate(
        tool_name="claude_write",
        label="a Claude Code sub-agent",
        engine_factory=engine_factory,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        model="sonnet",
        validate_model=validate_model,
        accept_model_override=False,
        extra_params={"session": ExtraParam(description="resume")},
    )

    result = await tool.func(objective="do it", session="resume-me", model="gpt-5")

    assert result.startswith("REJECTED: unexpected argument(s)")
    assert "model" in result
    assert _jobs(registry, store) == []


@pytest.mark.asyncio
async def test_default_delegate_tool_signature_is_unchanged_without_dynamic_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero behaviour change for the pre-1.8 shape: no accept_*_override and
    no extra_params means the generated tool's real Python signature (and
    therefore its JSON Schema) is exactly ``delegate(objective: str)``."""
    _install_fake_agent(monkeypatch)
    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=object,
        registry=JobRegistry(Store()),
        background_tasks=set(),
        notify=None,
        doc="delegate",
    )
    assert list(inspect.signature(tool.func).parameters) == ["objective"]
    assert isinstance(tool, Tool)


# ---------------------------------------------------------------------------
# Gap 2 -- per-call model/effort overrides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accept_model_and_effort_override_resolve_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    built: list[dict[str, Any]] = []
    validated: list[str | None] = []

    def engine_factory(*, model: str | None = None, effort: str | None = None) -> Any:
        built.append({"model": model, "effort": effort})
        return _EngineStub(model=model, effort=effort)

    def validate_model(model: str | None) -> str | None:
        validated.append(model)
        return None

    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=engine_factory,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        model="sonnet",
        effort="low",
        validate_model=validate_model,
        accept_model_override=True,
        accept_effort_override=True,
    )

    # Call 1: no override -- falls back to the tool's own defaults.
    await tool.func(objective="first")
    await _drain()
    # Call 2: per-call override on both.
    await tool.func(objective="second", model="opus", effort="high")
    await _drain()

    assert built == [{"model": "sonnet", "effort": "low"}, {"model": "opus", "effort": "high"}]
    assert validated == ["sonnet", "opus"]
    jobs = sorted(_jobs(registry, store), key=lambda j: j["objective"])
    assert [j["status"] for j in jobs] == ["done", "done"]


@pytest.mark.asyncio
async def test_validate_model_runs_against_the_overridden_model_before_anything_is_recorded() -> None:
    def engine_factory(*, model: str | None = None) -> Any:
        pytest.fail("engine must not be built once validate_model refuses the override")

    def validate_model(model: str | None) -> str | None:
        if model == "gpt-5":
            return f"REJECTED: {model} is not an Anthropic model"
        return None

    store = Store()
    registry = JobRegistry(store)
    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=engine_factory,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        model="sonnet",
        validate_model=validate_model,
        accept_model_override=True,
    )

    result = await tool.func(objective="do it", model="gpt-5")

    assert result == "REJECTED: gpt-5 is not an Anthropic model"
    assert _jobs(registry, store) == []


# ---------------------------------------------------------------------------
# Gap 2 -- persistent consultant: fresh / model / effort overrides
# ---------------------------------------------------------------------------


class _FakeOverridableConsultantTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        async def ask(
            question: str, thread_id: str | None = None, model: str | None = None, effort: str | None = None
        ) -> str:
            self.calls.append({"question": question, "thread_id": thread_id, "model": model, "effort": effort})
            return f"[label] answer thread_id=handle-{len(self.calls)}"

        self.func = ask


@pytest.mark.asyncio
async def test_persistent_consultant_fresh_ignores_stored_handle_but_still_updates_it() -> None:
    fake = _FakeOverridableConsultantTool()
    tool = make_persistent_consultant(
        lambda: fake,
        tool_name="ask_codex",
        registry=JobRegistry(Store()),
        background_tasks=set(),
        accept_fresh=True,
    )

    await tool.func("first question")
    await _drain()
    assert fake.calls[0]["thread_id"] is None

    # Without fresh, the second call continues the remembered handle.
    await tool.func("second question")
    await _drain()
    assert fake.calls[1]["thread_id"] == "handle-1"

    # fresh=True ignores the remembered handle for this call...
    await tool.func("third question", fresh=True)
    await _drain()
    assert fake.calls[2]["thread_id"] is None

    # ...but the new handle it gets back still becomes the one subsequent
    # calls continue.
    await tool.func("fourth question")
    await _drain()
    assert fake.calls[3]["thread_id"] == "handle-3"


@pytest.mark.asyncio
async def test_persistent_consultant_model_and_effort_override_are_forwarded() -> None:
    fake = _FakeOverridableConsultantTool()
    tool = make_persistent_consultant(
        lambda: fake,
        tool_name="ask_codex",
        registry=JobRegistry(Store()),
        background_tasks=set(),
        accept_model_override=True,
        accept_effort_override=True,
    )

    await tool.func("question", model="opus", effort="high")
    await _drain()

    assert fake.calls == [{"question": "question", "thread_id": None, "model": "opus", "effort": "high"}]


@pytest.mark.asyncio
async def test_persistent_consultant_rejects_unsupported_overrides() -> None:
    fake = _FakeOverridableConsultantTool()
    tool = make_persistent_consultant(
        lambda: fake,
        tool_name="ask_codex",
        registry=JobRegistry(Store()),
        background_tasks=set(),
        accept_fresh=True,  # only fresh is accepted -- model/effort are not
    )

    assert "model override" in await tool.func("q", fresh=False, model="opus")
    assert "effort override" in await tool.func("q", effort="high")
    assert fake.calls == []


def test_persistent_consultant_accept_model_override_requires_model_parameter() -> None:
    class NoModelParam:
        def __init__(self) -> None:
            async def ask(question: str, thread_id: str | None = None) -> str:
                return question

            self.func = ask

    with pytest.raises(TypeError, match="accept_model_override=True"):
        make_persistent_consultant(
            NoModelParam,
            tool_name="ask_codex",
            registry=JobRegistry(Store()),
            background_tasks=set(),
            accept_model_override=True,
        )


def test_persistent_consultant_default_signature_is_unchanged() -> None:
    fake = _FakeConsultantTool("thread_id")
    tool = make_persistent_consultant(
        lambda: fake,
        tool_name="ask_codex",
        registry=JobRegistry(Store()),
        background_tasks=set(),
    )
    assert list(inspect.signature(tool.func).parameters) == ["question"]


# ---------------------------------------------------------------------------
# Gap 3 -- admission reservation protocol: release()/refund()
# ---------------------------------------------------------------------------


class _FakeAdmission:
    """A granted or denied admission that records release()/refund() calls.

    ``release``/``refund`` default to sync no-return callables; a test can
    swap in an async variant via the ``async_methods`` flag to prove both
    shapes are supported."""

    def __init__(self, *, allowed: bool = True, reason: str = "quota exhausted", async_methods: bool = False) -> None:
        self.allowed = allowed
        self.reason = reason
        self.released = 0
        self.refunded = 0
        if async_methods:

            async def release() -> None:
                self.released += 1

            async def refund() -> None:
                self.refunded += 1
        else:

            def release() -> None:
                self.released += 1

            def refund() -> None:
                self.refunded += 1

        self.release = release
        self.refund = refund

    def rejection_text(self) -> str:
        return self.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("async_methods", [False, True])
async def test_background_delegate_releases_admission_exactly_once_after_the_job_runs(
    monkeypatch: pytest.MonkeyPatch, async_methods: bool
) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    admission = _FakeAdmission(allowed=True, async_methods=async_methods)

    async def approve(_objective: str) -> bool:
        return True

    async def admission_gate() -> Any:
        return admission

    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=_EngineStub,
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        pre_confirm=approve,
        admission_gate=admission_gate,
    )
    await tool.func("do it")
    await _drain()

    assert _jobs(registry, store)[0]["status"] == "done"
    assert admission.released == 1
    assert admission.refunded == 0


@pytest.mark.asyncio
async def test_background_delegate_refusal_after_approval_does_not_release_a_never_granted_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    refused = _FakeAdmission(allowed=False)

    async def approve(_objective: str) -> bool:
        return True

    async def admission_gate() -> Any:
        return refused

    tool = make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=lambda: pytest.fail("must not be built"),
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        pre_confirm=approve,
        admission_gate=admission_gate,
    )
    await tool.func("do it")
    await _drain()

    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert "quota exhausted" in job["error"]
    # Nothing was ever granted -- release()/refund() must not fire for a
    # denied admission.
    assert refused.released == 0
    assert refused.refunded == 0


@pytest.mark.asyncio
async def test_parallel_delegate_refunds_earlier_grants_on_a_later_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    granted = [_FakeAdmission(allowed=True), _FakeAdmission(allowed=True)]
    refusal = _FakeAdmission(allowed=False)
    decisions = iter([*granted, refusal])

    async def admission_gate() -> Any:
        return next(decisions)

    tool = make_parallel_delegate(
        engine_factory=lambda: pytest.fail("no engine must be built when the batch is refused"),
        registry=registry,
        background_tasks=set(),
        doc="parallel",
        admission_gate=admission_gate,
    )
    result = await tool.func(["one", "two", "three"])

    assert result.startswith("REJECTED: quota exhausted")
    assert all(g.refunded == 1 and g.released == 0 for g in granted)
    assert refusal.refunded == 0  # never granted in the first place
    assert _jobs(registry, store) == []


@pytest.mark.asyncio
async def test_parallel_delegate_refunds_all_grants_on_the_capacity_race(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    background_tasks: set[asyncio.Task[Any]] = set()
    admission = _FakeAdmission(allowed=True)

    async def admission_gate() -> Any:
        background_tasks.add(asyncio.ensure_future(asyncio.sleep(10)))
        return admission

    tool = make_parallel_delegate(
        engine_factory=lambda: pytest.fail("no engine must be built once capacity is exceeded"),
        registry=registry,
        background_tasks=background_tasks,
        doc="parallel",
        max_in_flight_delegate_tasks=1,
        admission_gate=admission_gate,
    )
    result = await tool.func(["one"])

    assert result.startswith("REJECTED")
    assert admission.refunded == 1
    assert admission.released == 0

    for task in list(background_tasks):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            _ = await task


@pytest.mark.asyncio
async def test_refund_and_clear_makes_a_repeated_refund_pass_a_noop() -> None:
    """The real scenario this guards: a cancellation lands between two
    awaits inside a batch-wide refund loop, and the outer ``except
    BaseException`` handler re-runs the SAME loop over the SAME list to
    make sure everything is given back. Without clearing each slot as it
    is refunded, that second pass would refund an admission a second
    time -- crediting a quota or restoring a one-use approval twice for a
    single reservation. Found by Codex review before this ever shipped."""
    from lazybridge.ext.delegation.background import _refund_and_clear

    first, second = _FakeAdmission(allowed=True), _FakeAdmission(allowed=True)
    admissions: list[Any] = [first, second]

    # First pass gets partway through (only index 0) before being
    # interrupted -- exactly what a cancellation between the two awaits
    # would leave behind.
    await _refund_and_clear(admissions, 0)
    assert admissions[0] is None
    assert first.refunded == 1

    # The outer handler re-runs the WHOLE loop regardless of how far the
    # first pass got.
    for i in range(len(admissions)):
        await _refund_and_clear(admissions, i)

    assert first.refunded == 1  # not refunded twice
    assert second.refunded == 1


@pytest.mark.asyncio
async def test_parallel_delegate_refunds_earlier_grants_when_admission_gate_raises_mid_pass() -> None:
    """A raised admission_gate (service unreachable, ...) mid-pass must not
    leak the admissions already granted for earlier objectives in the same
    batch -- same refund discipline as an ordinary refusal, just reached via
    an exception instead of an ``allowed=False`` return. Found by Codex
    review before this ever shipped."""
    store = Store()
    registry = JobRegistry(store)
    granted = [_FakeAdmission(allowed=True), _FakeAdmission(allowed=True)]
    decisions = iter([*granted, RuntimeError("admission service unreachable")])

    async def admission_gate() -> Any:
        decision = next(decisions)
        if isinstance(decision, Exception):
            raise decision
        return decision

    tool = make_parallel_delegate(
        engine_factory=lambda: pytest.fail("no engine must be built once admission_gate raises"),
        registry=registry,
        background_tasks=set(),
        doc="parallel",
        admission_gate=admission_gate,
    )

    with pytest.raises(RuntimeError, match="admission service unreachable"):
        await tool.func(["one", "two", "three"])

    assert all(g.refunded == 1 and g.released == 0 for g in granted)
    assert _jobs(registry, store) == []


@pytest.mark.asyncio
async def test_parallel_delegate_refunds_admission_when_scheduling_one_objective_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-objective scheduling failure (registry.write/_track raising)
    must refund THAT objective's admission rather than hold it for an
    attempt that never ran, and must not abort objectives already
    scheduled earlier in the same batch. Found by Codex review before this
    ever shipped."""
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    admissions = [_FakeAdmission(allowed=True), _FakeAdmission(allowed=True)]
    decisions = iter(admissions)
    real_track = _track
    calls = 0

    def flaky_track(tasks: set[Any], coro: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            coro.close()
            raise RuntimeError("scheduler unavailable")
        return real_track(tasks, coro)

    async def admission_gate() -> Any:
        return next(decisions)

    monkeypatch.setattr("lazybridge.ext.delegation.background._track", flaky_track)
    tool = make_parallel_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        doc="parallel",
        admission_gate=admission_gate,
    )
    result = await tool.func(["one", "two"])
    await _drain()

    assert "Started 1 job(s)" in result
    assert "scheduler unavailable" in result
    jobs = sorted(_jobs(registry, store), key=lambda j: j["objective"])
    assert [j["status"] for j in jobs] == ["done", "failed"]
    # The first objective's job ran and released its admission; the
    # second's scheduling failed before it ever started, so it is refunded.
    assert admissions[0].released == 1 and admissions[0].refunded == 0
    assert admissions[1].refunded == 1 and admissions[1].released == 0


@pytest.mark.asyncio
async def test_parallel_delegate_releases_admission_after_each_job_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    admissions = [_FakeAdmission(allowed=True), _FakeAdmission(allowed=True)]
    decisions = iter(admissions)

    async def admission_gate() -> Any:
        return next(decisions)

    tool = make_parallel_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        doc="parallel",
        admission_gate=admission_gate,
    )
    await tool.func(["one", "two"])
    await _drain()

    assert {job["status"] for job in _jobs(registry, store)} == {"done"}
    assert all(a.released == 1 and a.refunded == 0 for a in admissions)


@pytest.mark.asyncio
async def test_plan_delegate_admission_refusal_claims_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["task one"])
    refused = _FakeAdmission(allowed=False)

    async def admission_gate() -> Any:
        return refused

    tool = make_plan_delegate(
        engine_factory=lambda: pytest.fail("must not be built"),
        registry=registry,
        background_tasks=set(),
        board=board,
        owner="agent:test",
        admission_gate=admission_gate,
    )
    result = await tool.func([{"task_index": 0, "expected_text": "task one", "objective": "implement it"}])

    assert "quota exhausted" in result
    assert board.snapshot().tasks[0]["status"] == "todo"
    assert refused.released == 0
    assert refused.refunded == 0
    assert _jobs(registry, store) == []


@pytest.mark.asyncio
async def test_plan_delegate_refunds_admission_when_the_claim_loses_the_race(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["task one"])
    admission = _FakeAdmission(allowed=True)

    async def admission_gate() -> Any:
        return admission

    tool = make_plan_delegate(
        engine_factory=lambda: pytest.fail("must not be built"),
        registry=registry,
        background_tasks=set(),
        board=board,
        owner="agent:test",
        admission_gate=admission_gate,
    )
    # expected_text does not match -- the claim itself is refused.
    result = await tool.func([{"task_index": 0, "expected_text": "stale text", "objective": "implement it"}])

    assert "does not match expected_text" in result
    assert admission.refunded == 1
    assert admission.released == 0


@pytest.mark.asyncio
async def test_plan_delegate_rechecks_capacity_after_admission_and_refunds_on_the_race() -> None:
    """admission_gate's own await can suspend for real time, during which a
    CONCURRENT delegate_plan_tasks call sharing background_tasks can also
    schedule work -- the top-of-call capacity check alone cannot see that.
    Found by Codex review before this ever shipped."""
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["task one"])
    admission = _FakeAdmission(allowed=True)
    background_tasks: set[asyncio.Task[Any]] = set()

    async def admission_gate() -> Any:
        # Simulate another caller filling up capacity WHILE this call is
        # suspended awaiting admission.
        background_tasks.add(asyncio.ensure_future(asyncio.sleep(10)))
        return admission

    tool = make_plan_delegate(
        engine_factory=lambda: pytest.fail("must not be built once capacity is exceeded"),
        registry=registry,
        background_tasks=background_tasks,
        board=board,
        owner="agent:test",
        max_in_flight_delegate_tasks=1,
        admission_gate=admission_gate,
    )
    result = await tool.func([{"task_index": 0, "expected_text": "task one", "objective": "implement it"}])

    assert "capacity was taken by another call" in result
    assert board.snapshot().tasks[0]["status"] == "todo"  # never claimed
    assert admission.refunded == 1
    assert admission.released == 0

    for task in list(background_tasks):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            _ = await task


@pytest.mark.asyncio
async def test_plan_delegate_refunds_admission_when_claim_task_raises() -> None:
    """``board.claim_task`` raising outright (rather than returning its
    usual "REJECTED: ..." string -- a real DurableBlackboard exhausting its
    CAS retries, for instance) must still refund an already-granted
    admission. Found by Codex review before this ever shipped."""
    store = Store()
    registry = JobRegistry(store)
    admission = _FakeAdmission(allowed=True)

    class BrokenBoard:
        plan_id = "plan-a"

        def claim_task(self, *_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("CAS retries exhausted")

    async def admission_gate() -> Any:
        return admission

    tool = make_plan_delegate(
        engine_factory=lambda: pytest.fail("must not be built"),
        registry=registry,
        background_tasks=set(),
        board=BrokenBoard(),
        owner="agent:test",
        admission_gate=admission_gate,
    )
    result = await tool.func([{"task_index": 0, "expected_text": "task one", "objective": "implement it"}])

    assert "claim_task failed" in result and "CAS retries exhausted" in result
    assert admission.refunded == 1
    assert admission.released == 0


@pytest.mark.asyncio
async def test_plan_delegate_setup_failure_refunds_admission_alongside_the_board_unwind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["task one"])
    admission = _FakeAdmission(allowed=True)

    async def admission_gate() -> Any:
        return admission

    def broken_track(_tasks: set[Any], _coroutine: Any) -> None:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("lazybridge.ext.delegation.background._track", broken_track)
    tool = make_plan_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        board=board,
        owner="agent:test",
        admission_gate=admission_gate,
    )
    result = await tool.func([{"task_index": 0, "expected_text": "task one", "objective": "do it"}])

    assert "setup failed" in result and "scheduler unavailable" in result
    task = board.snapshot().tasks[0]
    assert task["status"] == "todo" and task["owner"] is None
    [job] = _jobs(registry, store)
    assert job["status"] == "failed"
    assert admission.refunded == 1
    assert admission.released == 0


@pytest.mark.asyncio
async def test_plan_delegate_releases_admission_after_the_job_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agent(monkeypatch)
    store = Store()
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan-a")
    board.set_plan("work", ["task one"])
    admission = _FakeAdmission(allowed=True)

    async def admission_gate() -> Any:
        return admission

    tool = make_plan_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        board=board,
        owner="agent:test",
        admission_gate=admission_gate,
    )
    await tool.func([{"task_index": 0, "expected_text": "task one", "objective": "implement it"}])
    await _drain()

    assert _jobs(registry, store)[0]["status"] == "done"
    assert admission.released == 1
    assert admission.refunded == 0


def test_scheduling_primitives_are_public_and_are_the_ones_the_builders_use():
    from lazybridge.ext import delegation
    from lazybridge.ext.delegation import background

    assert delegation.track_background_task is background._track
    assert delegation.schedule_with_admission_release is background._schedule_with_admission_release
