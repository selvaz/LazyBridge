"""Process-wide cap on concurrent Codex App Servers (D126)."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

import pytest

from lazybridge import Envelope, Store
from lazybridge.engines.codex import app_server
from lazybridge.ext.delegation import JobRegistry, make_background_delegate, schedule_with_admission_release


@pytest.fixture(autouse=True)
def _reset_cap(monkeypatch):
    monkeypatch.delenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", raising=False)
    app_server.reset_codex_max_concurrency()
    assert app_server._concurrency_active == 0, "a previous test leaked a Codex slot"
    yield
    app_server.reset_codex_max_concurrency()
    leaked = app_server._concurrency_active
    app_server._concurrency_active = 0
    assert leaked == 0, f"test leaked {leaked} Codex slot(s)"


async def _peak(n_tasks: int) -> int:
    active = peak = 0

    async def job():
        nonlocal active, peak
        async with app_server.codex_concurrency_slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1

    await asyncio.gather(*[job() for _ in range(n_tasks)])
    return peak


def test_unlimited_by_default():
    assert asyncio.run(_peak(4)) == 4


def test_explicit_cap_serialises():
    app_server.set_codex_max_concurrency(1)
    assert asyncio.run(_peak(4)) == 1


def test_env_cap_is_read(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", "2")
    assert asyncio.run(_peak(5)) == 2


def test_invalid_env_is_ignored(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", "lots")
    assert asyncio.run(_peak(3)) == 3


def test_negative_cap_rejected():
    with pytest.raises(ValueError):
        app_server.set_codex_max_concurrency(-1)


def test_waiting_caller_is_cancellable():
    app_server.set_codex_max_concurrency(1)

    async def scenario():
        async def holder():
            async with app_server.codex_concurrency_slot():
                await asyncio.sleep(0.3)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        async def waiter():
            async with app_server.codex_concurrency_slot():
                pass

        w = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        w.cancel()
        with pytest.raises(asyncio.CancelledError):
            await w
        await h
        # the slot is free again after the holder exits
        async with app_server.codex_concurrency_slot():
            pass

    asyncio.run(scenario())


def test_explicit_unlimited_overrides_env(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", "1")
    app_server.set_codex_max_concurrency(None)
    assert asyncio.run(_peak(3)) == 3


def test_nested_agents_never_deadlock_even_two_or_three_levels():
    app_server.set_codex_max_concurrency(1)

    async def level(depth):
        async with app_server.codex_concurrency_slot():
            if depth:
                # a tool callback awaiting two nested agents, each nesting again
                return sum(await asyncio.gather(level(depth - 1), level(depth - 1))) + 1
            return 1

    assert asyncio.run(asyncio.wait_for(level(2), 3)) == 7


def test_stream_slot_released_from_another_task():
    app_server.set_codex_max_concurrency(1)

    async def scenario():
        holder = await app_server.acquire_codex_slot()

        async def producer():
            app_server.enter_codex_slot(holder)
            await asyncio.sleep(0.01)

        await asyncio.create_task(producer())
        await asyncio.create_task(asyncio.to_thread(app_server.release_codex_slot, holder))
        # the only slot is free again
        again = await asyncio.wait_for(app_server.acquire_codex_slot(), 1)
        app_server.release_codex_slot(again)

    asyncio.run(scenario())


def test_background_task_does_not_ride_a_released_slot():
    app_server.set_codex_max_concurrency(1)

    async def scenario():
        late = {}

        async def background():
            await asyncio.sleep(0.1)  # starts after the parent released
            t0 = asyncio.get_running_loop().time()
            async with app_server.codex_concurrency_slot():
                late["waited"] = asyncio.get_running_loop().time() - t0

        async with app_server.codex_concurrency_slot():
            bg = asyncio.create_task(background())
        # an unrelated turn takes the only slot before the background task asks
        async with app_server.codex_concurrency_slot():
            await asyncio.sleep(0.3)
        await bg
        return late["waited"]

    assert asyncio.run(scenario()) >= 0.1


def test_nested_call_reenters_instead_of_deadlocking():
    app_server.set_codex_max_concurrency(1)

    async def scenario():
        async with app_server.codex_concurrency_slot():
            # a tool callback of the running turn starts another Codex agent
            async def nested():
                async with app_server.codex_concurrency_slot():
                    return "ran"

            return await asyncio.wait_for(asyncio.create_task(nested()), 2)

    assert asyncio.run(scenario()) == "ran"


@pytest.mark.parametrize("cap", [1, 2])
async def test_detached_delegate_owns_a_slot_after_parent_exits(cap):
    app_server.set_codex_max_concurrency(cap)
    attempting = asyncio.Event()
    entered = asyncio.Event()
    finish = asyncio.Event()
    request = ContextVar("request", default="missing")
    request.set("delegating request")
    tasks = set()
    store = Store()
    registry = JobRegistry(store)

    class WorkerEngine:
        async def run(self, env, **kwargs):
            attempting.set()
            async with app_server.codex_concurrency_slot():
                entered.set()
                await finish.wait()
                return Envelope(payload=request.get())

    delegate = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=WorkerEngine,
        registry=registry,
        background_tasks=tasks,
        notify=None,
        doc="delegate",
    )

    async def inline():
        async with app_server.codex_concurrency_slot():
            return app_server._concurrency_active

    outsider_attempted = asyncio.Event()
    outsider_entered = asyncio.Event()

    async def unrelated():
        outsider_attempted.set()
        async with app_server.codex_concurrency_slot():
            outsider_entered.set()

    outsider = None
    try:
        async with app_server.codex_concurrency_slot():
            await delegate.run(objective="work in the background")
            await asyncio.wait_for(attempting.wait(), 2)
            assert entered.is_set() is (cap == 2)
            assert await asyncio.wait_for(inline(), 2) == cap
        await asyncio.wait_for(entered.wait(), 2)
        assert app_server._concurrency_active == 1

        outsider = asyncio.create_task(unrelated())
        await asyncio.wait_for(outsider_attempted.wait(), 2)
        assert outsider_entered.is_set() is (cap == 2)
    finally:
        finish.set()
        await asyncio.gather(*tasks)
        if outsider is not None:
            await outsider

    [record] = [value for _, value in store.items(prefix=registry._prefix)]
    assert record["status"] == "done" and record["result"] == "delegating request"


async def test_custom_detached_scheduler_does_not_inherit_the_parent_slot():
    app_server.set_codex_max_concurrency(1)
    attempting = asyncio.Event()
    acquired = asyncio.Event()
    tasks = set()

    def custom_track(background_tasks, coro):
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        return task

    async def detached():
        attempting.set()
        async with app_server.codex_concurrency_slot():
            acquired.set()

    try:
        async with app_server.codex_concurrency_slot():
            schedule_with_admission_release(tasks, detached(), None, track=custom_track)
            await asyncio.wait_for(attempting.wait(), 2)
            assert not acquired.is_set()
    finally:
        await asyncio.gather(*tasks)
    assert acquired.is_set()


def test_lowering_the_cap_blocks_new_turns_until_active_drop():
    app_server.set_codex_max_concurrency(2)

    async def scenario():
        started = []

        async def turn(name, hold):
            async with app_server.codex_concurrency_slot():
                started.append(name)
                await asyncio.sleep(hold)

        a = asyncio.create_task(turn("a", 0.8))
        b = asyncio.create_task(turn("b", 0.8))
        await asyncio.sleep(0.2)
        app_server.set_codex_max_concurrency(1)
        c = asyncio.create_task(turn("c", 0.0))
        await asyncio.sleep(0.3)
        assert started == ["a", "b"]  # c waits: 2 active >= new cap 1
        await asyncio.gather(a, b, c)
        assert started == ["a", "b", "c"]

    asyncio.run(scenario())


def test_queue_time_does_not_consume_the_request_timeout():
    """The engine takes the slot outside its timeouts (engine.py), so a queued turn waits, then runs."""
    import inspect

    from lazybridge.engines.codex import engine

    src = inspect.getsource(engine.CodexEngine)
    assert "async with codex_concurrency_slot():" in src
    i = src.index("async with codex_concurrency_slot():")
    assert "_run_with_idle_timeout" in src[i : i + 200] or "_retry_crashed_turn" in src[i : i + 200]
