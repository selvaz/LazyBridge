"""Process-wide cap on concurrent Codex App Servers (D126)."""

from __future__ import annotations

import asyncio

import pytest

from lazybridge.engines.codex import app_server


@pytest.fixture(autouse=True)
def _reset_cap(monkeypatch):
    monkeypatch.delenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", raising=False)
    app_server.reset_codex_max_concurrency()
    yield
    app_server.reset_codex_max_concurrency()


async def _peak(n_tasks: int) -> int:
    active = peak = 0

    async def job():
        nonlocal active, peak
        async with app_server._codex_concurrency_slot():
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
            async with app_server._codex_concurrency_slot():
                await asyncio.sleep(0.3)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        async def waiter():
            async with app_server._codex_concurrency_slot():
                pass

        w = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        w.cancel()
        with pytest.raises(asyncio.CancelledError):
            await w
        await h
        # the slot is free again after the holder exits
        async with app_server._codex_concurrency_slot():
            pass

    asyncio.run(scenario())


def test_explicit_unlimited_overrides_env(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", "1")
    app_server.set_codex_max_concurrency(None)
    assert asyncio.run(_peak(3)) == 3


def test_nested_call_reenters_instead_of_deadlocking():
    app_server.set_codex_max_concurrency(1)

    async def scenario():
        async with app_server._codex_concurrency_slot():
            # a tool callback of the running turn starts another Codex agent
            async def nested():
                async with app_server._codex_concurrency_slot():
                    return "ran"

            return await asyncio.wait_for(asyncio.create_task(nested()), 2)

    assert asyncio.run(scenario()) == "ran"


def test_lowering_the_cap_blocks_new_turns_until_active_drop():
    app_server.set_codex_max_concurrency(2)

    async def scenario():
        started = []

        async def turn(name, hold):
            async with app_server._codex_concurrency_slot():
                started.append(name)
                await asyncio.sleep(hold)

        a = asyncio.create_task(turn("a", 0.3))
        b = asyncio.create_task(turn("b", 0.3))
        await asyncio.sleep(0.05)
        app_server.set_codex_max_concurrency(1)
        c = asyncio.create_task(turn("c", 0.0))
        await asyncio.sleep(0.1)
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
