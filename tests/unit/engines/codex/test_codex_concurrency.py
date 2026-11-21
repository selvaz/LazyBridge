"""Process-wide cap on concurrent Codex App Servers (D126)."""

from __future__ import annotations

import asyncio

import pytest

from lazybridge.engines.codex import app_server


@pytest.fixture(autouse=True)
def _reset_cap(monkeypatch):
    monkeypatch.delenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", raising=False)
    app_server.set_codex_max_concurrency(None)
    yield
    app_server.set_codex_max_concurrency(None)


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
    app_server.set_codex_max_concurrency(None)
    monkeypatch.setenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", "2")
    app_server._concurrency_limit = None
    app_server._concurrency_sem = None
    assert asyncio.run(_peak(5)) == 2


def test_invalid_env_is_ignored(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", "lots")
    app_server._concurrency_limit = None
    app_server._concurrency_sem = None
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
