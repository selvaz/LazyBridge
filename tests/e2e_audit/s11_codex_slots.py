"""Detached delegates hold their own Codex slot without an app server."""

import asyncio
import sys
from pathlib import Path

from _harness import scenario

from lazybridge import Envelope, Store
from lazybridge.engines.codex import app_server
from lazybridge.ext.delegation import JobRegistry, make_background_delegate


async def check(directory):
    app_server.set_codex_max_concurrency(1)
    attempting, entered, finish, outsider_entered = (asyncio.Event() for _ in range(4))
    tasks = set()

    class Worker:
        async def run(self, env, **kwargs):
            attempting.set()
            async with app_server.codex_concurrency_slot():
                entered.set()
                await finish.wait()
                return Envelope(payload="done")

    async def outsider():
        async with app_server.codex_concurrency_slot():
            outsider_entered.set()

    with Store(db=str(directory / "jobs.sqlite")) as store:
        registry = JobRegistry(store)
        delegate = make_background_delegate(
            tool_name="delegate",
            label="worker",
            engine_factory=Worker,
            registry=registry,
            background_tasks=tasks,
            notify=None,
            doc="delegate",
        )
        other = None
        try:
            async with app_server.codex_concurrency_slot():
                await delegate.run(objective="work")
                await asyncio.wait_for(attempting.wait(), 5)
                assert not entered.is_set(), "detached worker inherited parent slot"
            await asyncio.wait_for(entered.wait(), 5)
            other = asyncio.create_task(outsider())
            await asyncio.sleep(0)
            assert not outsider_entered.is_set(), "worker did not hold its slot"
        finally:
            finish.set()
            await asyncio.gather(*tasks)
            if other:
                await asyncio.wait_for(other, 5)
        assert outsider_entered.is_set()
        [record] = [value for _, value in store.items(prefix="delegation:job:")]
        assert record["status"] == "done"
    return "Detached worker acquired its own slot and blocked an unrelated turn until completion."


if __name__ == "__main__":
    sys.exit(
        scenario(
            Path(__file__).stem,
            "M4 Codex slots",
            "Detached delegate occupies its own slot until it finishes.",
            lambda d: asyncio.run(check(d)),
        )
    )
