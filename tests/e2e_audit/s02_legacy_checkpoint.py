"""Exercise 1.8.10 checkpoints without owner fields."""

import asyncio
import logging
import sys
import time
from pathlib import Path

from _harness import scenario

from lazybridge import Agent, Plan, Step, Store
from lazybridge.engines.plan import ConcurrentPlanRunError
from lazybridge.testing import MockAgent


async def check(directory):
    with Store(db=str(directory / "legacy.sqlite")) as store:
        worker = MockAgent("finished", name="only")

        def plan(**kwargs):
            return Agent(
                engine=Plan(Step(worker, name="only"), store=store, checkpoint_key="checkpoint", resume=True, **kwargs),
                name="plan",
            )

        recent = {"status": "running", "next_step": "only", "run_uid": "legacy", "updated_at": time.time()}
        store.write("checkpoint", recent)
        try:
            await plan().run("work")
        except ConcurrentPlanRunError as exc:
            assert "unverified active owner" in str(exc)
        else:
            raise AssertionError("recent legacy checkpoint adopted")
        store.write("checkpoint", {**recent, "updated_at": time.time() - 901})
        messages = []

        class Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        logger = logging.getLogger("lazybridge.engines.plan._checkpoint")
        handler = Capture()
        logger.addHandler(handler)
        try:
            assert (await plan().run("work")).ok
        finally:
            logger.removeHandler(handler)
        assert any("Adopting unowned active checkpoint" in message for message in messages)
        store.write("checkpoint", recent)
        assert (await plan(adopt_unowned=True).run("work")).ok
        completed = store.read("checkpoint")
        for key in ("owner_pid", "owner_boot_id", "owner_start_time"):
            completed.pop(key, None)
        store.write("checkpoint", completed)
        count = worker.call_count
        assert (await plan().run("work")).ok
        assert worker.call_count == count, "done checkpoint reran worker"
    return "Recent refused; stale adopted with warning; override adopted; done resumed from cache."


if __name__ == "__main__":
    sys.exit(
        scenario(
            Path(__file__).stem,
            "A4 legacy checkpoint",
            "Refuse recent unowned; warn and adopt stale or explicit override; reuse done.",
            lambda directory: asyncio.run(check(directory)),
        )
    )
