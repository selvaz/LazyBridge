"""Check Windows process creation stamps against live and exited processes."""

import asyncio
import os
import sys
import time
from pathlib import Path

from _harness import SkipScenario, child, scenario, scenario_dir, start_trace

from lazybridge import Agent, Plan, Step, Store
from lazybridge._process_owner import _process_start_time, owner_is_alive
from lazybridge.engines.plan import ConcurrentPlanRunError
from lazybridge.ext.delegation import JobRegistry
from lazybridge.testing import MockAgent

NAME = Path(__file__).stem


async def check(directory, process):
    stamp = _process_start_time(process.pid)
    assert stamp is not None, "Windows creation time unavailable"
    live = {"owner_pid": process.pid, "owner_boot_id": "foreign", "owner_start_time": stamp}
    reused = {**live, "owner_start_time": stamp + 1_000_000_000}
    assert owner_is_alive(live) is True and owner_is_alive(reused) is False
    with Store(db=str(directory / "owners.sqlite")) as store:
        registry = JobRegistry(store)
        registry.write("live", "work", tool_name="delegate", status="running", **live)
        registry.write("reused", "work", tool_name="delegate", status="running", **reused)
        assert registry.reclaim_interrupted() == ["reused"]
        assert registry.find("live")["status"] == "running"

        def make_plan():
            return Agent(
                engine=Plan(
                    Step(MockAgent("ok", name="only"), name="only"),
                    store=store,
                    checkpoint_key="checkpoint",
                    resume=True,
                ),
                name="plan",
            )

        checkpoint = {"status": "running", "next_step": "only", "run_uid": "other"}
        store.write("checkpoint", {**checkpoint, **live})
        try:
            await make_plan().run("work")
        except ConcurrentPlanRunError:
            pass
        else:
            raise AssertionError("live process checkpoint adopted")
        store.write("checkpoint", {**checkpoint, **reused})
        assert (await make_plan().run("work")).ok
        process.kill()
        process.wait(timeout=10)
        assert owner_is_alive(live) is False
        assert registry.reclaim_interrupted() == ["live"]
    return "Live creation stamp protected; different stamp and exited PID reclaimed."


def run(directory):
    if os.name != "nt" and _process_start_time(os.getpid()) is None:
        raise SkipScenario("Process creation stamps are unavailable on this platform")
    process = child(__file__, "sleeper")
    try:
        deadline = time.monotonic() + 15
        while not (directory / "sleeper-ready").exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        return asyncio.run(check(directory, process))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = scenario_dir(NAME)
        stream = start_trace(directory)
        (directory / "sleeper-ready").touch()
        try:
            time.sleep(60)
        finally:
            sys.setprofile(None)
            stream.close()
    else:
        sys.exit(
            scenario(
                NAME, "Windows PID reuse", "Reclaim a mismatched or exited owner; protect a matching live owner.", run
            )
        )
