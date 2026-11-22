"""A live checkpoint is protected; a killed owner can be resumed."""

import asyncio
import sys
import time
from pathlib import Path

from _harness import child, scenario, scenario_dir, start_trace

from lazybridge import Agent, Plan, Step, Store
from lazybridge.engines.plan import ConcurrentPlanRunError

NAME = Path(__file__).stem


def plan(directory, resume):
    async def first(task):
        with (directory / "step1.txt").open("a", encoding="utf-8") as stream:
            stream.write("called\n")
        return "first complete"

    async def second(task):
        (directory / "entered").touch()
        deadline = time.monotonic() + 60
        while not (directory / "release").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("release signal not received")
            await asyncio.sleep(0.05)
        return "second complete"

    return Agent(
        engine=Plan(
            Step(first, name="first"),
            Step(second, name="second"),
            store=Store(db=str(directory / "checkpoint.sqlite")),
            checkpoint_key="shared",
            resume=resume,
        ),
        name="plan",
    )


async def worker(directory, mode):
    try:
        result = await plan(directory, mode != "hold").run("work")
        assert mode != "refuse", "live owner was not refused"
        assert result.ok, result.error
    except ConcurrentPlanRunError:
        if mode != "refuse":
            raise


def run(directory):
    owner = child(__file__, "hold")
    try:
        deadline = time.monotonic() + 15
        while not (directory / "entered").exists():
            assert owner.poll() is None, f"owner exited: {owner.returncode}"
            assert time.monotonic() < deadline, "owner did not enter step 2"
            time.sleep(0.05)
        contender = child(__file__, "refuse")
        assert contender.wait(timeout=15) == 0, "live owner was not refused"
        owner.kill()
        owner.wait(timeout=10)
        (directory / "release").touch()
        adopter = child(__file__, "resume")
        assert adopter.wait(timeout=15) == 0, "dead owner was not adopted"
        assert (directory / "step1.txt").read_text().splitlines() == ["called"]
        with Store(db=str(directory / "checkpoint.sqlite")) as store:
            assert store.read("shared")["status"] == "done"
        return "Live owner refused; killed owner adopted; step 1 ran once; checkpoint done."
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=10)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = scenario_dir(NAME)
        stream = start_trace(directory)
        try:
            asyncio.run(worker(directory, sys.argv[1]))
        finally:
            sys.setprofile(None)
            stream.close()
    else:
        sys.exit(
            scenario(
                NAME,
                "A4 live checkpoint takeover",
                "Live owner refused; dead owner resumed without rerunning step 1.",
                run,
            )
        )
