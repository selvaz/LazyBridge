"""Exercise lease renewal and fencing with a separate delegate process."""

import asyncio
import sys
import time
from pathlib import Path

from _harness import child, scenario, scenario_dir, start_trace

from lazybridge import Envelope, Store
from lazybridge.ext.delegation import JobRegistry, make_plan_delegate
from lazybridge.ext.planners.durable_blackboard import DurableBlackboard

NAME = Path(__file__).stem
LEASE = 0.6


async def worker(directory, mode):
    class Engine:
        async def run(self, env, **kwargs):
            (directory / (mode + "-entered")).touch()
            deadline = time.monotonic() + 15
            while not (directory / (mode + "-finish")).exists():
                assert time.monotonic() < deadline
                if mode == "stale":
                    # Model a worker that blocks its process's renewal loop.
                    time.sleep(0.03)
                else:
                    await asyncio.sleep(0.03)
            return Envelope(payload=mode + " result")

    with Store(db=str(directory / "board.sqlite")) as store:
        board = DurableBlackboard(store, mode, lease_seconds=LEASE)
        tasks = set()
        tool = make_plan_delegate(
            engine_factory=Engine,
            registry=JobRegistry(store),
            background_tasks=tasks,
            board=board,
            owner="shared",
            structured_outcomes=True,
        )
        outcome = await tool.run(delegations=[{"task_index": 0, "expected_text": "draft", "objective": "draft"}])
        (directory / (mode + "-job")).write_text(outcome["items"][0]["job_id"])
        await asyncio.gather(*list(tasks))


def wait_for(predicate, process=None, timeout=12):
    deadline = time.monotonic() + timeout
    while not predicate():
        if process is not None:
            assert process.poll() is None, f"worker exited {process.returncode}"
        assert time.monotonic() < deadline, "timed out waiting for worker state"
        time.sleep(0.03)


def run(directory):
    processes = []
    try:
        with Store(db=str(directory / "board.sqlite")) as store:
            boards = {
                name: DurableBlackboard(store, name, lease_seconds=LEASE) for name in ("renew", "stale", "released")
            }
            for board in boards.values():
                board.set_plan("work", ["draft"])
            for mode in ("renew", "stale"):
                process = child(__file__, mode)
                processes.append(process)
                wait_for(
                    lambda mode=mode: (
                        (directory / (mode + "-entered")).exists() and (directory / (mode + "-job")).exists()
                    ),
                    process,
                )
                board = boards[mode]
                job_id = (directory / (mode + "-job")).read_text()
                initial = board.snapshot().tasks[0]["delegation_lease_until"]
                if mode == "renew":
                    wait_for(
                        lambda board=board, initial=initial: (
                            board.snapshot().tasks[0]["delegation_lease_until"] > initial + 0.1
                        ),
                        process,
                    )
                    wait_for(lambda initial=initial: time.time() > initial, process)
                    assert isinstance(board.claim_task(0, "draft", owner="other", delegation_id="other"), str)
                    (directory / (mode + "-finish")).touch()
                    assert process.wait(timeout=10) == 0
                    record = JobRegistry(store).find(job_id)
                    assert record["status"] == "done" and record["result"] == "renew result", record
                    assert not board.mark_done(0, record["result"], owner="shared", delegation_id=job_id).startswith(
                        "REJECTED"
                    )
                else:
                    wait_for(lambda initial=initial: time.time() > initial + 0.03, process)

                    async def replace(board=board):
                        class Replacement:
                            async def run(self, env, **kwargs):
                                return Envelope(payload="replacement result")

                        tasks = set()
                        registry = JobRegistry(store)
                        tool = make_plan_delegate(
                            engine_factory=Replacement,
                            registry=registry,
                            background_tasks=tasks,
                            board=board,
                            owner="shared",
                            structured_outcomes=True,
                        )
                        outcome = await tool.run(
                            delegations=[{"task_index": 0, "expected_text": "draft", "objective": "draft"}]
                        )
                        replacement_id = outcome["items"][0]["job_id"]
                        await asyncio.gather(*list(tasks))
                        assert registry.find(replacement_id)["result"] == "replacement result"
                        return replacement_id

                    replacement_id = asyncio.run(replace())
                    (directory / (mode + "-finish")).touch()
                    assert process.wait(timeout=10) == 0
                    record = JobRegistry(store).find(job_id)
                    assert record["status"] == "failed" and "result" not in record, record
                    assert board.mark_done(0, "stale", owner="shared", delegation_id=job_id).startswith("REJECTED")
                    assert not board.mark_done(0, "new", owner="shared", delegation_id=replacement_id).startswith(
                        "REJECTED"
                    )
            board = boards["released"]
            assert board.claim_task(0, "draft", owner="shared", delegation_id="external") == (0, "draft")
            document = store.read(board.key)
            document["tasks"][0]["delegation_active"] = False
            store.write(board.key, document)
            assert not board.mark_done(0, "reviewed", owner="shared").startswith("REJECTED")
        return (
            "Live lease renewed past original deadline; stale result fenced; replacement and inactive release accepted."
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


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
                "A5 blackboard lease",
                "Renew live lease, fence stale attempt, accept replacement and released claim.",
                run,
            )
        )
