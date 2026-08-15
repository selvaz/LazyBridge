"""Unit tests for the durable blackboard.

The point of this planner is what happens *between* runs, so most of these
tests kill and rebuild the object (and the ``Store`` handle) rather than
exercising one long-lived instance.
"""

from __future__ import annotations

import asyncio

import pytest

from lazybridge import Agent, Envelope, Store
from lazybridge.ext.planners import DurableBlackboard, durable_blackboard_agent
from lazybridge.testing import MockAgent

TASKS = ["gather the filings", "extract the numbers", "write the memo"]


def _board(store: Store, **kwargs) -> DurableBlackboard:
    return DurableBlackboard(store, "quarterly-memo", **kwargs)


def test_plan_survives_a_new_object_over_the_same_store():
    store = Store()
    _board(store).set_plan("quarterly review", TASKS)

    # A fresh object with no shared memory — the restart case.
    reopened = _board(store)
    claimed = reopened.claim_next(owner="worker-1")
    reopened.mark_done(0, "pulled 4 filings", owner="worker-1")

    assert claimed == (0, "gather the filings")
    assert [t["status"] for t in _board(store).snapshot().tasks] == ["done", "todo", "todo"]


def test_plan_survives_a_reopened_sqlite_store(tmp_path):
    db = str(tmp_path / "planner.sqlite")
    with Store(db=db) as first:
        _board(first).set_plan("quarterly review", TASKS)
        _board(first).claim_next(owner="worker-1")
        _board(first).mark_done(0, "pulled 4 filings", owner="worker-1")

    # Process restart: new Store handle onto the same file.
    with Store(db=db) as second:
        snapshot = _board(second).snapshot()

    assert [t["status"] for t in snapshot.tasks] == ["done", "todo", "todo"]
    assert snapshot.tasks[0]["result"] == "pulled 4 filings"


def test_set_plan_refuses_to_discard_work_in_progress():
    store = Store()
    board = _board(store)
    board.set_plan("first attempt", TASKS)
    board.claim_next(owner="worker-1")
    board.mark_done(0, "done already")

    refusal = board.set_plan("second thoughts", ["start over"])

    assert refusal.startswith("REJECTED")
    assert [t["text"] for t in board.snapshot().tasks] == TASKS
    # ...but an explicit replace still works.
    board.set_plan("second thoughts", ["start over"], replace=True)
    assert [t["text"] for t in board.snapshot().tasks] == ["start over"]


def test_a_finished_plan_can_be_replaced_without_the_flag():
    store = Store()
    board = _board(store)
    board.set_plan("first", ["only task"])
    board.claim_next()
    board.mark_done(0, "finished")

    assert not board.set_plan("next job", ["new task"]).startswith("REJECTED")
    assert [t["text"] for t in board.snapshot().tasks] == ["new task"]


def test_two_workers_never_get_the_same_task():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)

    first = board.claim_next(owner="worker-1")
    second = board.claim_next(owner="worker-2")
    third = board.claim_next(owner="worker-3")
    fourth = board.claim_next(owner="worker-4")

    assert [first, second, third] == [(0, TASKS[0]), (1, TASKS[1]), (2, TASKS[2])]
    assert fourth is None  # everything is claimed; nothing is handed out twice


def test_an_abandoned_claim_is_reclaimed_after_its_lease():
    """The worker died mid-task: nobody will ever call mark_done for it."""
    store = Store()
    board = _board(store, lease_seconds=0.05)
    board.set_plan("crash test", ["the task that kills the worker"])

    assert board.claim_next(owner="doomed") == (0, "the task that kills the worker")
    assert board.claim_next(owner="next-worker") is None  # lease still valid

    import time

    time.sleep(0.06)

    assert board.claim_next(owner="next-worker") == (0, "the task that kills the worker")
    assert board.snapshot().tasks[0]["attempts"] == 2


def test_a_task_that_keeps_failing_is_parked_instead_of_looping():
    store = Store()
    board = _board(store, max_attempts=2)
    board.set_plan("poison", ["always fails", "fine"])

    board.claim_next(owner="w")
    board.mark_failed(0, "boom")
    board.claim_next(owner="w")
    board.mark_failed(0, "boom again")

    task = board.snapshot().tasks[0]
    assert task["status"] == "failed"
    assert task["attempts"] == 2
    # The rest of the plan still runs, and the plan can reach completion.
    assert board.claim_next(owner="w") == (1, "fine")
    board.mark_done(1, "did the good one")
    assert board.snapshot().complete


def test_a_stale_worker_cannot_tick_a_reassigned_task():
    store = Store()
    board = _board(store, lease_seconds=0.05)
    board.set_plan("handover", ["long task"])
    board.claim_next(owner="slow-worker")

    import time

    time.sleep(0.06)
    board.claim_next(owner="fresh-worker")

    refusal = board.mark_done(0, "I finally finished", owner="slow-worker")

    assert refusal.startswith("REJECTED")
    assert board.snapshot().tasks[0]["status"] == "claimed"


def test_render_tells_the_agent_where_it_is():
    store = Store()
    board = _board(store)
    assert "no plan yet" in board.render()

    board.set_plan("reasoning here", TASKS)
    board.claim_next(owner="w")
    board.mark_done(0, "first done")

    rendered = board.render()

    assert "[x] gather the filings" in rendered
    assert "→ first done" in rendered
    assert "next claimable: 1" in rendered


def test_marking_a_missing_or_out_of_range_task_is_refused():
    store = Store()
    board = _board(store)

    assert board.mark_done(0, "nothing here").startswith("REJECTED")

    board.set_plan("small", ["only one"])

    assert board.mark_done(5, "wrong index").startswith("REJECTED")
    assert board.mark_done(0, "   ").startswith("REJECTED")


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


class _ToolCallingEngine:
    """Engine stand-in that runs a scripted list of tool calls."""

    model = "mock-model"

    def __init__(self, script):
        self.script = script
        self.seen: list[str] = []

    async def run(self, env, *, tools, **kwargs):
        by_name = {t.name: t for t in tools}
        for tool_name, arguments in self.script:
            self.seen.append(str(await by_name[tool_name].run(**arguments)))
        return Envelope(task=env.task, payload=self.seen[-1] if self.seen else "")

    async def stream(self, env, *, tools, **kwargs):  # pragma: no cover - unused
        yield ""


def _worker() -> MockAgent:
    return MockAgent(lambda env: f"WORKED[{env.text()[:40]}]", name="worker", description="does the task")


def test_the_agent_does_not_reset_the_plan_between_runs():
    """The ephemeral blackboard resets on every run; this one must not."""
    store = Store()
    plan_engine = _ToolCallingEngine([("set_plan", {"reasoning": "because", "tasks": TASKS})])
    agent = durable_blackboard_agent([_worker()], store=store, plan_id="p1", engine=plan_engine)

    asyncio.run(agent.run("plan the work"))

    work_engine = _ToolCallingEngine([("claim_next", {}), ("mark_done", {"task_index": 0, "result_summary": "ok"})])
    second_run = durable_blackboard_agent([_worker()], store=store, plan_id="p1", engine=work_engine)
    asyncio.run(second_run.run("do the next thing"))

    tasks = DurableBlackboard(store, "p1").snapshot().tasks
    assert [t["status"] for t in tasks] == ["done", "todo", "todo"]
    assert "claimed task 0" in work_engine.seen[0]


def test_the_agent_exposes_the_blackboard_verbs_alongside_sub_agents():
    store = Store()
    agent = durable_blackboard_agent([_worker()], store=store, plan_id="p1", engine=_ToolCallingEngine([]))

    tool_names = set(agent._tool_map)

    assert {"set_plan", "get_plan", "claim_next", "mark_done", "mark_failed", "worker"} <= tool_names


def test_duplicate_sub_agent_names_are_rejected():
    store = Store()
    with pytest.raises(ValueError, match="unique names"):
        durable_blackboard_agent([_worker(), _worker()], store=store, plan_id="p1", engine=_ToolCallingEngine([]))


def test_an_empty_agent_list_is_rejected():
    store = Store()
    with pytest.raises(ValueError, match="must not be empty"):
        durable_blackboard_agent([], store=store, plan_id="p1", engine=_ToolCallingEngine([]))


def test_plan_id_and_limits_are_validated():
    store = Store()
    with pytest.raises(ValueError, match="plan_id"):
        DurableBlackboard(store, "")
    with pytest.raises(ValueError, match="lease_seconds"):
        DurableBlackboard(store, "p", lease_seconds=0)
    with pytest.raises(ValueError, match="max_attempts"):
        DurableBlackboard(store, "p", max_attempts=0)


def test_isinstance_of_agent():
    store = Store()
    agent = durable_blackboard_agent([_worker()], store=store, plan_id="p1", engine=_ToolCallingEngine([]))
    assert isinstance(agent, Agent)
