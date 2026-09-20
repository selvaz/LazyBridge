"""Unit tests for the durable blackboard.

The point of this planner is what happens *between* runs, so most of these
tests kill and rebuild the object (and the ``Store`` handle) rather than
exercising one long-lived instance.
"""

from __future__ import annotations

import asyncio
import time

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
# add_tasks / cancel_task (incremental plan revision)
# ---------------------------------------------------------------------------


def test_add_tasks_appends_without_disturbing_existing_indices():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)
    board.claim_next(owner="w")  # index 0 in flight

    board.add_tasks(["a newly discovered task"])

    snapshot = board.snapshot()
    assert [t["text"] for t in snapshot.tasks] == [*TASKS, "a newly discovered task"]
    assert snapshot.tasks[0]["status"] == "claimed"  # untouched by the append
    # The in-flight claim from before the append still closes cleanly.
    assert not board.mark_done(0, "finished").startswith("REJECTED")


def test_add_tasks_is_rejected_without_a_plan():
    store = Store()
    board = _board(store)
    assert board.add_tasks(["x"]).startswith("REJECTED")


def test_add_tasks_rejects_an_empty_or_blank_list():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)
    assert board.add_tasks([]).startswith("REJECTED")
    assert board.add_tasks(["   "]).startswith("REJECTED")
    assert len(board.snapshot().tasks) == len(TASKS)


def test_cancel_task_drops_an_unclaimed_task():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)

    result = board.cancel_task(1, TASKS[1], "no longer needed")

    assert not result.startswith("REJECTED")
    task = board.snapshot().tasks[1]
    assert task["status"] == "cancelled"
    assert task["cancel_reason"] == "no longer needed"


def test_cancel_task_rejects_a_stale_text_mismatch():
    """Guards against the LLM acting on an outdated get_plan() rendering."""
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)

    refusal = board.cancel_task(1, "a task that no longer matches", "stale")

    assert refusal.startswith("REJECTED")
    assert board.snapshot().tasks[1]["status"] == "todo"


def test_cancel_task_refuses_a_claimed_task():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)
    board.claim_next(owner="w")

    refusal = board.cancel_task(0, TASKS[0], "changed my mind")

    assert refusal.startswith("REJECTED")
    assert board.snapshot().tasks[0]["status"] == "claimed"


def test_cancel_task_accepts_a_task_parked_after_exhausting_its_attempts():
    store = Store()
    board = _board(store, max_attempts=1)
    board.set_plan("shared", ["exhausted task"])
    board.claim_next(owner="w")
    board.mark_failed(0, "worker was interrupted", owner="w")
    assert board.snapshot().tasks[0]["status"] == "failed"

    result = board.cancel_task(0, "exhausted task", "replacement task completed the work")

    assert not result.startswith("REJECTED")
    task = board.snapshot().tasks[0]
    assert task["status"] == "cancelled"
    assert task["cancel_reason"] == "replacement task completed the work"


@pytest.mark.parametrize("terminal_status", ["done", "cancelled"])
def test_cancel_task_refuses_done_or_already_cancelled_tasks(terminal_status):
    store = Store()
    board = _board(store)
    board.set_plan("shared", ["terminal task"])
    if terminal_status == "done":
        board.claim_next(owner="w")
        board.mark_done(0, "finished", owner="w")
    else:
        board.cancel_task(0, "terminal task", "first cancellation")

    refusal = board.cancel_task(0, "terminal task", "second cancellation")

    assert refusal.startswith("REJECTED")
    task = board.snapshot().tasks[0]
    assert task["status"] == terminal_status
    assert task["cancel_reason"] == ("first cancellation" if terminal_status == "cancelled" else "")


def test_cancel_task_requires_a_reason():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)
    assert board.cancel_task(0, TASKS[0], "  ").startswith("REJECTED")


def test_a_plan_of_only_cancelled_and_done_tasks_is_complete():
    store = Store()
    board = _board(store)
    board.set_plan("shared", ["do it", "skip it"])
    board.claim_next(owner="w")
    board.mark_done(0, "done")
    board.cancel_task(1, "skip it", "not needed after all")

    assert board.snapshot().complete
    assert "1 cancelled" in board.render()


# ---------------------------------------------------------------------------
# per-task scheduling
# ---------------------------------------------------------------------------


def test_fresh_plan_snapshot_has_no_schedule_events():
    board = _board(Store())
    board.set_plan("unscheduled", ["task"])

    assert board.snapshot().schedule_events == []


def test_snapshot_exposes_schedule_events_in_append_order():
    board = _board(Store())
    board.set_plan("scheduled", ["task"])
    board.set_task_schedule(0, "task", due_at=200.0, reason="first estimate")
    board.set_task_schedule(0, "task", planned_start_at=250.0, due_at=300.0, reason="replanned")

    events = board.snapshot().schedule_events

    assert events == [
        {
            "task_index": 0,
            "planned_start_at": None,
            "due_at": 200.0,
            "reason": "first estimate",
            "at": events[0]["at"],
        },
        {
            "task_index": 0,
            "planned_start_at": 250.0,
            "due_at": 300.0,
            "reason": "replanned",
            "at": events[1]["at"],
        },
    ]
    assert all(isinstance(event["at"], float) for event in events)


def test_snapshot_schedule_events_is_a_defensive_list_copy():
    board = _board(Store())
    board.set_plan("scheduled", ["task"])
    board.set_task_schedule(0, "task", due_at=200.0, reason="deadline")

    returned_events = board.snapshot().schedule_events
    returned_events.clear()

    assert len(board.snapshot().schedule_events) == 1


def test_legacy_tasks_without_schedule_fields_still_support_every_transition():
    store = Store()
    board = _board(store, max_attempts=1)
    board.set_plan("legacy", ["complete", "fail", "cancel"])
    doc = store.read(board.key)
    for task in doc["tasks"]:
        task.pop("planned_start_at")
        task.pop("due_at")
    doc.pop("schedule_events")
    store.write(board.key, doc)

    board.claim_task(0, "complete", owner="w")
    board.mark_done(0, "done", owner="w")
    board.claim_task(1, "fail", owner="w")
    board.mark_failed(1, "failed", owner="w")
    board.cancel_task(2, "cancel", "obsolete")

    assert [task["status"] for task in board.snapshot().tasks] == ["done", "failed", "cancelled"]
    assert "plan complete" in board.render()


def test_schedule_and_history_survive_retry_and_completion():
    store = Store()
    board = _board(store, max_attempts=2)
    board.set_plan("scheduled", ["retry me"])

    result = board.set_task_schedule(0, "retry me", planned_start_at=100.0, due_at=200.0, reason="initial estimate")
    assert not result.startswith("REJECTED")
    board.claim_next(owner="w")
    board.mark_failed(0, "transient", owner="w")
    board.claim_next(owner="w")
    board.mark_done(0, "finished", owner="w")

    task = board.snapshot().tasks[0]
    assert (task["planned_start_at"], task["due_at"], task["status"]) == (100.0, 200.0, "done")
    events = store.read(board.key)["schedule_events"]
    assert len(events) == 1
    assert {key: value for key, value in events[0].items() if key != "at"} == {
        "task_index": 0,
        "planned_start_at": 100.0,
        "due_at": 200.0,
        "reason": "initial estimate",
    }
    assert isinstance(events[0]["at"], float)


def test_schedule_survives_cancellation():
    store = Store()
    board = _board(store)
    board.set_plan("scheduled", ["cancel me"])
    board.set_task_schedule(0, "cancel me", due_at=200.0, reason="deadline")

    board.cancel_task(0, "cancel me", "obsolete")

    task = board.snapshot().tasks[0]
    assert task["status"] == "cancelled"
    assert task["due_at"] == 200.0
    assert len(store.read(board.key)["schedule_events"]) == 1


def test_rescheduling_appends_history_instead_of_replacing_it():
    store = Store()
    board = _board(store)
    board.set_plan("scheduled", ["task"])

    board.set_task_schedule(0, "task", due_at=200.0, reason="first estimate")
    board.set_task_schedule(0, "task", planned_start_at=250.0, due_at=300.0, reason="replanned")

    doc = store.read(board.key)
    assert doc["tasks"][0]["planned_start_at"] == 250.0
    assert doc["tasks"][0]["due_at"] == 300.0
    assert [event["reason"] for event in doc["schedule_events"]] == ["first estimate", "replanned"]
    assert [event["due_at"] for event in doc["schedule_events"]] == [200.0, 300.0]


def test_completion_winning_a_schedule_cas_race_rejects_the_reschedule(tmp_path, monkeypatch):
    db = str(tmp_path / "schedule-race.sqlite")
    with Store(db=db) as scheduling_store, Store(db=db) as completing_store:
        scheduling = _board(scheduling_store)
        completing = _board(completing_store)
        scheduling.set_plan("race", ["task"])
        scheduling.claim_next(owner="worker")
        original_cas = scheduling_store.compare_and_swap
        raced = False

        def complete_before_first_schedule_cas(key, expected, new, *, agent_id=None):
            nonlocal raced
            if not raced:
                raced = True
                assert not completing.mark_done(0, "finished", owner="worker").startswith("REJECTED")
            return original_cas(key, expected, new, agent_id=agent_id)

        monkeypatch.setattr(scheduling_store, "compare_and_swap", complete_before_first_schedule_cas)
        result = scheduling.set_task_schedule(0, "task", due_at=200.0, reason="new deadline")

        assert result.startswith("REJECTED")
        assert "done" in result
        assert scheduling.snapshot().tasks[0]["status"] == "done"
        assert scheduling.snapshot().tasks[0]["due_at"] is None
        assert scheduling_store.read(scheduling.key)["schedule_events"] == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "tomorrow", True])
def test_invalid_schedule_timestamps_are_rejected_without_touching_any_blackboard(value):
    store = Store()
    board = _board(store)
    other = DurableBlackboard(store, "other-project")
    board.set_plan("scheduled", ["first", "second"])
    other.set_plan("unrelated", ["other"])
    before = store.read(board.key)
    other_before = store.read(other.key)

    with pytest.raises(ValueError, match="due_at"):
        board.set_task_schedule(0, "first", due_at=value, reason="bad input")

    assert store.read(board.key) == before
    assert store.read(other.key) == other_before


def test_schedule_rejects_reversed_interval_and_blank_reason_without_writing():
    store = Store()
    board = _board(store)
    board.set_plan("scheduled", ["task"])
    before = store.read(board.key)

    with pytest.raises(ValueError, match="planned_start_at must be <= due_at"):
        board.set_task_schedule(0, "task", planned_start_at=2.0, due_at=1.0, reason="reversed")
    with pytest.raises(ValueError, match="reason"):
        board.set_task_schedule(0, "task", reason="   ")

    assert store.read(board.key) == before


def test_set_task_schedule_rejects_a_stale_text_mismatch():
    store = Store()
    board = _board(store)
    board.set_plan("scheduled", TASKS)

    refusal = board.set_task_schedule(1, "a task that no longer matches", due_at=200.0, reason="deadline")

    assert refusal.startswith("REJECTED")
    assert board.snapshot().tasks[1]["due_at"] is None
    assert store.read(board.key)["schedule_events"] == []


@pytest.mark.parametrize("terminal_status", ["done", "failed", "cancelled"])
def test_render_marks_only_open_past_due_tasks_overdue(terminal_status):
    import time

    store = Store()
    board = _board(store, max_attempts=1)
    terminal_text = f"{terminal_status} overdue"
    board.set_plan("deadlines", ["todo overdue", "claimed overdue", terminal_text])
    past = time.time() - 60
    for index, text in enumerate(["todo overdue", "claimed overdue", terminal_text]):
        board.set_task_schedule(index, text, planned_start_at=past - 60, due_at=past, reason="deadline")
    board.claim_task(1, "claimed overdue", owner="w")
    if terminal_status == "cancelled":
        board.cancel_task(2, terminal_text, "obsolete")
    else:
        board.claim_task(2, terminal_text, owner="w")
        if terminal_status == "done":
            board.mark_done(2, "finished", owner="w")
        else:
            board.mark_failed(2, "boom", owner="w")

    rendered = board.render()
    assert "planned start:" in rendered
    assert "due:" in rendered
    assert rendered.count("OVERDUE") == 2


# --- completed_at: a closed task can finally be placed in time -----------
#
# Found by a real incident: LazyCEO's end-of-day report showed a specialist's
# "recently closed" tasks under today's date, when the closures were from
# days earlier -- there was no way to tell, because no completion timestamp
# existed anywhere in a closed task's record (only claimed_at, which a
# terminal write always clears to None). These tests prove the field a
# consumer would actually need to answer "was this closed today" now exists
# and behaves correctly, not just that it's present.


def test_a_fresh_task_has_no_completed_at():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)
    assert board.snapshot().tasks[0]["completed_at"] is None


def test_mark_done_stamps_completed_at():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)
    board.claim_next(owner="w")

    import time

    before = time.time()
    board.mark_done(0, "finished")
    after = time.time()

    completed_at = board.snapshot().tasks[0]["completed_at"]
    assert completed_at is not None
    assert before <= completed_at <= after


def test_mark_failed_sent_back_to_todo_for_a_retry_is_not_completed():
    """A retry is not a close -- the task is still open, headed back to the
    queue, and completed_at must not claim otherwise."""
    store = Store()
    board = _board(store, max_attempts=5)
    board.set_plan("shared", TASKS)
    board.claim_next(owner="w")

    board.mark_failed(0, "transient error")

    task = board.snapshot().tasks[0]
    assert task["status"] == "todo"
    assert task["completed_at"] is None


def test_mark_failed_exhausted_stamps_completed_at():
    """The attempts-exhausted branch of mark_failed IS a real close -- the
    task is permanently parked, not retried -- and must be stamped like any
    other terminal state."""
    store = Store()
    board = _board(store, max_attempts=1)
    board.set_plan("shared", ["always fails"])
    board.claim_next(owner="w")

    import time

    before = time.time()
    board.mark_failed(0, "boom")
    after = time.time()

    task = board.snapshot().tasks[0]
    assert task["status"] == "failed"
    completed_at = task["completed_at"]
    assert completed_at is not None
    assert before <= completed_at <= after


def test_cancel_task_stamps_completed_at():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)

    import time

    before = time.time()
    board.cancel_task(1, TASKS[1], "no longer needed")
    after = time.time()

    completed_at = board.snapshot().tasks[1]["completed_at"]
    assert completed_at is not None
    assert before <= completed_at <= after


def test_an_exhausted_lease_parked_by_claim_next_stamps_completed_at():
    """The OTHER poison-task park -- reached via claim_next reclaiming an
    expired lease straight into exhaustion, not via mark_failed -- must be
    stamped the same way. A live crash-loop is exactly this path, not the
    mark_failed one."""
    store = Store()
    board = _board(store, lease_seconds=0.05, max_attempts=1)
    board.set_plan("crash test", ["the task that kills the worker"])
    board.claim_next(owner="doomed")

    import time

    time.sleep(0.06)
    before = time.time()
    assert board.claim_next(owner="next-worker") is None  # parked, not handed out
    after = time.time()

    task = board.snapshot().tasks[0]
    assert task["status"] == "failed"
    completed_at = task["completed_at"]
    assert completed_at is not None
    assert before <= completed_at <= after


def test_a_reclaimed_but_not_exhausted_lease_is_not_completed():
    """The lease expired and the task was handed to a new worker -- it is
    still open (now claimed again), not closed, and must not be stamped."""
    store = Store()
    board = _board(store, lease_seconds=0.05, max_attempts=5)
    board.set_plan("crash test", ["the task that kills the worker"])
    board.claim_next(owner="doomed")

    import time

    time.sleep(0.06)
    assert board.claim_next(owner="next-worker") == (0, "the task that kills the worker")

    task = board.snapshot().tasks[0]
    assert task["status"] == "claimed"
    assert task["completed_at"] is None


def test_completed_at_actually_distinguishes_an_old_closure_from_a_fresh_one():
    """The invariant a real consumer needs, not just "the field is set":
    given one task closed BEFORE a cutoff and one closed AFTER it, filtering
    on completed_at must keep only the second -- exactly the LazyCEO
    end-of-day report's "what closed since last night" question. Written to
    fail if completed_at stopped varying per task (e.g. a bug that stamped
    every closure with the SAME timestamp, or none at all) -- a test that
    only checked "completed_at is not None" would pass even then."""
    store = Store()
    board = _board(store, max_attempts=5)
    board.set_plan("shared", ["closed yesterday", "closed today"])

    board.claim_next(owner="w")
    board.mark_done(0, "old work, done a while ago")

    import time

    # A real gap on BOTH sides of the cutoff, not just after it -- the
    # first sleep guarantees the cutoff is strictly later than task 0's
    # completed_at even on a coarse system clock, so the boundary itself
    # is never a tie.
    time.sleep(0.02)
    cutoff = time.time()
    time.sleep(0.02)

    board.claim_next(owner="w")
    board.mark_done(1, "fresh work, done just now")

    tasks = board.snapshot().tasks
    closed_since_cutoff = [t["text"] for t in tasks if t["completed_at"] is not None and t["completed_at"] > cutoff]

    assert closed_since_cutoff == ["closed today"]  # NOT "closed yesterday" too


def test_claim_next_does_not_starve_an_expired_claim_behind_new_todo_tasks():
    """A plan that keeps growing via add_tasks() must not indefinitely delay
    reclaiming an abandoned worker's task just because a fresher todo item
    is always available first in scan order."""
    store = Store()
    board = _board(store, lease_seconds=0.05)
    board.set_plan("shared", ["the abandoned task"])
    board.claim_next(owner="doomed")

    import time

    time.sleep(0.06)
    board.add_tasks(["a brand new task"])  # would starve the expired claim under naive todo-first scanning

    assert board.claim_next(owner="rescuer") == (0, "the abandoned task")


def test_claim_task_takes_a_specific_task_out_of_order():
    """The fix for real friction: reaching task 2 must not require claiming
    (and closing) task 0 and 1 first just to advance claim_next()'s scan."""
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)

    claimed = board.claim_task(2, TASKS[2], owner="w")

    assert claimed == (2, TASKS[2])
    statuses = [t["status"] for t in board.snapshot().tasks]
    assert statuses == ["todo", "todo", "claimed"]
    assert board.snapshot().tasks[2]["owner"] == "w"
    assert board.snapshot().tasks[2]["attempts"] == 1


def test_claim_task_rejects_a_stale_text_mismatch():
    """Same stale-index guard as cancel_task."""
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)

    refusal = board.claim_task(1, "a task that no longer matches", owner="w")

    assert isinstance(refusal, str) and refusal.startswith("REJECTED")
    assert board.snapshot().tasks[1]["status"] == "todo"


def test_claim_task_rejects_out_of_range_index():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)

    refusal = board.claim_task(99, "does not matter", owner="w")

    assert isinstance(refusal, str) and refusal.startswith("REJECTED")


def test_claim_task_refuses_an_already_claimed_unexpired_task():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)
    board.claim_next(owner="first")

    refusal = board.claim_task(0, TASKS[0], owner="second")

    assert isinstance(refusal, str) and refusal.startswith("REJECTED")
    assert board.snapshot().tasks[0]["owner"] == "first"


def test_claim_task_refuses_a_done_task():
    store = Store()
    board = _board(store)
    board.set_plan("shared", TASKS)
    board.claim_next(owner="w")
    board.mark_done(0, "already finished", owner="w")

    refusal = board.claim_task(0, TASKS[0], owner="w")

    assert isinstance(refusal, str) and refusal.startswith("REJECTED")


def test_claim_task_reclaims_an_expired_lease_like_claim_next_does():
    store = Store()
    board = _board(store, lease_seconds=0.05)
    board.set_plan("shared", TASKS)
    board.claim_next(owner="doomed")

    import time

    time.sleep(0.06)
    claimed = board.claim_task(0, TASKS[0], owner="rescuer")

    assert claimed == (0, TASKS[0])
    assert board.snapshot().tasks[0]["owner"] == "rescuer"
    assert board.snapshot().tasks[0]["attempts"] == 2  # doomed's attempt still counted


def test_claim_task_parks_an_exhausted_task_as_failed_instead_of_reclaiming_it():
    store = Store()
    board = _board(store, lease_seconds=0.05, max_attempts=1)
    board.set_plan("shared", ["only task"])
    board.claim_next(owner="doomed")  # consumes the only attempt

    import time

    time.sleep(0.06)
    refusal = board.claim_task(0, "only task", owner="rescuer")

    assert isinstance(refusal, str) and refusal.startswith("REJECTED")
    assert board.snapshot().tasks[0]["status"] == "failed"


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

    assert {
        "set_plan",
        "get_plan",
        "add_tasks",
        "cancel_task",
        "set_task_schedule",
        "claim_next",
        "claim_task",
        "mark_done",
        "mark_failed",
        "worker",
    } <= tool_names


def test_claim_task_tool_passes_through_a_successful_claim_and_a_rejection():
    store = Store()
    DurableBlackboard(store, "p1").set_plan("shared", TASKS)
    agent = durable_blackboard_agent([_worker()], store=store, plan_id="p1", engine=_ToolCallingEngine([]))

    ok = asyncio.run(agent._tool_map["claim_task"].run(task_index=2, expected_text=TASKS[2]))
    assert "claimed task 2" in str(ok)

    rejected = asyncio.run(agent._tool_map["claim_task"].run(task_index=1, expected_text="wrong text"))
    assert str(rejected).startswith("REJECTED")


def test_the_claim_task_tool_can_renew_a_claim_its_own_agent_holds_once():
    """The recipe documents claim_task(index, expected_text, renew=False) as
    the agent's tool. The wrapper took only two arguments, so a model that
    followed the recipe got a validation error instead of the re-entry."""
    store = Store()
    DurableBlackboard(store, "p1").set_plan("shared", TASKS)
    agent = durable_blackboard_agent([_worker()], store=store, plan_id="p1", engine=_ToolCallingEngine([]))
    claim = agent._tool_map["claim_task"]

    first = asyncio.run(claim.run(task_index=0, expected_text=TASKS[0]))
    assert "claimed task 0" in str(first)
    assert str(asyncio.run(claim.run(task_index=0, expected_text=TASKS[0]))).startswith("REJECTED")  # default: refused

    renewed = asyncio.run(claim.run(task_index=0, expected_text=TASKS[0], renew=True))
    assert "claimed task 0" in str(renewed)
    assert str(asyncio.run(claim.run(task_index=0, expected_text=TASKS[0], renew=True))).startswith(
        "REJECTED"
    )  # once only


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


def test_a_stale_worker_cannot_overwrite_a_finished_task():
    """The hole an ``owner is None`` allowance leaves open.

    Closing a task clears its owner, so a task that a replacement worker has
    already completed looks unowned again — and the run whose lease expired
    would be free to overwrite the newer result.
    """
    store = Store()
    board = _board(store, lease_seconds=0.05)
    board.set_plan("handover", ["long task"])
    board.claim_next(owner="slow-worker")

    import time

    time.sleep(0.06)
    board.claim_next(owner="fresh-worker")
    board.mark_done(0, "the replacement finished it", owner="fresh-worker")

    late = board.mark_done(0, "the stale worker's answer", owner="slow-worker")

    assert late.startswith("REJECTED")
    assert board.snapshot().tasks[0]["result"] == "the replacement finished it"


def test_a_task_cannot_be_closed_without_being_claimed():
    store = Store()
    board = _board(store)
    board.set_plan("no shortcuts", ["do the thing"])

    assert board.mark_done(0, "pretending").startswith("REJECTED")
    assert board.mark_failed(0, "pretending").startswith("REJECTED")
    assert board.snapshot().tasks[0]["status"] == "todo"


def test_each_planner_instance_claims_under_its_own_identity():
    """Two runs built from the same ``name`` are still different workers."""
    store = Store()
    DurableBlackboard(store, "p1").set_plan("shared", ["a", "b"])

    first = durable_blackboard_agent([_worker()], store=store, plan_id="p1", engine=_ToolCallingEngine([]))
    second = durable_blackboard_agent([_worker()], store=store, plan_id="p1", engine=_ToolCallingEngine([]))

    asyncio.run(first._tool_map["claim_next"].run())
    asyncio.run(second._tool_map["claim_next"].run())

    owners = [t["owner"] for t in DurableBlackboard(store, "p1").snapshot().tasks]
    assert owners[0] != owners[1]
    # ...and the second worker cannot close the first worker's task.
    assert str(asyncio.run(second._tool_map["mark_done"].run(task_index=0, result_summary="not mine"))).startswith(
        "REJECTED"
    )


def test_a_worker_can_re_enter_the_task_it_already_holds():
    """The deadlock in LazyCEO's own intended flow.

    claim_project_task -> open_task_contract -> delegate_project_task is
    how a contracted task is meant to move, and the third step claims the
    same task again to attach the job to it. The lease check never
    compared owners, so the holder was refused entry to its own task and
    told "another worker" held it -- with no other worker in the system.

    Observed live on project lazyceostudio: task 3 sat claimed by
    agent:lazyceo-smoke-simple at attempts=3, and task 4 was driven to
    'failed' at attempts=3 deliberately, to exhaust the budget and free
    the queue. Nothing was wrong with the work; tasks were recorded as
    failures to escape a competitor that did not exist.
    """
    board = _board(Store())
    board.set_plan("reason", TASKS)
    first = board.claim_task(0, TASKS[0], owner="worker-a")
    assert not isinstance(first, str), first

    again = board.claim_task(0, TASKS[0], owner="worker-a", renew=True)

    assert not isinstance(again, str), f"the holder was refused its own task: {again}"
    assert again == (0, TASKS[0])


def test_re_entering_a_held_task_does_not_spend_an_attempt():
    """attempts is the poison-task budget: how many times this work has
    been tried and not finished. A holder stepping back into a task it
    never left has not tried it a second time, and counting it that way
    walks a healthy task to 'failed' in three ordinary steps of the
    intended flow."""
    board = _board(Store(), max_attempts=3)
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="worker-a")

    # One renewal: the claim-to-delegate transition. A second is refused
    # on purpose (see test_a_task_can_only_be_renewed_once) -- what this
    # test pins is that the one allowed re-entry costs nothing.
    assert not isinstance(board.claim_task(0, TASKS[0], owner="worker-a", renew=True), str)

    task = board.snapshot().tasks[0]
    assert task["attempts"] == 1
    assert task["status"] == "claimed"


def test_a_genuinely_different_worker_is_still_refused():
    """The protection this check exists for has to survive the fix: two
    workers must not hold one task at once."""
    board = _board(Store())
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="worker-a")

    refused = board.claim_task(0, TASKS[0], owner="worker-b")

    assert isinstance(refused, str)
    assert "already claimed" in refused


def test_an_anonymous_claim_never_counts_as_the_same_worker():
    """owner=None means "I am not identifying myself". Two anonymous
    callers are not the same caller, so they must not be able to walk into
    each other's leases."""
    board = _board(Store())
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0])

    refused = board.claim_task(0, TASKS[0])

    assert isinstance(refused, str)
    assert "already claimed" in refused


def test_renewing_actually_extends_the_lease():
    """Returning success without moving claimed_at would pass every other
    test here and still lose the task: the original timestamp keeps
    ageing, and another worker takes it the moment it expires. Found by
    Codex review."""
    store = Store()
    board = _board(store, lease_seconds=900.0)
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="worker-a")
    first_claimed_at = store.read(board.key)["tasks"][0]["claimed_at"]

    time.sleep(0.01)
    board.claim_task(0, TASKS[0], owner="worker-a", renew=True)

    assert store.read(board.key)["tasks"][0]["claimed_at"] > first_claimed_at


def test_a_holder_whose_lease_ran_out_spends_an_attempt_to_come_back():
    """The other half of the renewal rule. A stable owner identity -- which
    every agent here has -- would otherwise renew forever across
    stall-kills, never growing attempts, so a task that can never be
    finished would never be parked. Taking the task back after the lease
    lapsed is a fresh attempt, because the previous one did not finish."""
    store = Store()
    board = _board(store, lease_seconds=0.01)
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="worker-a")
    assert store.read(board.key)["tasks"][0]["attempts"] == 1

    time.sleep(0.05)  # the lease lapses while worker-a is stalled
    again = board.claim_task(0, TASKS[0], owner="worker-a", renew=True)

    assert not isinstance(again, str), again
    assert store.read(board.key)["tasks"][0]["attempts"] == 2


def test_a_stale_holder_cannot_renew_a_task_someone_else_took_over():
    """Interleaving: A's lease lapses, B takes the task, then A comes back
    still believing it holds it. A is no longer the owner, so it must be
    refused rather than stealing the task back out from under B."""
    board = _board(Store(), lease_seconds=0.01)
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="worker-a")
    time.sleep(0.05)
    assert not isinstance(board.claim_task(0, TASKS[0], owner="worker-b"), str)

    refused = board.claim_task(0, TASKS[0], owner="worker-a", renew=True)

    assert isinstance(refused, str)
    assert "already claimed" in refused


def test_a_second_delegation_for_a_held_task_is_refused_without_opting_in():
    """The single-worker guarantee, against the caller that actually
    threatens it.

    delegate_plan_tasks claims every item of a batch under ONE
    process-level owner string. If the same task_index appears twice in a
    batch -- or the batch is retried while its first job is still running
    -- owner equality is true, so inferring renewal from it would start a
    second job with real write access on a task that already has one.
    Renewal is opt-in for exactly this reason: a caller that does not say
    it is re-entering gets the old refusal. Found by Codex review on PR
    #169.
    """
    board = _board(Store())
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="shared-process-owner")

    second = board.claim_task(0, TASKS[0], owner="shared-process-owner")

    assert isinstance(second, str)
    assert "already claimed" in second


def test_a_task_can_only_be_renewed_once():
    """Renewal attaches a worker to a task. A second one attaches a second
    worker to the same checkout -- concurrent edits, doubled spend.

    Guarding this OUTSIDE the claim cannot work: scan, claim and record
    are then three separate steps, and two concurrent callers both read
    "nothing running" before either writes. The claim is the mutual
    exclusion primitive, so the exclusion has to live inside its
    compare-and-swap. Found by Codex review on PR #45.
    """
    board = _board(Store())
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="worker-a")
    assert not isinstance(board.claim_task(0, TASKS[0], owner="worker-a", renew=True), str)

    second = board.claim_task(0, TASKS[0], owner="worker-a", renew=True)

    assert isinstance(second, str)
    assert "already claimed" in second


def test_a_lapsed_lease_restores_the_right_to_renew():
    """The release path, and the reason no separate reaper is needed. A
    worker that dies holding a renewal must not freeze the task forever:
    its lease lapses, the next claim is an ordinary fresh one, and that
    resets the right to renew along with everything else."""
    board = _board(Store(), lease_seconds=0.01)
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="worker-a")
    board.claim_task(0, TASKS[0], owner="worker-a", renew=True)

    time.sleep(0.05)  # the holder dies without ever closing the task
    assert not isinstance(board.claim_task(0, TASKS[0], owner="worker-b"), str)

    assert not isinstance(board.claim_task(0, TASKS[0], owner="worker-b", renew=True), str)


def test_reclaiming_via_claim_next_also_restores_the_right_to_renew():
    """The other fresh-claim path. claim_task's own fresh-claim branch resets
    renewed=False, but claim_next() is a SEPARATE apply function with its own
    fresh-claim update -- and claim_project_task actually calls claim_next()
    as its default route, on the very same project board delegate_project_task
    later renews with claim_task(renew=True). If claim_next() left a stale
    renewed=True on a task a prior holder renewed and then abandoned, the new
    holder's own first, legitimate renewal would be refused as if it had
    already spent it. Found by independent review while verifying an
    unrelated task on this same board.
    """
    board = _board(Store(), lease_seconds=0.01)
    board.set_plan("reason", TASKS)
    board.claim_task(0, TASKS[0], owner="worker-a")
    board.claim_task(0, TASKS[0], owner="worker-a", renew=True)

    time.sleep(0.05)  # the holder dies without ever closing the task
    reclaimed = board.claim_next(owner="worker-b")
    assert reclaimed is not None
    assert reclaimed[0] == 0

    assert not isinstance(board.claim_task(0, TASKS[0], owner="worker-b", renew=True), str)


def test_a_task_written_by_1_4_0_reads_with_the_new_fields_as_none() -> None:
    """Tasks persisted before per-task scheduling and completed_at existed
    lack those keys, so a consumer reading task["completed_at"] raised
    KeyError on every such task that had not been through another transition
    since the upgrade. The fields are normalised on read; nothing is
    rewritten in storage."""
    from lazybridge import Store
    from lazybridge.ext.planners import DurableBlackboard

    store = Store()
    board = DurableBlackboard(store, "legacy")
    board.set_plan("reason", ["an old task"])
    key = next(k for k, _ in store.items() if "legacy" in k)
    doc = store.read(key)
    for task in doc["tasks"]:  # what 1.4.0 wrote: no scheduling fields, no completed_at
        for field in ("planned_start_at", "due_at", "completed_at"):
            task.pop(field, None)
    store.write(key, doc)

    task = board.snapshot().tasks[0]

    assert task["completed_at"] is None and task["due_at"] is None and task["planned_start_at"] is None
    assert "completed_at" not in store.read(key)["tasks"][0]  # storage untouched
