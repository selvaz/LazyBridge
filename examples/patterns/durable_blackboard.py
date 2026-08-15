"""Always-on worker: a plan that survives the process that runs it.

The ephemeral blackboard (``blackboard_planner.py``) resets its to-do list on
every invocation — right for a one-shot planner, wrong for an agent that wakes
up, does one thing, and has to know where it is next time.

Here the plan lives in a ``Store(db=...)``. Each wake-up claims **one** task,
does it, ticks it, and stops. Kill the process at any point and the next start
picks up exactly where it left off — including a task that was claimed but
never finished, which returns to the queue once its lease expires.

Run it twice to see the resume: the second run continues the same plan.

    python examples/patterns/durable_blackboard.py
"""

from __future__ import annotations

from lazybridge import Agent, Store
from lazybridge.ext.planners import DURABLE_BLACKBOARD_GUIDANCE, DurableBlackboard, durable_blackboard_agent

PLAN_DB = "durable_blackboard.sqlite"
PLAN_ID = "quarterly-memo"

TASKS = [
    "collect the three latest filings",
    "extract revenue and margin",
    "write a two-paragraph memo",
]


def show(text: str) -> None:
    """Print board state without dying on a legacy console.

    ``render()`` uses ``→`` for results; a default Windows console is cp1252
    and raises ``UnicodeEncodeError`` on it, which would crash the example
    rather than the thing it is demonstrating.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def build_worker() -> Agent:
    """The specialist that actually does each task."""
    return Agent(
        name="analyst",
        description="Performs one research or writing task and reports the result.",
        model="claude-opus-4-7",
    )


def tick(store: Store) -> str:
    """One wake-up: plan if needed, otherwise advance the plan by one task."""
    planner = durable_blackboard_agent(
        [build_worker()],
        store=store,
        plan_id=PLAN_ID,
        model="claude-opus-4-7",
        system=DURABLE_BLACKBOARD_GUIDANCE,
        lease_seconds=600,
        max_attempts=3,
    )
    board = DurableBlackboard(store, PLAN_ID)
    if not board.snapshot().tasks:
        return planner(f"Create a plan for this job: {'; '.join(TASKS)}").text()
    return planner("Continue the plan: claim the next task, do it, and tick it off.").text()


def main() -> None:
    # A file-backed Store is what makes this survive restarts; the default
    # in-memory Store would put us right back to a per-process plan.
    with Store(db=PLAN_DB) as store:
        board = DurableBlackboard(store, PLAN_ID)
        show("state at start:\n" + board.render() + "\n")

        if board.snapshot().complete:
            show(f"plan already complete — delete {PLAN_DB} to start a new one")
            return

        show(tick(store))
        show("\nstate after this run:\n" + board.render())


if __name__ == "__main__":
    main()
