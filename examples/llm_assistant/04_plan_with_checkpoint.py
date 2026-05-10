"""Sequential plan with a checkpoint Store.

``Agent(engine=Plan(Step('a'), Step('b')))`` is the canonical
sequential composition; pass ``store=Store(db='state.sqlite')`` for
crash-resume.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from lazybridge import Agent, Plan, Step, Store
from lazybridge.testing import MockAgent


def main() -> None:
    research = MockAgent(["Background: LazyBridge is an agent framework."], name="research")
    write = MockAgent(["LazyBridge is a Python framework for building LLM agents."], name="write")

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.sqlite"
        pipeline = Agent(
            engine=Plan(Step("research"), Step("write")),
            tools=[research, write],
            store=Store(db=str(db)),
            name="research_pipeline",
        )
        env = pipeline("LazyBridge")
        print("final text:", env.text())
        # ``Store`` is durable — a second pipeline pointed at the same db
        # would resume from any committed checkpoint.


if __name__ == "__main__":
    main()
