"""HIL as a pipeline entrypoint — the human's input *starts* the run.

Same primitive as ``01_clarify.py``, but moved to the head of the plan:
the human types the question, the agent answers, the pipeline
terminates.  This is the minimal "ask-and-answer" web app skeleton —
what was a leaf in 01 becomes the front door here.

Open the printed URL in a browser when prompted.

Run:
    python examples/hil_app/02_entrypoint.py
"""

from __future__ import annotations

from lazybridge import Agent, Plan, Step
from lazybridge.ext.hil import human_agent
from lazybridge.testing import MockAgent


def main() -> None:
    ask = human_agent(ui="web", name="ask")
    answer = MockAgent(
        ["LazyBridge is a zero-boilerplate multi-provider agent framework."],
        name="answer",
    )

    pipeline = Agent(
        engine=Plan(
            Step(ask, task="What would you like to know?"),
            Step(answer),
        ),
        name="assistant",
    )
    result = pipeline("(human enters the real query)")
    print("\n→", result.text())


if __name__ == "__main__":
    main()
