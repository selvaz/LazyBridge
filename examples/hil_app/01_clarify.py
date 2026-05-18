"""HIL as a leaf clarifier — the simplest role.

A multi-step pipeline pauses to ask the human for a missing piece of
information mid-flight.  The HIL step is just another node in the
:class:`Plan` — indistinguishable, from the composition's point of view,
from any other ``Agent``-shaped step.

Run (terminal UI — no browser involved):
    python examples/hil_app/01_clarify.py
"""

from __future__ import annotations

from lazybridge import Agent, Plan, Step
from lazybridge.ext.hil import human_agent
from lazybridge.testing import MockAgent


def main() -> None:
    gather = MockAgent(
        ["Partial address detected — missing city."],
        name="gather",
    )
    finalise = MockAgent(["Address normalised."], name="finalise")
    ask_city = human_agent(name="ask_city")  # terminal UI by default

    pipeline = Agent(
        engine=Plan(
            Step(gather),
            Step(ask_city, task="What city is the user in?"),
            Step(finalise),
        ),
        name="address_normaliser",
    )
    result = pipeline("Normalise this contact record")
    print("\n→", result.text())


if __name__ == "__main__":
    main()
