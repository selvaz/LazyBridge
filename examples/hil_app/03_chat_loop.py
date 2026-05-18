"""HIL as a cyclic entrypoint — multi-turn chat via Plan ``routes``.

The pipeline routes from the agent step back to the HIL step, forming a
cycle that lasts as long as the human keeps responding.  Three things
make this work without any chat-specific framework code:

- :class:`Plan` already supports cycles via ``routes`` pointing at an
  earlier step (``engines/plan/_types.py:116``).
- The default ``task=from_prev`` sentinel propagates the agent's
  previous reply forward, so every turn after the first shows the
  agent's last response as the task for HIL — natural conversation
  flow with no manual history wiring.
- The web ``HumanEngine`` keeps its HTTP server alive across
  ``prompt()`` calls, so the browser tab stays on the same URL through
  every turn (post-submit auto-redirects, "Agent is thinking…"
  placeholder during processing).

Type ``exit`` (any case, anywhere in the input) to end the session.

Run:
    python examples/hil_app/03_chat_loop.py
"""

from __future__ import annotations

from lazybridge import Agent, Plan, Step
from lazybridge.ext.hil import human_agent
from lazybridge.testing import MockAgent


def _is_exit(env) -> bool:
    return "exit" in (env.text() or "").lower()


def main() -> None:
    ask = human_agent(ui="web", name="ask")
    answer = MockAgent(
        # MockAgent cycles its scripted answers, so each turn gets the
        # next one and the demo stays self-contained (no API key).
        [
            "LazyBridge is a Python framework for building LLM agents.",
            "It supports multiple providers natively (Anthropic, OpenAI, Google, …).",
            "Composition is via Plan(Step(...), Step(...)).",
            "HIL is the human-in-the-loop primitive — see ext/hil/human.py.",
        ],
        name="answer",
    )
    farewell = MockAgent(["Goodbye!"], name="farewell")

    pipeline = Agent(
        engine=Plan(
            # No explicit task=: the default ``from_prev`` makes HIL
            # see the Plan's initial prompt on turn 1, and the agent's
            # previous reply on every subsequent turn.
            Step(ask, routes={"farewell": _is_exit}),
            Step(answer, routes={"ask": lambda _e: True}),
            Step(farewell),
            max_iterations=10_000,
        ),
        name="chat",
    )
    result = pipeline("Welcome — what would you like to ask? (type 'exit' to stop)")
    print("\n→ session ended:", result.text())


if __name__ == "__main__":
    main()
