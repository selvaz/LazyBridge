"""Demo: ``lazybridge.planners.make_planner`` with three sub-agents.

The factory itself lives in :mod:`lazybridge.planners.builder` — this file
just shows a minimal usage pattern. Run it with provider credentials in
the environment to see the planner pick between direct sub-agent calls
and ``execute_plan`` for multi-step work.
"""

from lazybridge import Agent, LLMEngine
from lazybridge.planners import make_planner


def web_search(query: str) -> str:
    """Look up current facts (stub — wire to a real search API)."""
    return f"[stub web result for {query!r}]"


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def main() -> None:
    research = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Look up facts via web_search."),
        tools=[web_search],
        name="research",
        description="Web lookups for current facts. No math.",
    )
    math = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Solve arithmetic with add/multiply."),
        tools=[add, multiply],
        name="math",
        description="Arithmetic (add, multiply). No facts.",
    )
    writer = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Synthesise prior results into prose."),
        name="writer",
        description="Turns prior results into a short paragraph.",
    )

    planner = make_planner([research, math, writer], verbose=True)

    queries = [
        "What does FAANG stand for?",                                    # trivial
        "What is 17 * 23 + 5?",                                          # one agent
        "Research quantum networking and write a one-paragraph brief.",  # multi-step plan
        "Look up the FAANG headcounts in parallel and write a summary.", # parallel band + N-branch synth
    ]
    for q in queries:
        print(f"\n>>> {q}")
        print(planner(q).text())


if __name__ == "__main__":
    main()
