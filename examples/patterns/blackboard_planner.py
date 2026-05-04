"""Demo: ``lazybridge.ext.planners.make_blackboard_planner`` with three sub-agents.

The factory itself lives in :mod:`lazybridge.ext.planners.blackboard` — this
file just shows a minimal usage pattern. The blackboard planner manages
a flat to-do list (``set_plan`` / ``get_plan`` / ``mark_done``) instead
of composing a DAG; less precise but easier to prompt for exploratory work.
"""

from lazybridge import Agent, LLMEngine
from lazybridge.ext.planners import make_blackboard_planner


def web_search(query: str) -> str:
    """Look up current facts (stub — wire to a real search API)."""
    return f"[stub web result for {query!r}]"


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def main() -> None:
    research = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Look up facts via web_search."),
        tools=[web_search],
        name="research",
        description="Web lookups. No math.",
    )
    math = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Solve arithmetic with add."),
        tools=[add],
        name="math",
        description="Arithmetic only.",
    )
    writer = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Synthesise prior results into prose."),
        name="writer",
        description="Turns prior results into a short paragraph.",
    )

    planner = make_blackboard_planner([research, math, writer], verbose=True)

    queries = [
        "What does FAANG stand for?",
        "What is 17 * 23 + 5?",
        "Research recent agent frameworks and write a one-paragraph summary.",
    ]
    for q in queries:
        print(f"\n>>> {q}")
        print(planner(q).text())


if __name__ == "__main__":
    main()
