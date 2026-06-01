"""Dynamic planner with re-planning, parallel execution, and checkpoint/resume.

Pattern
-------
A planner agent reasons about the user's query and emits a *round* of
independent tasks (run in parallel). After each round it sees the results
and either emits another round or declares the work done. This is "ReAct
on tasks" — the planner re-plans every round, adapting to intermediate
findings instead of committing to a fixed task list up front.

Why not :class:`lazybridge.Plan`?
    ``Plan`` is compiled at construction time — its DAG is fixed. This file
    targets the case where the *shape* of the work depends on the query and
    on intermediate results.

Why :class:`lazybridge.ReplanEngine` instead of a raw Python loop?
    ``ReplanEngine`` is the guardian: it checkpoints after every round so a
    restart continues from the correct round rather than re-executing completed
    work.  Pass ``store=`` and ``checkpoint_key=`` to enable persistence.

Architecture (LazyBridge "everything is a tool")
-------------------------------------------------
The planner and all workers are tools in the guardian Agent's tool_map:

    guardian.tools = [planner, research_agent, math_agent, writer_agent]
                       ↑ (output=PlanRound)   ↑ workers dispatched by ReplanEngine

The planner receives the available tool schemas + history dynamically — its
system prompt does not need to hardcode worker names.
"""

from __future__ import annotations

from lazybridge import Agent, LLMEngine, ReplanEngine
from lazybridge.engines.replan import PlanRound

# ---------------------------------------------------------------------------
# 1. Sub-agents — each owns its own tool set / system prompt
# ---------------------------------------------------------------------------


def web_search(query: str) -> str:
    """Look up current facts on the web (stub)."""
    return f"[stub web result for {query!r}]"


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


research_agent = Agent(
    engine=LLMEngine(
        "deepseek-v4-flash",
        system="You look up current facts via web_search. You do not do math.",
    ),
    tools=[web_search],
    name="research",
    description="Web lookups, facts, news. Cannot do math.",
)

math_agent = Agent(
    engine=LLMEngine(
        "deepseek-v4-flash",
        system="You solve arithmetic with the add/multiply tools. One tool at a time.",
    ),
    tools=[add, multiply],
    name="math",
    description="Arithmetic only.",
)

writer_agent = Agent(
    engine=LLMEngine(
        "deepseek-v4-flash",
        system="You synthesise prior results into clear prose. No new facts.",
    ),
    name="writer",
    description="Synthesise prior results into prose. Adds no new facts.",
)


# ---------------------------------------------------------------------------
# 2. Planner — emits PlanRound each turn
#
# The system prompt is minimal: ReplanEngine injects the available tool
# schemas and the accumulated history into every planner call dynamically.
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """\
You are a task planner. Each turn you receive:
  - "Available tools:" — the tool names, signatures, and descriptions
  - "Task:" — the original user query
  - "History:" — outputs from prior rounds

Produce ONE PlanRound. Rules:
1. Tasks within a round run IN PARALLEL — put dependent tasks in the next round.
2. Use tool names and kwargs exactly as listed in "Available tools".
3. When the question is answered, set done=true and put the answer in final_answer.
4. Be greedy with parallelism: independent lookups belong in the same round.
"""

planner = Agent(
    engine=LLMEngine("deepseek-v4-flash", system=PLANNER_SYSTEM),
    output=PlanRound,
    name="planner",
)


# ---------------------------------------------------------------------------
# 3. Guardian — ReplanEngine wraps the replan loop with checkpoint/resume
# ---------------------------------------------------------------------------

guardian = Agent(
    engine=ReplanEngine(max_rounds=10),  # add store= + checkpoint_key= for persistence
    tools=[planner, research_agent, math_agent, writer_agent],
    name="guardian",
)


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    query = (
        "What is the combined headcount of Apple and Google in 2024, and "
        "write a one-paragraph note on what those numbers say about the "
        "two companies' staffing strategies?"
    )
    env = guardian(query)
    if env.error:
        print(f"ERROR: {env.error.message}")
    else:
        print("\n=== FINAL ANSWER ===\n" + (env.text() or ""))


if __name__ == "__main__":
    main()
