"""Dynamic planner with re-planning and per-round parallel execution.

Pattern
-------
A planner agent reasons about the user's query and emits a *round* of
independent tasks (run in parallel). After each round it sees the results
and either emits another round or declares the work done. This is "ReAct
on tasks" — the planner re-plans every round, so it can adapt to
intermediate findings instead of committing to a fixed task list up front.

Why not :class:`lazybridge.Plan`?
    ``Plan`` is compiled at construction time — its DAG is fixed. This file
    targets the case where the *shape* of the work depends on the query and
    on intermediate results, so we use a normal ``Agent`` with structured
    output and a tiny Python orchestration loop.

Why structured output?
    ``Agent(output=PlanRound)`` forces the planner's response through Pydantic
    validation, so the dispatcher gets typed ``Task`` objects, not free-form
    text it has to parse.

Knobs to tune
-------------
- ``max_rounds``        : safety cap on re-plans.
- ``done`` flag         : planner short-circuits when nothing more is needed.
- ``parallel=False``    : per-task escape hatch when ordering matters within
                          a round (rare — usually you'd just emit two rounds).
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from lazybridge import Agent, LLMEngine

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
        "claude-opus-4-7",
        system="You look up current facts via web_search. You do not do math.",
    ),
    tools=[web_search],
    name="research",
)

math_agent = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You solve arithmetic with the add/multiply tools. One tool at a time.",
    ),
    tools=[add, multiply],
    name="math",
)

writer_agent = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You synthesise prior results into clear prose. No new facts.",
    ),
    name="writer",
)

AGENTS: dict[str, Agent] = {
    "research": research_agent,
    "math": math_agent,
    "writer": writer_agent,
}


# ---------------------------------------------------------------------------
# 2. Typed plan — one *round* at a time. Planner emits this each iteration.
# ---------------------------------------------------------------------------


AgentName = Literal["research", "math", "writer"]


class Task(BaseModel):
    agent: AgentName = Field(..., description="Which sub-agent runs this task.")
    instruction: str = Field(..., description="Self-contained instruction for the sub-agent.")
    parallel: bool = Field(
        default=True,
        description="If False, run sequentially after parallel siblings in the same round.",
    )


class PlanRound(BaseModel):
    reasoning: str = Field(..., description="Why this round of tasks was chosen.")
    tasks: list[Task] = Field(default_factory=list, description="Tasks for this round.")
    done: bool = Field(default=False, description="True when no further rounds are needed.")
    final_answer: str | None = Field(default=None, description="Required when done=True; the user-facing answer.")


PLANNER_SYSTEM = """\
You are a planner. Each turn you produce ONE round of tasks to run.

Available sub-agents:
- research : web lookups, facts, news. Cannot do math.
- math     : arithmetic only.
- writer   : synthesise prior results into prose. Adds no new facts.

Rules:
1. Tasks within a round run IN PARALLEL — do not put dependent tasks in the
   same round. Emit dependents in the next round, after you've seen results.
2. Tasks marked parallel=False run sequentially after the parallel ones in
   the same round (rarely needed).
3. When the user's question can be answered from the accumulated results,
   set done=true and put the user-facing answer in final_answer.
4. Be greedy with parallelism: if two facts can be looked up independently,
   put them in the same round.
"""


planner = Agent(
    engine=LLMEngine("claude-opus-4-7", system=PLANNER_SYSTEM),
    output=PlanRound,
    name="planner",
)


# ---------------------------------------------------------------------------
# 3. Orchestration loop — replan + parallel dispatch
# ---------------------------------------------------------------------------


def _format_history(query: str, history: list[tuple[Task, str]]) -> str:
    lines = [f"User query: {query}"]
    if history:
        lines.append("\nResults from prior rounds:")
        for i, (t, out) in enumerate(history, 1):
            lines.append(f"  {i}. [{t.agent}] {t.instruction}\n     → {out}")
    else:
        lines.append("\n(no prior rounds — this is round 1)")
    return "\n".join(lines)


async def _run_round(tasks: list[Task]) -> list[str]:
    """Run a round: parallel tasks concurrently, then sequential ones in order."""
    parallel = [t for t in tasks if t.parallel]
    sequential = [t for t in tasks if not t.parallel]

    par_results: list[str] = []
    if parallel:
        envs = await asyncio.gather(*(AGENTS[t.agent].run(t.instruction) for t in parallel))
        par_results = [e.text() for e in envs]

    seq_results: list[str] = []
    for t in sequential:
        env = await AGENTS[t.agent].run(t.instruction)
        seq_results.append(env.text())

    # Re-interleave so the result list matches the original tasks order.
    out: list[str] = []
    p_iter = iter(par_results)
    s_iter = iter(seq_results)
    for t in tasks:
        out.append(next(p_iter) if t.parallel else next(s_iter))
    return out


async def solve(query: str, max_rounds: int = 5) -> str:
    history: list[tuple[Task, str]] = []
    for round_num in range(1, max_rounds + 1):
        plan_env = await planner.run(_format_history(query, history))
        plan: PlanRound = plan_env.payload

        print(f"\n=== round {round_num} ===")
        print(f"reasoning: {plan.reasoning}")
        if plan.done:
            print("planner: DONE")
            return plan.final_answer or (history[-1][1] if history else "")

        if not plan.tasks:
            # Pathological: not done but no tasks — break to avoid an infinite loop.
            return plan.final_answer or "planner emitted empty round; aborting"

        for t in plan.tasks:
            print(f"  → [{t.agent}{'' if t.parallel else ' (seq)'}] {t.instruction}")

        outputs = await _run_round(plan.tasks)
        history.extend(zip(plan.tasks, outputs))

    return "max_rounds reached without done=True; partial: " + (history[-1][1] if history else "")


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    query = (
        "What is the combined headcount of Apple and Google in 2024, and "
        "write a one-paragraph note on what those numbers say about the "
        "two companies' staffing strategies?"
    )
    answer = asyncio.run(solve(query))
    print("\n=== FINAL ANSWER ===\n" + answer)


if __name__ == "__main__":
    main()
