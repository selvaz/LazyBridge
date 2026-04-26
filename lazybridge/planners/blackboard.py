"""Blackboard planner — flat todo list, no DAG.

Pattern: ``set_plan`` → loop(``get_plan`` → call sub-agent → ``mark_done``)
        → reply. The LLM manages a list of tasks, ticks them as it goes.

Less precise than :func:`lazybridge.planners.make_planner` (no native
parallel, no structural validation) but easier to prompt and very flexible —
the LLM revises freely by calling ``set_plan`` again. Use it for exploratory
work where the shape emerges as you go.
"""

from typing import Optional

from lazybridge import Agent, LLMEngine, Tool


BLACKBOARD_PLANNER_GUIDANCE = """\
# How to handle a request

You have specialist sub-agents as direct tools, plus three blackboard tools:

- ``set_plan(reasoning, tasks)``     — start or reset the plan.
- ``get_plan()``                     — read current state.
- ``mark_done(task_index, summary)`` — tick a task; record a brief result.

## Workflow

1. **Trivial query** — answer directly.
2. **One sub-agent suffices** — call it directly.
3. **Multi-step work**:
   a. ``set_plan(reasoning="…", tasks=[…])`` — flat list, 3-6 coarse items
      in execution order. Each task self-contained.
   b. Loop: pick the next ``[ ]`` task → call the right sub-agent → call
      ``mark_done(idx, "<1-3 sentence summary>")``.
   c. When all ``[x]``, synthesise the final answer for the user.

Revise mid-flow by calling ``set_plan`` again. Pitfalls: don't skip
``mark_done``; don't dump full sub-agent output into summaries.
"""


def make_blackboard_planner(
    agents: list[Agent],
    *,
    model: str = "claude-opus-4-7",
    system: Optional[str] = None,
    name: str = "blackboard_planner",
    verbose: bool = False,
    verify: Optional[Agent] = None,
    max_verify: int = 3,
) -> Agent:
    """Planner that manages a shared todo-list blackboard.

    Args:
        agents: Sub-agents the planner may call. Unique ``.name`` required.
        model: Provider model id for the planner LLM.
        system: Override default (``BLACKBOARD_PLANNER_GUIDANCE``).
        name: Display name.
        verbose: Print event traces to stdout.
        verify / max_verify: LazyBridge built-in judge loop, off by default.
    """
    if not agents:
        raise ValueError("agents list must not be empty")
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError(f"agents must have unique names; got {names}")

    state = {"reasoning": "", "tasks": [], "done": [], "results": []}

    def _show() -> str:
        if not state["tasks"]:
            return "(no plan; call set_plan)"
        lines = [f"reasoning: {state['reasoning']}"]
        for i, (t, d, r) in enumerate(
            zip(state["tasks"], state["done"], state["results"])
        ):
            mark = "[x]" if d else "[ ]"
            row = f"  {i}. {mark} {t}"
            if r:
                row += f"\n       → {r}"
            lines.append(row)
        nxt = next((i for i, d in enumerate(state["done"]) if not d), None)
        lines.append(
            f"next: {nxt}" if nxt is not None else "all tasks done — reply to user"
        )
        return "\n".join(lines)

    def set_plan(reasoning: str, tasks: list[str]) -> str:
        """Initialise or reset the plan. 3-6 coarse tasks in execution order."""
        if not reasoning.strip() or not tasks:
            return "REJECTED: reasoning and non-empty tasks list are both required."
        state.update(
            reasoning=reasoning.strip(),
            tasks=list(tasks),
            done=[False] * len(tasks),
            results=[""] * len(tasks),
        )
        return _show()

    def get_plan() -> str:
        """Return current plan state with [x]/[ ] marks and recorded results."""
        return _show()

    def mark_done(task_index: int, result_summary: str) -> str:
        """Tick a task; record a 1-3 sentence summary."""
        if not state["tasks"]:
            return "REJECTED: no plan set; call set_plan first."
        if not (0 <= task_index < len(state["tasks"])):
            return f"REJECTED: task_index out of range (valid: 0..{len(state['tasks']) - 1})."
        if not result_summary.strip():
            return "REJECTED: result_summary required (1-3 sentences)."
        state["done"][task_index] = True
        state["results"][task_index] = result_summary.strip()
        return _show()

    return Agent(
        engine=LLMEngine(model, system=system or BLACKBOARD_PLANNER_GUIDANCE),
        tools=[*agents, Tool(set_plan), Tool(get_plan), Tool(mark_done)],
        name=name,
        verbose=verbose,
        verify=verify,
        max_verify=max_verify,
    )
