"""Planner agent emits a typed PlanSpec; we materialise it into a real Plan.

Why this is the cleanest "planner builds a plan" pattern in LazyBridge
---------------------------------------------------------------------
- The planner returns a ``PlanSpec`` (Pydantic). No free-form text to parse.
- ``materialize()`` turns the spec into a real ``Plan(Step(...), ...)`` with
  live agents bound to step targets. ``PlanCompiler`` validates the DAG when
  the Plan is wrapped by ``Agent.from_engine(plan)`` — broken DAGs (forward
  ``from_step`` references, unknown step names, duplicates) surface as
  ``PlanCompileError`` *before* any LLM call.
- ``Step(parallel=True)`` runs siblings concurrently. No asyncio loop in user
  code: ``Plan`` does the dispatch.
- Re-planning is opt-in (``solve(replan=True)``): after the Plan finishes, the
  planner sees the result and either declares done or builds a new Plan.

When to prefer the "asyncio loop" pattern in ``dynamic_planner.py`` instead:
when you need to feed *partial* results back mid-round (the planner inspects
results before the round finishes). Plan-based execution is round-atomic.
"""

from typing import Literal

from pydantic import BaseModel, Field

from lazybridge import Agent, LLMEngine, Plan, Step, from_parallel, from_prev, from_step

# ---------------------------------------------------------------------------
# 1. Sub-agents (the registry the planner is allowed to pick from)
# ---------------------------------------------------------------------------


def web_search(query: str) -> str:
    """Look up current facts (stub)."""
    return f"[stub web result for {query!r}]"


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


research_agent = Agent(
    engine=LLMEngine("claude-opus-4-7", system="Look up facts via web_search."),
    tools=[web_search],
    name="research",
)
math_agent = Agent(
    engine=LLMEngine("claude-opus-4-7", system="Solve arithmetic with add/multiply."),
    tools=[add, multiply],
    name="math",
)
writer_agent = Agent(
    engine=LLMEngine("claude-opus-4-7", system="Synthesise prior results into prose."),
    name="writer",
)

REGISTRY: dict[str, Agent] = {
    "research": research_agent,
    "math": math_agent,
    "writer": writer_agent,
}


# ---------------------------------------------------------------------------
# 2. PlanSpec — what the planner emits (structured output)
# ---------------------------------------------------------------------------


class StepSpec(BaseModel):
    name: str = Field(..., description="Unique step name; used by from_step references.")
    agent: Literal["research", "math", "writer"] = Field(..., description="Which sub-agent runs this step.")
    task_kind: Literal["literal", "from_prev", "from_step", "from_parallel"] = Field(
        default="from_prev",
        description=(
            "How this step receives its task: literal=use task_text, "
            "from_prev=output of previous step, from_step=output of named step, "
            "from_parallel=list of envelopes from a parallel sibling group."
        ),
    )
    task_text: str | None = Field(default=None, description="Required when task_kind='literal'.")
    task_step: str | None = Field(
        default=None,
        description="Required when task_kind='from_step' or 'from_parallel'.",
    )
    parallel: bool = Field(default=False, description="If True, run concurrently with adjacent parallel siblings.")


class PlanSpec(BaseModel):
    reasoning: str = Field(..., description="Why this DAG was chosen.")
    steps: list[StepSpec] = Field(default_factory=list, description="DAG steps in order.")
    done: bool = Field(default=False, description="Set True with no steps to short-circuit.")
    final_answer: str | None = Field(default=None, description="If done=True, the user-facing answer.")


PLANNER_SYSTEM = f"""\
You are a planner that produces a small DAG of work.

Available sub-agents (pick exactly one per step):
- research : web lookups. Cannot do math.
- math     : arithmetic only.
- writer   : synthesise prior results into prose. Adds no new facts.

You emit a PlanSpec with ordered steps. Each step has:
  - name      : unique short identifier (snake_case)
  - agent     : one of {sorted(REGISTRY)}
  - task_kind : 'literal' (provide task_text), 'from_prev' (default — uses
                previous step's output as task), 'from_step' (uses named
                step's output; provide task_step), 'from_parallel' (gather
                list of envelopes from a parallel sibling group)
  - parallel  : True to run concurrently with adjacent parallel siblings

Rules:
1. The first step must use task_kind='literal' (no previous step exists).
2. To fan out, mark several adjacent steps parallel=True; their join step
   should use task_kind='from_parallel' and task_step=<name of first parallel sibling>.
3. If the question can be answered without any work, set done=true and put
   the answer in final_answer (leave steps empty).
4. End the DAG with a 'writer' step when prose is required.
"""


planner = Agent(
    engine=LLMEngine("claude-opus-4-7", system=PLANNER_SYSTEM),
    output=PlanSpec,
    name="planner",
)


# ---------------------------------------------------------------------------
# 3. Materialise a PlanSpec into a real Plan (compile-time validation)
# ---------------------------------------------------------------------------


def materialize(spec: PlanSpec) -> Plan:
    steps: list[Step] = []
    for s in spec.steps:
        if s.task_kind == "literal":
            if not s.task_text:
                raise ValueError(f"step {s.name!r}: task_kind='literal' needs task_text")
            task = s.task_text
        elif s.task_kind == "from_step":
            if not s.task_step:
                raise ValueError(f"step {s.name!r}: task_kind='from_step' needs task_step")
            task = from_step(s.task_step)
        elif s.task_kind == "from_parallel":
            if not s.task_step:
                raise ValueError(f"step {s.name!r}: task_kind='from_parallel' needs task_step")
            task = from_parallel(s.task_step)
        else:  # from_prev
            task = from_prev

        steps.append(
            Step(
                target=REGISTRY[s.agent],
                name=s.name,
                task=task,
                parallel=s.parallel,
            )
        )
    # Plan() runs PlanCompiler — bad references raise PlanCompileError here.
    return Plan(*steps, max_iterations=max(20, len(steps) * 3))


# ---------------------------------------------------------------------------
# 4. solve() — one-shot or replan
# ---------------------------------------------------------------------------


def _format_with_history(query: str, history: list[str]) -> str:
    if not history:
        return query
    rounds = "\n\n".join(f"--- prior round {i + 1} ---\n{h}" for i, h in enumerate(history))
    return f"User query: {query}\n\nPrior plan results:\n{rounds}\n\nDecide next plan or set done=true."


def solve(query: str, *, replan: bool = False, max_rounds: int = 5) -> str:
    """Execute one Plan. If replan=True, loop until the planner sets done=true."""
    history: list[str] = []
    for round_num in range(1, max_rounds + 1):
        spec: PlanSpec = planner(_format_with_history(query, history)).payload
        print(f"\n=== plan round {round_num} ===")
        print("reasoning:", spec.reasoning)

        if spec.done:
            print("planner: DONE")
            return spec.final_answer or (history[-1] if history else "")

        if not spec.steps:
            return spec.final_answer or "planner emitted empty plan; aborting"

        plan = materialize(spec)  # ← compile-time validated
        print(f"materialised plan with {len(spec.steps)} step(s); running…")

        result = Agent.from_engine(plan)(query).text()
        history.append(result)

        if not replan:
            return result

    return f"max_rounds reached; partial: {history[-1] if history else ''}"


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    query = (
        "Get the headcounts of Apple, Google, and Meta in 2024 (you can look "
        "these up in parallel). Then compute the total. Finally, write a "
        "short paragraph commenting on the total."
    )
    answer = solve(query, replan=False)  # set replan=True for adaptive multi-round
    print("\n=== FINAL ANSWER ===\n" + answer)


if __name__ == "__main__":
    main()
