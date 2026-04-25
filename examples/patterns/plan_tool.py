"""``make_planner`` — give an LLM a list of sub-agents and a tool to compose
them into a Plan.

The whole pattern in one factory: pass your sub-agents, get back a planner
:class:`lazybridge.Agent`. The planner has those agents as direct tools (so
it can call them one at a time when one is enough) plus an ``execute_plan``
tool (so it can compose them into a multi-step DAG when one isn't).

Usage
-----
::

    from examples.patterns.plan_tool import make_planner

    research = Agent("claude-opus-4-7", tools=[web_search],
                     name="research", description="Web lookups. No math.")
    math     = Agent("claude-opus-4-7", tools=[add],
                     name="math",     description="Arithmetic only.")
    writer   = Agent("claude-opus-4-7",
                     name="writer",   description="Prose synthesis.")

    planner = make_planner([research, math, writer])
    planner("Research recent agent frameworks and write a one-paragraph summary.")

What ``execute_plan`` does
--------------------------
The LLM emits a ``PlanSpec`` (an ordered list of typed ``StepSpec`` items).
``_materialize`` turns it into a real :class:`lazybridge.Plan`;
``Agent.from_engine(plan)`` triggers ``PlanCompiler`` so forward
``from_step`` references, unknown step names, and duplicates are rejected
**before any inner LLM call runs**. On rejection the tool returns
``PLAN_REJECTED: <reason>`` so the planner LLM can self-correct.

Honest about a limitation
-------------------------
``from_parallel("name")`` is an alias for ``from_step`` — it forwards a
single specific branch's envelope, not a list of all parallel siblings.
The planner can express linear pipelines and a parallel band whose follow-up
reads *one* branch (or two — one as ``task``, one as ``context``, see
``StepSpec.context_kind``). For "fan out N independent legs and synthesise
all of them", the planner is better off calling the underlying agents
directly (or multiple times) rather than packing the work into one Plan;
``execute_plan`` won't deliver list-of-branches to a join step.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from lazybridge import Agent, LLMEngine, Plan, Step, Tool, from_parallel, from_prev, from_step
from lazybridge.engines.plan import PlanCompileError


# ---------------------------------------------------------------------------
# Plan spec — what the planner LLM emits when calling execute_plan
# ---------------------------------------------------------------------------


class StepSpec(BaseModel):
    """One node in the plan DAG."""

    name: str = Field(..., description="Unique step identifier; referenced by from_step.")
    agent: str = Field(..., description="Sub-agent name; must match one of the planner's agents.")
    task_kind: Literal["literal", "from_prev", "from_step", "from_parallel"] = Field(
        default="from_prev",
        description=(
            "literal=use task_text; from_prev=output of preceding step "
            "(default); from_step=output of step named in task_step; "
            "from_parallel=alias of from_step, one branch's output."
        ),
    )
    task_text: Optional[str] = Field(
        default=None, description="Required when task_kind='literal'."
    )
    task_step: Optional[str] = Field(
        default=None,
        description="Required when task_kind='from_step' or 'from_parallel'.",
    )
    context_kind: Optional[Literal["from_step", "from_parallel"]] = Field(
        default=None,
        description=(
            "Optional secondary input pulled into the step's context. Useful "
            "to combine TWO parallel branches' outputs in a join step "
            "(task_kind=from_parallel for branch A, context_kind=from_parallel "
            "for branch B). For more than two branches, plan-only composition "
            "doesn't deliver them; call the agents directly instead."
        ),
    )
    context_step: Optional[str] = Field(
        default=None, description="Required when context_kind is set."
    )
    parallel: bool = Field(
        default=False,
        description="If True, run concurrently with adjacent parallel siblings.",
    )


class PlanSpec(BaseModel):
    """The argument shape of execute_plan."""

    task: str = Field(..., description="The user task that drives the plan.")
    steps: list[StepSpec] = Field(..., description="Ordered DAG steps.")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _PlanToolError(Exception):
    """Raised on bad spec; surfaced to the LLM as a tool result string."""


def _resolve_task(s: StepSpec) -> Any:
    if s.task_kind == "literal":
        if not s.task_text:
            raise _PlanToolError(
                f"Step {s.name!r}: task_kind='literal' requires task_text."
            )
        return s.task_text
    if s.task_kind == "from_step":
        if not s.task_step:
            raise _PlanToolError(
                f"Step {s.name!r}: task_kind='from_step' requires task_step."
            )
        return from_step(s.task_step)
    if s.task_kind == "from_parallel":
        if not s.task_step:
            raise _PlanToolError(
                f"Step {s.name!r}: task_kind='from_parallel' requires task_step."
            )
        return from_parallel(s.task_step)
    return from_prev


def _resolve_context(s: StepSpec) -> Any:
    if s.context_kind is None:
        return None
    if not s.context_step:
        raise _PlanToolError(
            f"Step {s.name!r}: context_kind={s.context_kind!r} requires context_step."
        )
    if s.context_kind == "from_parallel":
        return from_parallel(s.context_step)
    return from_step(s.context_step)


def _materialize(spec: PlanSpec, registry: dict[str, Agent]) -> Plan:
    if not spec.steps:
        raise _PlanToolError("PlanSpec.steps is empty; emit at least one step or skip the plan.")

    unknown = sorted({s.agent for s in spec.steps if s.agent not in registry})
    if unknown:
        raise _PlanToolError(
            f"Unknown agent name(s) {unknown!r}. "
            f"Available: {sorted(registry)!r}."
        )

    real_steps: list[Step] = []
    for s in spec.steps:
        real_steps.append(
            Step(
                target=registry[s.agent],
                name=s.name,
                task=_resolve_task(s),
                context=_resolve_context(s),
                parallel=s.parallel,
            )
        )
    return Plan(*real_steps, max_iterations=max(20, len(real_steps) * 3))


def _format_compile_error(err: PlanCompileError, registry: dict[str, Agent]) -> str:
    return (
        "PLAN_REJECTED: " + str(err) + "\n\n"
        "Hints: from_step / from_parallel targets must be defined earlier "
        "in the DAG; step names must be unique. "
        f"Valid agents: {sorted(registry)!r}. "
        "Re-emit a corrected PlanSpec."
    )


# ---------------------------------------------------------------------------
# The plan tool
# ---------------------------------------------------------------------------


def make_execute_plan_tool(
    registry: dict[str, Agent],
    *,
    name: str = "execute_plan",
    description: Optional[str] = None,
) -> Tool:
    """Return a :class:`lazybridge.Tool` that builds and runs a Plan.

    Tool signature: ``execute_plan(steps: list[StepSpec], task: str) -> str``.

    Args:
        registry: Mapping ``{agent_name: Agent}`` the plan may dispatch to.
        name: Tool name visible to the LLM. Defaults to ``"execute_plan"``.
        description: Override the tool's LLM-facing description. The default
            includes the registry's agent names and one-line summaries.

    Raises:
        ValueError: if the registry is empty.
    """
    if not registry:
        raise ValueError("plan tool registry must contain at least one agent")

    if description is None:
        agent_lines = []
        for n, a in registry.items():
            d = (a.description or "").strip() or "no description"
            agent_lines.append(f"- {n}: {d}")
        description = (
            "Compose and run a multi-step plan over the available sub-agents. "
            "The plan is a DAG; declare parallel=true on siblings to run them "
            "concurrently. Returns the final step's text. Use this when the "
            "task requires more than one step or coordinated sub-agents.\n\n"
            "Available sub-agents:\n" + "\n".join(agent_lines)
        )

    async def execute_plan(steps: list[StepSpec], task: str) -> str:
        """Build and run a plan over the registered sub-agents."""
        spec = PlanSpec(task=task, steps=steps)
        try:
            plan = _materialize(spec, registry)
        except _PlanToolError as e:
            return f"PLAN_REJECTED: {e}"

        try:
            runner = Agent.from_engine(plan)  # PlanCompiler runs here.
        except PlanCompileError as e:
            return _format_compile_error(e, registry)

        try:
            env = await runner.run(spec.task)
        except Exception as e:  # noqa: BLE001 — surface anything to the LLM
            return f"PLAN_RUNTIME_ERROR: {type(e).__name__}: {e}"

        if env.error:
            return f"PLAN_RUNTIME_ERROR: {env.error.message}"
        return env.text()

    return Tool(execute_plan, name=name, description=description, mode="signature")


# ---------------------------------------------------------------------------
# Guidance — drop into the planner's system prompt
# ---------------------------------------------------------------------------

PLANNER_GUIDANCE = """\
# How to handle a request

You have a set of specialist sub-agents available as direct tools, plus an
``execute_plan`` tool to compose them when one agent isn't enough.

## Decision rules

1. **Trivial query** — answer directly. No tool call.
2. **One sub-agent suffices** — call that agent directly as a tool.
3. **Multiple steps that depend on each other, OR a parallel band followed
   by a single step that reads one branch** — call ``execute_plan``.

That's it. Don't reach for ``execute_plan`` when calling one sub-agent
twice in a row would be just as clear.

## execute_plan reference

Each ``StepSpec`` has:

- ``name``      : unique snake_case identifier.
- ``agent``     : sub-agent name (must match one of the agents you were given).
- ``task_kind`` : how this step receives its task —
    * ``"literal"``       : use ``task_text``. Required for the first step
                            when you want a hand-crafted prompt; otherwise
                            ``from_prev`` reads the user task verbatim.
    * ``"from_prev"``     : output of the preceding step (default).
    * ``"from_step"``     : output of the step named in ``task_step``;
                            that step **must come earlier** in the DAG.
    * ``"from_parallel"`` : alias of ``from_step`` for readability when the
                            referenced step ran with ``parallel=true``.
                            Reads ONE specific branch's output.
- ``task_text`` : required when ``task_kind="literal"``.
- ``task_step`` : required when ``task_kind`` is ``from_step`` or ``from_parallel``.
- ``context_kind`` / ``context_step`` (optional) : pull a SECOND step's
  output into the step's context. Useful to combine two parallel branches
  in a join step (one as task, one as context).
- ``parallel``  : run concurrently with adjacent ``parallel=true`` siblings.

## Worked examples

### Trivial — answer directly
User: "What does FAANG stand for?"
You: "FAANG = Facebook (Meta), Apple, Amazon, Netflix, Google."

### One agent — call it directly
User: "What is 17 * 23 + 5?"
You: call ``math("Compute 17 * 23 + 5.")``

### Sequential pipeline — execute_plan
User: "Research quantum networking and write a one-paragraph brief."
You:
``execute_plan(
    task="Quantum networking",
    steps=[
        {"name": "r", "agent": "research"},
        {"name": "w", "agent": "writer"},
    ],
)``

### Parallel band + read one branch — execute_plan
User: "Look up Apple and Google headcounts in parallel; report Apple's."
You:
``execute_plan(
    task="...",
    steps=[
        {"name": "hc_apple",  "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Apple in 2024",  "parallel": true},
        {"name": "hc_google", "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Google in 2024", "parallel": true},
        {"name": "report",    "agent": "writer",   "task_kind": "from_parallel",
         "task_step": "hc_apple"},
    ],
)``

### Parallel band + combine two branches in the join — execute_plan
User: "Look up Apple and Google headcounts in parallel; write a comparison."
You:
``execute_plan(
    task="...",
    steps=[
        {"name": "hc_apple",  "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Apple in 2024",  "parallel": true},
        {"name": "hc_google", "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Google in 2024", "parallel": true},
        {"name": "report",    "agent": "writer",   "task_kind": "from_parallel",
         "task_step": "hc_apple",
         "context_kind": "from_parallel", "context_step": "hc_google"},
    ],
)``

## Pitfalls

- ``from_step`` / ``from_parallel`` must point at a step defined EARLIER in
  the list. Forward references are rejected at compile time.
- Step names must be unique within a plan.
- ``from_parallel`` reads ONE branch. To synthesise three or more parallel
  legs into a single answer, ``execute_plan`` is not the right tool — call
  the agents directly (one tool call each, or one combined call to a
  research-style agent that does the lookups internally).
- The tool result starting with ``PLAN_REJECTED`` is self-correctable —
  read the message, fix the spec, re-emit. Don't apologise to the user.
"""


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


def make_planner(
    agents: list[Agent],
    *,
    model: str = "claude-opus-4-7",
    system: Optional[str] = None,
    name: str = "planner",
    verbose: bool = False,
) -> Agent:
    """Build a planner :class:`Agent` over the given sub-agents.

    The returned agent has:
      - each sub-agent in ``agents`` as a direct tool (so it can call one
        when that's enough);
      - an ``execute_plan`` tool that materialises a typed PlanSpec into a
        real :class:`Plan` and runs it (with compile-time DAG validation).

    Args:
        agents: The sub-agents the planner may dispatch to. Each must have
            a unique ``.name``; the planner addresses them by that name in
            ``StepSpec.agent``.
        model: Provider model id for the planner LLM. Default
            ``"claude-opus-4-7"``.
        system: Override the planner's system prompt. By default we prepend
            "You are a generalist assistant." to :data:`PLANNER_GUIDANCE`
            so the LLM has decision rules and worked examples for
            ``execute_plan``.
        name: Display name for the planner agent.
        verbose: If True, print event traces to stdout.

    Returns:
        A configured planner :class:`Agent`. Call it with the user task.

    Raises:
        ValueError: if ``agents`` is empty or contains duplicate names.
    """
    if not agents:
        raise ValueError("make_planner: agents list must not be empty")
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError(f"make_planner: agents must have unique names; got {names}")

    registry = {a.name: a for a in agents}
    plan_tool = make_execute_plan_tool(registry)

    if system is None:
        system = "You are a generalist assistant.\n\n" + PLANNER_GUIDANCE

    return Agent(
        engine=LLMEngine(model, system=system),
        tools=[*agents, plan_tool],
        name=name,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    def web_search(query: str) -> str:
        """Look up current facts (stub)."""
        return f"[stub web result for {query!r}]"

    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

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
        "What does FAANG stand for?",                                # trivial
        "What is 17 * 23 + 5?",                                      # one agent
        "Research quantum networking and write a one-paragraph brief.",  # plan: pipeline
        "Look up Apple and Google headcounts in parallel; write a comparison.",  # plan: parallel + combine
    ]
    for q in queries:
        print(f"\n>>> {q}")
        print(planner(q).text())


if __name__ == "__main__":
    main()
