"""Orchestration tools — chain, parallel, plan — over a sub-agent registry.

See ``docs/recipes/orchestration-tools.md`` for the full guide
(decision rules, registry conventions, nested composition, error
recovery, pitfalls). This module is the implementation that doc points at.

Three reusable :class:`lazybridge.Tool` factories, ordered from simple to
expressive:

* ``make_execute_chain_tool``    — sequential pipeline (a → b → c).
* ``make_execute_parallel_tool`` — fan-out N independent jobs concurrently.
* ``make_execute_plan_tool``     — full DAG (mix of parallel + sequential).

Plus :data:`ORCHESTRATOR_GUIDANCE`, a long worked-examples document the
outer agent should drop into its system prompt — composing plans is a
non-trivial task and the LLM benefits from concrete patterns.

Usage
-----
::

    from examples.patterns.plan_tool import make_orchestration_tools, ORCHESTRATOR_GUIDANCE

    REGISTRY = {"research": research_agent, "math": math_agent, "writer": writer_agent}
    tools = make_orchestration_tools(REGISTRY)

    orchestrator = Agent(
        engine=LLMEngine(
            "claude-opus-4-7",
            system=("You are a generalist assistant.\n\n" + ORCHESTRATOR_GUIDANCE),
        ),
        tools=tools,
    )
    orchestrator("How many staff at FAANG companies and what does it imply?")

Contract
--------
The LLM calls ``execute_plan(steps=[...], task="...")``. The tool:

1. Validates the steps via Pydantic (auto-schema from ``StepSpec``/``PlanSpec``).
2. ``materialize()`` builds a real :class:`lazybridge.Plan` with live agents
   bound from the registry.
3. ``Agent.from_engine(plan)`` triggers ``PlanCompiler`` — forward
   ``from_step`` references, unknown step names, and duplicate names are
   rejected here, *before* any LLM call inside the plan runs.
4. The plan executes; the final step's text result is returned.
5. On any compile or runtime error, the tool returns a structured error
   message the LLM can read and self-correct from on its next attempt.

Why a tool, not a top-level helper
----------------------------------
Putting plan construction behind a tool means the *outer* agent's LLM
decides when to plan. It can mix simple tool calls (one-shot) with
composed plans (multi-step DAG) and skip planning altogether for trivial
queries — which is the right behaviour for a generalist assistant.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from lazybridge import Agent, Plan, Step, Tool, from_parallel, from_prev, from_step
from lazybridge.engines.plan import PlanCompileError

# ---------------------------------------------------------------------------
# Public contract — the LLM emits this shape via the tool's auto-schema.
# ---------------------------------------------------------------------------


class StepSpec(BaseModel):
    """One node in the plan DAG."""

    name: str = Field(..., description="Unique step identifier; referenced by from_step.")
    agent: str = Field(..., description="Sub-agent name from the registry.")
    task_kind: Literal["literal", "from_prev", "from_step", "from_parallel"] = Field(
        default="from_prev",
        description=(
            "literal=use task_text; from_prev=output of preceding step; "
            "from_step=output of step named in task_step; "
            "from_parallel=join list[Envelope] from a parallel sibling group "
            "starting at task_step."
        ),
    )
    task_text: Optional[str] = Field(
        default=None, description="Required when task_kind='literal'."
    )
    task_step: Optional[str] = Field(
        default=None,
        description="Required when task_kind='from_step' or 'from_parallel'.",
    )
    parallel: bool = Field(
        default=False,
        description="If True, run concurrently with adjacent parallel siblings.",
    )


class PlanSpec(BaseModel):
    """Full plan: an ordered list of steps + the user task."""

    task: str = Field(..., description="The user task that drives the plan.")
    steps: list[StepSpec] = Field(..., description="Ordered DAG steps.")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class PlanToolError(Exception):
    """Raised on bad spec; surfaced to the LLM as a tool result string."""


def _materialize(spec: PlanSpec, registry: dict[str, Agent]) -> Plan:
    if not spec.steps:
        raise PlanToolError("PlanSpec.steps is empty; emit at least one step or skip the plan.")

    unknown = sorted({s.agent for s in spec.steps if s.agent not in registry})
    if unknown:
        raise PlanToolError(
            f"Unknown agent name(s) {unknown!r}. "
            f"Available agents: {sorted(registry)!r}."
        )

    real_steps: list[Step] = []
    for s in spec.steps:
        if s.task_kind == "literal":
            if not s.task_text:
                raise PlanToolError(
                    f"Step {s.name!r}: task_kind='literal' requires task_text."
                )
            task_arg: Any = s.task_text
        elif s.task_kind == "from_step":
            if not s.task_step:
                raise PlanToolError(
                    f"Step {s.name!r}: task_kind='from_step' requires task_step."
                )
            task_arg = from_step(s.task_step)
        elif s.task_kind == "from_parallel":
            if not s.task_step:
                raise PlanToolError(
                    f"Step {s.name!r}: task_kind='from_parallel' requires "
                    "task_step pointing at the first parallel sibling."
                )
            task_arg = from_parallel(s.task_step)
        else:  # from_prev
            task_arg = from_prev

        real_steps.append(
            Step(
                target=registry[s.agent],
                name=s.name,
                task=task_arg,
                parallel=s.parallel,
            )
        )

    return Plan(*real_steps, max_iterations=max(20, len(real_steps) * 3))


def _format_compile_error(err: PlanCompileError, registry: dict[str, Agent]) -> str:
    return (
        "PLAN_REJECTED: " + str(err) + "\n\n"
        "Hints: from_step targets must be defined earlier in the DAG. "
        "Step names must be unique. "
        f"Valid agents: {sorted(registry)!r}. "
        "Re-emit a corrected PlanSpec."
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_execute_plan_tool(
    registry: dict[str, Agent],
    *,
    name: str = "execute_plan",
    description: Optional[str] = None,
) -> Tool:
    """Return a :class:`lazybridge.Tool` that builds and runs a Plan.

    The tool's signature is ``execute_plan(steps: list[StepSpec], task: str) -> str``
    — flat args so the model doesn't need an outer wrapper.

    Args:
        registry: Mapping ``{agent_name: Agent}`` the plan may dispatch to.
            The set of valid ``StepSpec.agent`` values is exactly ``registry.keys()``.
        name: Tool name visible to the LLM. Defaults to ``"execute_plan"``.
        description: Override the tool's LLM-facing description. The default
            includes the registry's agent names and a one-line summary of
            each agent (taken from ``Agent.description`` when set).

    Returns:
        A ``Tool`` ready to drop into ``Agent(tools=[...])``.

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
            "Compose and execute a multi-step plan over the available "
            "sub-agents. The plan is a DAG; declare parallel=True on "
            "siblings to run them concurrently. Returns the final step's "
            "text. Use this when the task requires more than one sub-agent "
            "or more than one step.\n\n"
            "Available sub-agents:\n" + "\n".join(agent_lines)
        )

    async def execute_plan(steps: list[StepSpec], task: str) -> str:
        """Build and run a plan over the registered sub-agents."""
        spec = PlanSpec(task=task, steps=steps)
        try:
            plan = _materialize(spec, registry)
        except PlanToolError as e:
            return f"PLAN_REJECTED: {e}"

        try:
            runner = Agent.from_engine(plan)  # PlanCompiler runs here
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
# Simpler shapes — chain and parallel
# ---------------------------------------------------------------------------


class ParallelJob(BaseModel):
    """One leg of a parallel fan-out: an agent name + the task to run on it."""

    agent: str = Field(..., description="Sub-agent name from the registry.")
    task: str = Field(..., description="Self-contained task for that sub-agent.")


def make_execute_chain_tool(
    registry: dict[str, Agent],
    *,
    name: str = "execute_chain",
    description: Optional[str] = None,
) -> Tool:
    """Tool: ``execute_chain(agents: list[str], task: str) -> str``.

    Runs sub-agents sequentially: each agent receives the *previous* agent's
    text output as its task; the first agent receives ``task``. Returns the
    last agent's text. Use when the work is a strict pipeline (research →
    summarise → critique) with no fan-out.
    """
    if not registry:
        raise ValueError("chain tool registry must contain at least one agent")

    if description is None:
        agent_lines = [
            f"- {n}: {(a.description or '').strip() or 'no description'}"
            for n, a in registry.items()
        ]
        description = (
            "Run sub-agents sequentially: each receives the previous agent's "
            "text output as its task. Use when the work is a strict pipeline "
            "with no parallel branches.\n\nAvailable sub-agents:\n"
            + "\n".join(agent_lines)
        )

    async def execute_chain(agents: list[str], task: str) -> str:
        unknown = [a for a in agents if a not in registry]
        if unknown:
            return (
                f"CHAIN_REJECTED: unknown agent(s) {unknown!r}. "
                f"Available: {sorted(registry)!r}."
            )
        if not agents:
            return "CHAIN_REJECTED: agents list is empty."

        chain_agent = Agent.chain(*(registry[a] for a in agents))
        try:
            env = await chain_agent.run(task)
        except Exception as e:  # noqa: BLE001
            return f"CHAIN_RUNTIME_ERROR: {type(e).__name__}: {e}"
        if env.error:
            return f"CHAIN_RUNTIME_ERROR: {env.error.message}"
        return env.text()

    return Tool(execute_chain, name=name, description=description, mode="signature")


def make_execute_parallel_tool(
    registry: dict[str, Agent],
    *,
    name: str = "execute_parallel",
    description: Optional[str] = None,
) -> Tool:
    """Tool: ``execute_parallel(jobs: list[ParallelJob]) -> str``.

    Runs each (agent, task) pair concurrently via :func:`asyncio.gather`.
    Returns the joined results, one section per job, labelled by agent.
    Use when the legs are independent and you don't need the framework
    to enforce an ordering — for fan-out+join with a synthesise step on
    top, prefer ``execute_plan`` so the join sees ``list[Envelope]``.
    """
    import asyncio

    if not registry:
        raise ValueError("parallel tool registry must contain at least one agent")

    if description is None:
        agent_lines = [
            f"- {n}: {(a.description or '').strip() or 'no description'}"
            for n, a in registry.items()
        ]
        description = (
            "Run multiple (agent, task) jobs concurrently. Each leg is "
            "independent. Returns the joined text, one labelled section per "
            "job. Use when the work fans out into independent lookups or "
            "computations and a separate synthesis step is unnecessary.\n\n"
            "Available sub-agents:\n" + "\n".join(agent_lines)
        )

    async def execute_parallel(jobs: list[ParallelJob]) -> str:
        if not jobs:
            return "PARALLEL_REJECTED: jobs list is empty."
        unknown = sorted({j.agent for j in jobs if j.agent not in registry})
        if unknown:
            return (
                f"PARALLEL_REJECTED: unknown agent(s) {unknown!r}. "
                f"Available: {sorted(registry)!r}."
            )

        try:
            envs = await asyncio.gather(
                *(registry[j.agent].run(j.task) for j in jobs)
            )
        except Exception as e:  # noqa: BLE001
            return f"PARALLEL_RUNTIME_ERROR: {type(e).__name__}: {e}"

        sections = []
        for j, env in zip(jobs, envs):
            label = f"[{j.agent}] {j.task}"
            body = env.text() if not env.error else f"(error) {env.error.message}"
            sections.append(f"{label}\n→ {body}")
        return "\n\n".join(sections)

    return Tool(execute_parallel, name=name, description=description, mode="signature")


def make_orchestration_tools(registry: dict[str, Agent]) -> list[Tool]:
    """Return [execute_chain, execute_parallel, execute_plan] for the registry.

    Convenience for the common case: drop all three into the outer agent's
    ``tools=[...]`` and let the LLM pick the simplest fit.
    """
    return [
        make_execute_chain_tool(registry),
        make_execute_parallel_tool(registry),
        make_execute_plan_tool(registry),
    ]


# ---------------------------------------------------------------------------
# Guidance — drop this into the orchestrator's system prompt.
#
# Composing a plan is non-trivial; the LLM benefits from concrete patterns.
# The doc below is structured: decision rules → tool reference → 8 worked
# examples (one per common shape) → pitfalls.
# ---------------------------------------------------------------------------

ORCHESTRATOR_GUIDANCE = """\
# How to compose work over the sub-agents

You have three orchestration tools, ordered from simple to expressive.
**Always pick the simplest tool that fits the task.**

| Tool              | Shape                              | When to use                                                |
|-------------------|------------------------------------|------------------------------------------------------------|
| (none — answer)   | direct response                    | Greetings, common knowledge, pure opinion.                 |
| execute_chain     | a → b → c (sequential)             | Strict pipeline, each step builds on previous output.      |
| execute_parallel  | [a, b, c] (independent)            | Independent legs; you don't need a separate synthesise step. |
| execute_plan      | DAG (parallel + sequential mix)    | Fan-out + join, conditional ordering, multi-stage work.    |

## Decision rules

1. **Trivial query** → answer directly. Don't call a tool.
2. **One sub-agent suffices** → call ``execute_chain`` with a single-element list,
   or — if the sub-agent is exposed directly — just call it.
3. **All steps depend on the previous output** → ``execute_chain``.
4. **Steps are independent and you can paste their text together** → ``execute_parallel``.
5. **Fan-out then synthesise** (need a join step that sees all parallel outputs) →
   ``execute_plan`` with ``parallel=true`` siblings + a ``from_parallel`` join step.
6. **Mixed DAG** (some parallel, some sequential, some references back) → ``execute_plan``.

## Tool reference

### ``execute_chain(agents: list[str], task: str) -> str``
Sequential pipeline. ``agents[0]`` receives ``task``. Each subsequent agent
receives the previous agent's text output. Returns the last agent's text.

### ``execute_parallel(jobs: list[{agent, task}]) -> str``
Runs every ``(agent, task)`` pair concurrently. Returns labelled sections
joined together. There is no separate synthesise step; if you need one,
use ``execute_plan`` instead.

### ``execute_plan(steps: list[StepSpec], task: str) -> str``
Builds and runs a DAG. Each ``StepSpec`` has:

- ``name``       : unique identifier (snake_case).
- ``agent``      : sub-agent name from the registry.
- ``task_kind``  : one of:
    * ``"literal"``       — provide ``task_text``. Use for the first step or
                            for steps that need a hand-crafted prompt.
    * ``"from_prev"``     — receive the previous step's text output (default).
    * ``"from_step"``     — receive the named step's output (provide ``task_step``).
                            The named step **must come earlier** in the DAG.
    * ``"from_parallel"`` — receive a ``list[Envelope]`` from a group of
                            parallel siblings whose first member is named
                            in ``task_step``. Use this for the join step
                            after a fan-out.
- ``task_text`` : required when ``task_kind="literal"``.
- ``task_step`` : required when ``task_kind="from_step"`` or ``"from_parallel"``.
- ``parallel``  : ``true`` to run concurrently with adjacent parallel siblings.

## Worked examples

### Example 1 — trivial: answer directly
User: "What does FAANG stand for?"
You: "FAANG stands for Facebook (Meta), Apple, Amazon, Netflix, Google."
(no tool call)

### Example 2 — single agent: execute_chain with one element
User: "What's the headcount of Apple in 2024?"
Tool: ``execute_chain(agents=["research"], task="Find the headcount of Apple in 2024.")``

### Example 3 — pipeline: execute_chain
User: "Research recent agent frameworks and write a one-paragraph summary."
Tool:
``execute_chain(
    agents=["research", "writer"],
    task="Recent agent frameworks (2025-2026)",
)``

### Example 4 — independent fan-out: execute_parallel
User: "Headcounts of Apple, Google, and Meta in 2024."
Tool:
``execute_parallel(jobs=[
    {"agent": "research", "task": "headcount of Apple in 2024"},
    {"agent": "research", "task": "headcount of Google in 2024"},
    {"agent": "research", "task": "headcount of Meta in 2024"},
])``

### Example 5 — fan-out + join + write: execute_plan
User: "Get the 2024 headcounts of Apple, Google, and Meta in parallel, sum them, and write a short comment."
Tool:
``execute_plan(
    task="...",
    steps=[
        {"name": "hc_apple",  "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Apple in 2024",  "parallel": true},
        {"name": "hc_google", "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Google in 2024", "parallel": true},
        {"name": "hc_meta",   "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Meta in 2024",  "parallel": true},
        {"name": "sum",       "agent": "math",
         "task_kind": "from_parallel", "task_step": "hc_apple"},
        {"name": "comment",   "agent": "writer",   "task_kind": "from_prev"},
    ],
)``
Note: the join step uses ``task_kind="from_parallel"`` with
``task_step="hc_apple"`` — the *first* parallel sibling names the group.

### Example 6 — branching with from_step
User: "Look up X, then in parallel write a marketing blurb and a technical brief from those facts."
Tool:
``execute_plan(
    task="...",
    steps=[
        {"name": "facts",  "agent": "research", "task_kind": "literal",
         "task_text": "Look up X."},
        {"name": "marketing", "agent": "writer", "task_kind": "from_step",
         "task_step": "facts", "parallel": true},
        {"name": "tech",      "agent": "writer", "task_kind": "from_step",
         "task_step": "facts", "parallel": true},
    ],
)``

### Example 7 — pure math, no research
User: "What is 17 * 23 + 5?"
Tool: ``execute_chain(agents=["math"], task="Compute 17 * 23 + 5.")``

### Example 8 — refused / impossible
If the registry has no agent for the work (e.g. a coding question with only
research/math/writer agents), say so to the user instead of inventing tasks.

### Example 9 — nested composition (registry entry is itself composed)
Registry entries can themselves be ``Agent.chain(...)``, ``Agent.parallel(...)``,
or ``Agent.from_engine(Plan(...))`` — these are still single agents from your
point of view. You pick them by name like any other; the inner composition
runs transparently and rolls its cost/tokens into the outer envelope.

For instance, if the registry has ``"deep_research"`` defined upstream as
``Agent.chain(searcher, fact_checker, summariser)`` and ``"multi_source"``
defined as ``Agent.parallel(google, bing, arxiv)``:

User: "Deep-research and write a brief on quantum networking."
Tool:
``execute_chain(agents=["deep_research", "writer"], task="Quantum networking")``

User: "Get headcounts for FAANG via every search source we have and total them."
Tool:
``execute_plan(
    task="...",
    steps=[
        {"name": "facts", "agent": "multi_source", "task_kind": "literal",
         "task_text": "FAANG headcounts 2024 from every source"},
        {"name": "total", "agent": "math", "task_kind": "from_prev"},
    ],
)``

You don't recurse into the composed agents' internals — they are leaves to you.

## Nested composition

There are two layers of nesting. Both are available to you.

### Layer 1 — pre-built registry entries (someone else composed them)
The registry can contain entries that are themselves composed agents:

- ``Agent.chain(a, b, c)`` — one agent that runs three internally in sequence.
- ``Agent.parallel(a, b, c)`` — one agent that runs three internally on the
  same task, returns the joined output.
- ``Agent.from_engine(Plan(...))`` — one agent backed by a full DAG.

From your perspective, all three are just an entry in the registry with a
name and a description. The nested cost / tokens / errors propagate up
automatically (LazyBridge folds them into ``Envelope.metadata.nested_*``).
Pick the simplest layer that fits — don't replicate inside ``execute_plan``
what a registry entry already does internally.

### Layer 2 — *you* compose at runtime via ``execute_plan``
You can build nested structures yourself: ``execute_plan`` is the universal
DAG expression, so any chain, any parallel, and any combination of the two
fits inside a single ``execute_plan`` call. Use ``parallel=true`` on adjacent
steps to fan out, and ``task_kind="from_parallel"`` on the next step to join.

### Example 10 — chain with a parallel block in the middle (you build it)
Pattern: ``a → [b ∥ c] → d``. ``a`` is sequential; ``b`` and ``c`` run
concurrently from a's output; ``d`` joins them.

Tool:
``execute_plan(
    task="...",
    steps=[
        {"name": "a", "agent": "research", "task_kind": "literal",
         "task_text": "Find background on X"},
        {"name": "b", "agent": "research", "task_kind": "from_step",
         "task_step": "a", "parallel": true},
        {"name": "c", "agent": "research", "task_kind": "from_step",
         "task_step": "a", "parallel": true},
        {"name": "d", "agent": "writer", "task_kind": "from_parallel",
         "task_step": "b"},
    ],
)``

### Example 11 — parallel of mini-pipelines (you build it)
Pattern: ``[(a → b) ∥ (c → d)] → e``. Two short pipelines run concurrently,
then merge.

Tool:
``execute_plan(
    task="...",
    steps=[
        {"name": "a", "agent": "research", "task_kind": "literal",
         "task_text": "...", "parallel": true},
        {"name": "b", "agent": "writer", "task_kind": "from_step",
         "task_step": "a", "parallel": true},
        {"name": "c", "agent": "research", "task_kind": "literal",
         "task_text": "...", "parallel": true},
        {"name": "d", "agent": "writer", "task_kind": "from_step",
         "task_step": "c", "parallel": true},
        {"name": "e", "agent": "writer", "task_kind": "from_parallel",
         "task_step": "a"},
    ],
)``
The trick: every step that should run in the parallel band gets ``parallel=true``.
``from_step`` references inside the band wire the mini-pipelines; the join
step ``e`` uses ``from_parallel`` pointing at the *first* step of the band.

### Bottom line
- Want a flat sequential pipeline? ``execute_chain``.
- Want a flat fan-out with no synthesis? ``execute_parallel``.
- Want anything else, including nested composition? ``execute_plan`` —
  build the DAG yourself, no recursion needed.

## Pitfalls

- **Forward references**: ``from_step``/``from_parallel`` must point at a
  step that appears *earlier* in the list. The plan compiler rejects forward refs.
- **Duplicate step names**: every step must have a unique ``name``.
- **First step needs ``literal``**: the first step has no predecessor, so
  ``from_prev`` reads the user task directly — that's fine — but if you
  want a hand-crafted prompt, use ``literal`` with ``task_text``.
- **Don't fan out single-agent work**: if all "parallel" jobs would call
  the same agent with similar tasks that the agent could batch, just write
  one task that asks for the batched answer.
- **Don't plan for trivia**: a one-line factual question doesn't need a plan.
- **The tool result starting with PLAN_REJECTED / CHAIN_REJECTED / PARALLEL_REJECTED**:
  read the message, fix the spec, re-emit the tool call. Don't apologise to the user;
  fix and retry.
"""


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _demo_registry() -> dict[str, Agent]:
    """Build a small demo registry. Stub tools so the example runs without keys.

    Demonstrates **nested composition**: ``deep_research`` is itself an
    ``Agent.chain`` of two leaf agents, and ``multi_source`` is an
    ``Agent.parallel`` of three. The orchestrator picks them by name like
    any other registry entry — it doesn't need to know they're composed.
    """
    from lazybridge import LLMEngine

    def web_search(query: str) -> str:
        """Look up current facts (stub)."""
        return f"[stub web result for {query!r}]"

    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    # Leaf agents
    research = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Look up facts via web_search."),
        tools=[web_search],
        name="research",
        description="Web lookups for current facts. No math.",
    )
    fact_checker = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Verify claims; flag uncertainty."),
        name="fact_checker",
        description="Checks the previous step's claims, flags weak ones.",
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

    # Nested composition — looks like a single agent to the orchestrator.
    # ``Agent.chain(...)`` returns a regular Agent → drops into the registry as-is.
    deep_research = Agent.chain(research, fact_checker)
    deep_research.name = "deep_research"
    deep_research.description = (
        "Two-step research pipeline: web lookup followed by fact-checking. "
        "Use when the answer needs to be defensible. Single call."
    )

    # ``Agent.parallel(...)`` returns a ``_ParallelAgent`` whose ``run()`` yields
    # ``list[Envelope]`` rather than a single Envelope. That's incompatible with
    # ``Step.target`` (which expects one envelope per call), so we wrap it as a
    # leaf Agent that joins the parallel outputs into a single text envelope.
    async def _multi_source_run(task: str) -> str:
        envs = await Agent.parallel(research, research).run(task)
        return "\n\n---\n\n".join(e.text() for e in envs)

    multi_source = Agent(
        engine=LLMEngine("claude-opus-4-7", system="You synthesise multi-source results."),
        tools=[_multi_source_run],
        name="multi_source",
        description=(
            "Run the research agent across multiple sources concurrently and "
            "synthesise their findings. Good for breadth queries."
        ),
    )

    return {
        "research": research,
        "math": math,
        "writer": writer,
        "deep_research": deep_research,    # nested chain — drops in as-is
        "multi_source": multi_source,      # parallel wrapped as a single agent
    }


def main() -> None:
    from lazybridge import LLMEngine

    registry = _demo_registry()

    orchestrator = Agent(
        engine=LLMEngine(
            "claude-opus-4-7",
            system="You are a generalist assistant.\n\n" + ORCHESTRATOR_GUIDANCE,
        ),
        tools=make_orchestration_tools(registry),
        verbose=True,
    )

    queries = [
        # Trivial — should not use any tool.
        "What does FAANG stand for?",
        # Pipeline — execute_chain.
        "Research recent agent frameworks and write a one-paragraph summary.",
        # Independent fan-out — execute_parallel.
        "Headcounts of Apple, Google, and Meta in 2024.",
        # Fan-out + join + write — execute_plan.
        "Get the 2024 headcounts of Apple, Google, and Meta in parallel, "
        "sum them, and write a short paragraph commenting on it.",
    ]
    for q in queries:
        print(f"\n>>> {q}")
        print(orchestrator(q).text())


if __name__ == "__main__":
    main()
