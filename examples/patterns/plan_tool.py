"""``make_planner`` — give an LLM sub-agents and an incremental Plan builder.

Pass your sub-agents, get back a planner :class:`lazybridge.Agent`. The
planner has each sub-agent as a direct tool (call one when one is enough)
plus five **builder** tools that compose a Plan one step at a time:

  - ``create_plan(reasoning)``                            → plan_id
  - ``add_step(plan_id, name, agent, …, parallel)``       → ok | error
  - ``inspect_plan(plan_id)``                             → current shape
  - ``run_plan(plan_id, task)``                           → result text
  - ``discard_plan(plan_id)``                             → ok

Why incremental instead of one atomic ``execute_plan``?
- Validation is **local**: each ``add_step`` rejects immediately with a
  pointed error (unknown agent, duplicate name, forward ``from_step``
  reference, missing ``task_text``…). The LLM corrects that single step
  rather than re-emitting the whole DAG.
- The LLM can ``inspect_plan`` between additions to refresh its context.
- No nested JSON: each tool call is one primitive operation.

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

Honest about a limitation
-------------------------
``from_parallel("name")`` is an alias for ``from_step`` — it forwards a
single specific branch's envelope, not a list of all parallel siblings.
A join step can combine at most two branches (one as ``task``, one as
``context``). For "fan out N legs and synthesise all of them", call the
sub-agents directly instead.
"""

import time
import uuid
from dataclasses import dataclass, field
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

    reasoning: str = Field(
        ...,
        description=(
            "Think first, in prose. Why these steps? Which sub-agents and "
            "why? What's the simplest shape that fits? This field is "
            "REQUIRED — empty or boilerplate values defeat its purpose."
        ),
    )
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


# ---------------------------------------------------------------------------
# Builder state
# ---------------------------------------------------------------------------


@dataclass
class _PlanInProgress:
    plan_id: str
    reasoning: str
    steps: list[StepSpec] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


def _format_progress(p: _PlanInProgress) -> str:
    if not p.steps:
        return f"plan {p.plan_id}: 0 steps. reasoning: {p.reasoning}"
    lines = [f"plan {p.plan_id}: {len(p.steps)} step(s). reasoning: {p.reasoning}"]
    for i, s in enumerate(p.steps):
        flags = []
        if s.parallel:
            flags.append("parallel")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        src_bits = []
        if s.task_kind == "literal":
            src_bits.append(f'task="{(s.task_text or "")[:40]}"')
        elif s.task_kind in ("from_step", "from_parallel"):
            src_bits.append(f"{s.task_kind}({s.task_step!r})")
        # default: from_prev
        if s.context_kind:
            src_bits.append(f"ctx={s.context_kind}({s.context_step!r})")
        sources = " " + ", ".join(src_bits) if src_bits else ""
        lines.append(f"  {i}. {s.name} → {s.agent}{flag_str}{sources}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers (run inside add_step before mutating state)
# ---------------------------------------------------------------------------


def _validate_step_addition(
    pip: _PlanInProgress,
    name: str,
    agent: str,
    task_kind: str,
    task_text: Optional[str],
    task_step: Optional[str],
    context_kind: Optional[str],
    context_step: Optional[str],
    registry: dict[str, Agent],
) -> Optional[str]:
    """Return an error string, or ``None`` if the step is valid."""
    if agent not in registry:
        return (
            f"unknown agent {agent!r}. Available: {sorted(registry)!r}."
        )
    existing_names = {s.name for s in pip.steps}
    if name in existing_names:
        return f"duplicate step name {name!r}; existing: {sorted(existing_names)!r}."
    if task_kind == "literal":
        if not task_text:
            return f"task_kind='literal' requires task_text."
    elif task_kind in ("from_step", "from_parallel"):
        if not task_step:
            return f"task_kind={task_kind!r} requires task_step."
        if task_step not in existing_names:
            return (
                f"task_step={task_step!r} not yet defined "
                f"(existing: {sorted(existing_names)!r}). Add that step first."
            )
    elif task_kind != "from_prev":
        return (
            f"task_kind must be one of 'literal' / 'from_prev' / "
            f"'from_step' / 'from_parallel'; got {task_kind!r}."
        )
    if context_kind is not None:
        if context_kind not in ("from_step", "from_parallel"):
            return (
                f"context_kind must be 'from_step' or 'from_parallel'; "
                f"got {context_kind!r}."
            )
        if not context_step:
            return f"context_kind={context_kind!r} requires context_step."
        if context_step not in existing_names:
            return (
                f"context_step={context_step!r} not yet defined "
                f"(existing: {sorted(existing_names)!r})."
            )
    return None


# ---------------------------------------------------------------------------
# Builder tool factory
# ---------------------------------------------------------------------------


def make_plan_builder_tools(
    registry: dict[str, Agent],
    *,
    max_plans: int = 50,
) -> list[Tool]:
    """Five builder tools that share state via closure.

    Returns ``[create_plan, add_step, inspect_plan, run_plan, discard_plan]``.

    The state is per-factory-instance — call ``make_plan_builder_tools``
    fresh for each planner agent (or each session) if you want isolated
    blackboards. ``run_plan`` and ``discard_plan`` consume the plan from
    the dict, so memory stays bounded as long as the planner finishes its
    plans. ``max_plans`` is a hard cap on concurrent in-progress plans
    (oldest-evicted on overflow) so a misbehaving planner can't leak memory.
    """
    if not registry:
        raise ValueError("plan tool registry must contain at least one agent")

    plans: dict[str, _PlanInProgress] = {}

    def _evict_if_full() -> None:
        if len(plans) >= max_plans:
            # Drop the oldest in-progress plan.
            oldest = min(plans.values(), key=lambda p: p.created_at)
            plans.pop(oldest.plan_id, None)

    # --- create_plan -----------------------------------------------------
    def create_plan(reasoning: str) -> str:
        """Start a new empty plan. Returns the plan_id to use in subsequent calls.

        Args:
            reasoning: Why this plan; which sub-agents and why; simplest
                shape that fits. Required — empty / boilerplate defeats
                the point of thinking first.
        """
        if not reasoning or not reasoning.strip():
            return (
                "REJECTED: reasoning is required and must be non-empty. "
                "Briefly state why this plan shape fits the task."
            )
        _evict_if_full()
        pid = uuid.uuid4().hex[:8]
        plans[pid] = _PlanInProgress(plan_id=pid, reasoning=reasoning.strip())
        return (
            f"plan_id={pid} (empty; add steps with add_step, then run_plan). "
            f"Available sub-agents: {sorted(registry)!r}."
        )

    # --- add_step --------------------------------------------------------
    def add_step(
        plan_id: str,
        name: str,
        agent: str,
        task_kind: Literal["literal", "from_prev", "from_step", "from_parallel"] = "from_prev",
        task_text: Optional[str] = None,
        task_step: Optional[str] = None,
        context_kind: Optional[Literal["from_step", "from_parallel"]] = None,
        context_step: Optional[str] = None,
        parallel: bool = False,
    ) -> str:
        """Append one step to a plan; validated immediately.

        On rejection, the plan is unchanged — fix the args and call again.

        Args:
            plan_id: From a prior ``create_plan``.
            name: Unique snake_case identifier within this plan.
            agent: Sub-agent name (must exist in the registry).
            task_kind: ``literal`` (use ``task_text``) / ``from_prev``
                (default; previous step's output) / ``from_step``
                (named earlier step's output) / ``from_parallel``
                (alias of ``from_step``, naming is for readability).
            task_text: Required when ``task_kind="literal"``.
            task_step: Required when ``task_kind`` is ``from_step`` or
                ``from_parallel``; must name an earlier step.
            context_kind: Optional secondary input pulled into the step's
                context. Useful to combine TWO parallel branches.
            context_step: Required when ``context_kind`` is set.
            parallel: ``true`` to run concurrently with adjacent
                ``parallel=true`` siblings.
        """
        if plan_id not in plans:
            return f"REJECTED: unknown plan_id {plan_id!r}."
        pip = plans[plan_id]
        err = _validate_step_addition(
            pip, name, agent, task_kind, task_text, task_step,
            context_kind, context_step, registry,
        )
        if err:
            return f"REJECTED: {err}"
        pip.steps.append(StepSpec(
            name=name,
            agent=agent,
            task_kind=task_kind,
            task_text=task_text,
            task_step=task_step,
            context_kind=context_kind,
            context_step=context_step,
            parallel=parallel,
        ))
        return f"ok ({len(pip.steps)} step(s) in plan {plan_id})"

    # --- inspect_plan ----------------------------------------------------
    def inspect_plan(plan_id: str) -> str:
        """Show the plan's current shape — useful between additions."""
        if plan_id not in plans:
            return f"REJECTED: unknown plan_id {plan_id!r}."
        return _format_progress(plans[plan_id])

    # --- run_plan --------------------------------------------------------
    async def run_plan(plan_id: str, task: str) -> str:
        """Materialise and run the plan; returns the final step's text.

        Consumes the plan (it's removed from the in-progress dict). To
        run again, build a new plan.
        """
        if plan_id not in plans:
            return f"REJECTED: unknown plan_id {plan_id!r}."
        pip = plans.pop(plan_id)
        if not pip.steps:
            return f"REJECTED: plan {plan_id} has no steps. Add at least one before running."
        spec = PlanSpec(reasoning=pip.reasoning, task=task, steps=pip.steps)
        try:
            plan = _materialize(spec, registry)
        except _PlanToolError as e:
            return f"PLAN_REJECTED: {e}"
        try:
            runner = Agent.from_engine(plan)  # PlanCompiler defense-in-depth.
        except PlanCompileError as e:
            return _format_compile_error(e, registry)
        try:
            env = await runner.run(spec.task)
        except Exception as e:  # noqa: BLE001
            return f"PLAN_RUNTIME_ERROR: {type(e).__name__}: {e}"
        if env.error:
            return f"PLAN_RUNTIME_ERROR: {env.error.message}"
        return env.text()

    # --- discard_plan ----------------------------------------------------
    def discard_plan(plan_id: str) -> str:
        """Drop an in-progress plan without running it."""
        if plan_id not in plans:
            return f"REJECTED: unknown plan_id {plan_id!r}."
        plans.pop(plan_id)
        return f"ok (plan {plan_id} discarded)"

    # Customise descriptions so the LLM sees the registry inline.
    agents_summary = "Available sub-agents:\n" + "\n".join(
        f"- {n}: {(a.description or '').strip() or 'no description'}"
        for n, a in registry.items()
    )

    add_step.__doc__ = (add_step.__doc__ or "") + "\n\n" + agents_summary
    create_plan.__doc__ = (create_plan.__doc__ or "") + "\n\n" + agents_summary

    return [
        Tool(create_plan, mode="signature"),
        Tool(add_step, mode="signature"),
        Tool(inspect_plan, mode="signature"),
        Tool(run_plan, mode="signature"),
        Tool(discard_plan, mode="signature"),
    ]


# Backwards-compatible alias — kept so docs / external imports don't break.
def make_execute_plan_tool(registry: dict[str, Agent], **_unused: Any) -> list[Tool]:
    """Deprecated alias for :func:`make_plan_builder_tools`. Returns the
    list of builder tools rather than a single Tool."""
    return make_plan_builder_tools(registry)


# ---------------------------------------------------------------------------
# Guidance — drop into the planner's system prompt
# ---------------------------------------------------------------------------

PLANNER_GUIDANCE = """\
# How to handle a request

You have a set of specialist sub-agents available as direct tools, plus
five **builder** tools that compose a Plan one step at a time:

- ``create_plan(reasoning)``                      → returns plan_id
- ``add_step(plan_id, name, agent, ...)``         → ok | REJECTED: <hint>
- ``inspect_plan(plan_id)``                       → current shape
- ``run_plan(plan_id, task)``                     → final result
- ``discard_plan(plan_id)``                       → drop without running

The builder validates each step locally — wrong agent name, duplicate
step name, forward ``from_step`` reference, missing ``task_text`` — so
you fix one error at a time instead of re-emitting a whole DAG.

## Five principles

1. **Think first, then structure.** ``create_plan`` requires a
   ``reasoning`` argument. Briefly say which agents and why, the simplest
   shape that fits, what could go wrong. Empty / boilerplate defeats the
   point.
2. **Coarse steps, not micro-steps.** Each ``add_step`` is one sub-agent
   doing one meaningful unit of work. Prefer 2-4 step plans.
3. **Re-plan, don't perfect-plan.** Two short plans beat one speculative
   big one. After ``run_plan`` returns, you can ``create_plan`` again
   with what you've learned.
4. **Verify the answer addresses the question.** Before replying to the
   user, check the final result actually answers what was asked. If not,
   plan another round.
5. **Prefer the simpler shape.** Direct sub-agent call > linear plan >
   parallel band > parallel band with combined branches.

## Decision rules

1. **Trivial query** — answer directly. No tool call.
2. **One sub-agent suffices** — call that sub-agent directly. No plan.
3. **Multiple steps with dependencies** — build a plan with the builder.
4. **Big or uncertain task** — short scout plan, then re-plan.

## Builder workflow

Typical sequence for a multi-step task:

```
pid = create_plan(reasoning="...")           # returns plan_id
add_step(pid, name="gather", agent="research")
add_step(pid, name="draft",  agent="writer") # task_kind="from_prev" by default
inspect_plan(pid)                            # optional sanity check
run_plan(pid, task="<the user's task>")      # returns the final text
```

If ``add_step`` returns ``REJECTED: <reason>``, the plan is unchanged —
just call ``add_step`` again with corrected args. Read the hint; it tells
you exactly what's wrong.

## ``add_step`` field reference

- ``name``      : unique snake_case identifier within the plan.
- ``agent``     : sub-agent name (must exist in the registry).
- ``task_kind`` :
    * ``"literal"``       — provide ``task_text``. Use for hand-crafted prompts.
    * ``"from_prev"``     — output of the preceding step (default).
    * ``"from_step"``     — output of the step named in ``task_step``;
                            must be a step you've already added.
    * ``"from_parallel"`` — alias of ``from_step``; readable name when the
                            referenced step ran with ``parallel=true``.
                            Reads ONE specific branch.
- ``task_text`` : required when ``task_kind="literal"``.
- ``task_step`` : required when ``task_kind`` is ``from_step`` /
                  ``from_parallel``.
- ``context_kind`` / ``context_step`` : optional secondary input pulled
                  into the step's context. Use to combine TWO parallel
                  branches in a join step (one as task, one as context).
- ``parallel``  : ``true`` to run concurrently with adjacent
                  ``parallel=true`` siblings.

## Worked examples

### Trivial — answer directly
User: "What does FAANG stand for?"
You: "FAANG = Facebook (Meta), Apple, Amazon, Netflix, Google."

### One agent — call it directly
User: "What is 17 * 23 + 5?"
You: call ``math("Compute 17 * 23 + 5.")``

### Sequential pipeline — builder
User: "Research quantum networking and write a one-paragraph brief."
```
pid = create_plan(reasoning="Two-step pipeline: research gathers facts, writer turns them into prose.")
add_step(pid, name="gather", agent="research")
add_step(pid, name="draft",  agent="writer")
run_plan(pid, task="Quantum networking")
```

### Parallel band + read one branch — builder
User: "Look up Apple and Google headcounts in parallel; report Apple's."
```
pid = create_plan(reasoning="Two parallel lookups, then writer reads only Apple per user instruction.")
add_step(pid, name="hc_apple",  agent="research", task_kind="literal", task_text="headcount of Apple in 2024",  parallel=True)
add_step(pid, name="hc_google", agent="research", task_kind="literal", task_text="headcount of Google in 2024", parallel=True)
add_step(pid, name="report",    agent="writer",   task_kind="from_parallel", task_step="hc_apple")
run_plan(pid, task="...")
```

### Parallel band + combine two branches — builder
User: "Look up Apple and Google headcounts in parallel; write a comparison."
```
pid = create_plan(reasoning="Two parallel lookups feed a comparison writer; Apple as task, Google as context.")
add_step(pid, name="hc_apple",  agent="research", task_kind="literal", task_text="headcount of Apple in 2024",  parallel=True)
add_step(pid, name="hc_google", agent="research", task_kind="literal", task_text="headcount of Google in 2024", parallel=True)
add_step(pid, name="report",    agent="writer",
         task_kind="from_parallel", task_step="hc_apple",
         context_kind="from_parallel", context_step="hc_google")
run_plan(pid, task="...")
```

### Big task — scout, then re-plan
User: "Build a competitive analysis of the top 5 AI agent frameworks."
First plan (scout):
```
pid = create_plan(reasoning="I don't yet know which 5 frameworks. Scout first; re-plan with the list in hand.")
add_step(pid, name="scout", agent="research", task_kind="literal",
         task_text="List the top 5 AI agent frameworks in 2026 with one-line descriptions.")
result = run_plan(pid, task="Top AI agent frameworks 2026")
```
Then, with ``result`` in hand, ``create_plan`` again to fan out the
per-framework deep-dives.

## Pitfalls

- **Add steps in dependency order.** ``from_step`` / ``from_parallel``
  must reference a step you've already added. ``add_step`` will reject
  forward references — just add the dependency first.
- **Don't forget ``run_plan``.** Building a plan without running it leaks
  it (it's discarded after a cap of 50 in-progress plans). If you change
  your mind, call ``discard_plan``.
- **``from_parallel`` reads ONE branch.** Three+ parallel legs that all
  need to feed one synthesis step? The builder isn't the right tool —
  call the sub-agents directly.
- **A ``REJECTED: <hint>`` from ``add_step`` is self-correctable.** Read
  the hint, call ``add_step`` again with the fix. Don't apologise to the
  user; just retry.
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
    verify: Optional[Agent] = None,
    max_verify: int = 3,
) -> Agent:
    """Build a planner :class:`Agent` over the given sub-agents.

    The returned agent has:
      - each sub-agent in ``agents`` as a direct tool (so it can call one
        when that's enough);
      - five **builder** tools (``create_plan``, ``add_step``,
        ``inspect_plan``, ``run_plan``, ``discard_plan``) that compose a
        :class:`Plan` one step at a time, with local validation per step.

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
        verify: Optional judge :class:`Agent` that vets the planner's final
            output. When set, the planner's response runs through
            ``verify`` (LazyBridge's built-in verify-with-retry loop). The
            judge should reply "approved" or "rejected: <reason>"; on
            rejection the planner retries up to ``max_verify`` times with
            the judge's feedback in context. Costs one extra LLM call per
            attempt — use it for tasks where wrong answers are expensive.
        max_verify: Max judge attempts when ``verify`` is set. Default 3.

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
    builder_tools = make_plan_builder_tools(registry)

    if system is None:
        system = "You are a generalist assistant.\n\n" + PLANNER_GUIDANCE

    return Agent(
        engine=LLMEngine(model, system=system),
        tools=[*agents, *builder_tools],
        name=name,
        verbose=verbose,
        verify=verify,
        max_verify=max_verify,
    )


# Suggested judge prompt for the verify= argument.
PLANNER_VERIFY_PROMPT = """\
You are a verification judge for a planner agent's output.

Read the user's original question and the planner's final answer. Reply with
EXACTLY one of:

- "approved" — the answer addresses the question fully and accurately.
- "rejected: <one-line reason>" — the answer misses part of the question,
  contradicts itself, contains unsupported claims, or is not actionable.

Do NOT rewrite the answer. Only judge it.
"""


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
