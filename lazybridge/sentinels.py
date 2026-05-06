"""Sentinels — typed markers for Step input/context resolution in PlanEngine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _FromPrev:
    """Use the Envelope produced by the previous step (default)."""

    pass


@dataclass(frozen=True)
class _FromStart:
    """Use the initial task/context Envelope passed to the Plan."""

    pass


@dataclass(frozen=True)
class _FromStep:
    """Use the Envelope produced by a named step."""

    name: str


@dataclass(frozen=True)
class _FromParallel:
    """Use the Envelope produced by a specific parallel branch.

    Alias of :class:`_FromStep` — forwards a single branch's envelope.
    For aggregating ALL siblings in a parallel band, use
    :class:`_FromParallelAll` (``from_parallel_all("name")``).
    """

    name: str


@dataclass(frozen=True)
class _FromParallelAll:
    """Aggregate every consecutive parallel sibling starting at ``name``.

    The runtime walks the contiguous block of ``parallel=True`` steps that
    begins at the named step (in declared order) and returns ONE envelope
    that carries all of them:

    - ``task``     : labelled-text join
                     (``"[branch_a]\\n<text>\\n\\n[branch_b]\\n<text>..."``)
                     so ordinary LLM steps consume it without changes.
    - ``payload``  : the same labelled-text join string as ``task`` — NOT
                     a ``list[Envelope]``.  Use ``from_parallel("name")``
                     to access an individual branch's typed payload.
    - ``metadata`` : summed input/output tokens and cost across branches.
    - ``error``    : first non-None branch error if any
                     (short-circuit semantics — caller can detect failure).

    Compile-time check: the named step must (a) exist, (b) come earlier in
    the plan, and (c) itself be ``parallel=True`` (otherwise the "band" is
    a single step and the result would be indistinguishable from ``from_step``).
    """

    name: str


@dataclass(frozen=True)
class _FromAgent:
    """Use the last output of the agent registered under ``name``.

    Unlike ``from_step``, which reads from the Plan's execution history,
    ``from_agent`` reads from a shared :class:`Store` where every agent
    writes its last output after running.  This means it works both inside
    a Plan and when agents are called independently by an LLM orchestrator.

    The name must match the key used in ``agent.as_tool("name")`` and the
    agent's ``name=`` attribute.  The store must be shared between the
    writing agent and the orchestrator that resolves the sentinel.

    If the named agent has not yet run (no entry in the store), the sentinel
    contributes nothing — silent no-op, no error.

    Compile-time check (inside Plan): the named tool must exist in the tool
    map and must be an agent (created via ``as_tool()``), not a plain function.

    Example::

        store = Store(db="shared.sqlite")

        researcher = Agent(
            engine=LLMEngine("claude-opus-4-7"),
            tools=[search.as_tool("search")],
            store=store,
            name="research",
        )
        writer = Agent(
            engine=LLMEngine("gpt-4o"),
            store=store,
            name="write",
        )

        # Works in Plan
        pipeline = Agent(
            engine=Plan(
                Step("research"),
                Step("write", context=from_agent("research")),
            ),
            tools=[researcher.as_tool("research"), writer.as_tool("write")],
            store=store,
        )

        # Also works when researcher is called standalone — writer reads
        # from_agent("research") after researcher has run and written to store.
    """

    name: str


@dataclass(frozen=True)
class _FromMemory:
    """Inject the live memory of the agent registered under ``name`` as context.

    Reads the memory at step execution time — never at plan construction —
    so the injected text always reflects the most recent state of that
    agent's conversation history.

    The name must match the key used in ``agent.as_tool("name")``, which
    is the same name used in ``Step(target="name")``.

    If the named agent has no memory or its memory is empty, this sentinel
    contributes nothing to the step's context (silent no-op, no error).

    Compile-time check: the named tool must exist in the tool map and must
    have been created via ``as_tool()`` from an agent that has a ``memory=``
    attached.

    Example::

        researcher = Agent(
            engine=LLMEngine("claude-opus-4-7"),
            tools=[search.as_tool("search")],
            memory=Memory(strategy="summary"),
        )
        writer = Agent(engine=LLMEngine("gpt-4o"))

        pipeline = Agent(
            engine=Plan(
                Step("research"),
                Step("write", context=from_memory("research")),
            ),
            tools=[
                researcher.as_tool("research"),
                writer.as_tool("write"),
            ],
        )
    """

    name: str


# Public singletons / factories
from_prev = _FromPrev()
from_start = _FromStart()


def from_step(name: str) -> _FromStep:
    return _FromStep(name=name)


def from_parallel(name: str) -> _FromParallel:
    return _FromParallel(name=name)


def from_parallel_all(name: str) -> _FromParallelAll:
    """Aggregate every consecutive parallel sibling starting at ``name``."""
    return _FromParallelAll(name=name)


def from_agent(name: str) -> _FromAgent:
    """Read the last output of the agent registered as ``name`` from the shared store."""
    return _FromAgent(name=name)


def from_memory(name: str) -> _FromMemory:
    """Inject the live memory of the agent registered as ``name`` at execution time."""
    return _FromMemory(name=name)


Sentinel = _FromPrev | _FromStart | _FromStep | _FromParallel | _FromParallelAll | _FromAgent | _FromMemory

#: Store key prefix used when an agent writes its last output.
#: Internal — not part of the public API.
_AGENT_OUTPUT_KEY_PREFIX = "__agent_output__:"
