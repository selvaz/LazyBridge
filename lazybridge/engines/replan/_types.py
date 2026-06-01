"""ReplanEngine dataclasses: Task and PlanRound.

These are the only types the user needs to interact with ReplanEngine:

- :class:`PlanRound` — structured output schema for the planner agent.
  Pass ``output=PlanRound`` when building the planner.
- :class:`Task` — one tool call within a round.  The planner emits a list
  of these; ReplanEngine dispatches them, parallelising where flagged.

Example::

    from lazybridge import Agent, LLMEngine
    from lazybridge.engines.replan import PlanRound

    planner = Agent(
        engine=LLMEngine("claude-opus-4-8", system="You are a task planner."),
        output=PlanRound,
        name="planner",
    )
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single tool call planned for this round.

    ``tool`` must match a key in the parent Agent's tool_map — an agent, a
    plain function, a pool route, or any other callable wrapped as a Tool.
    ``kwargs`` are forwarded verbatim to ``tool.run(**kwargs)`` so they must
    match the tool's JSON schema exactly.

    Examples::

        Task(tool="analyst",  kwargs={"task": "analyse the auth module"})
        Task(tool="route",    kwargs={"agent_name": "alice", "task": "write tests"})
        Task(tool="add",      kwargs={"a": 3, "b": 7})
    """

    tool: str = Field(..., description="Name of the tool in the tool_map.")
    kwargs: dict[str, Any] = Field(
        ...,
        description="Kwargs forwarded verbatim to tool.run(**kwargs).",
    )
    parallel: bool = Field(
        True,
        description=(
            "True → run concurrently with adjacent parallel=True siblings. "
            "False → run sequentially after the parallel group in this round."
        ),
    )


class PlanRound(BaseModel):
    """Structured output emitted by the planner agent each round.

    ReplanEngine calls the planner tool, deserialises this schema, and
    dispatches the tasks.  Tasks within the same round with ``parallel=True``
    run concurrently via ``asyncio.gather``; ``parallel=False`` tasks run
    sequentially after the parallel group.

    Dependent tasks belong in the *next* round — after the planner has seen
    the outputs from this one.
    """

    reasoning: str = Field(
        ...,
        description="Why this set of tasks was chosen for this round.",
    )
    tasks: list[Task] = Field(
        default_factory=list,
        description="Tasks to execute this round.",
    )
    done: bool = Field(
        False,
        description="Set True when no further rounds are needed.",
    )
    final_answer: str | None = Field(
        None,
        description="The user-facing answer. Required when done=True.",
    )
