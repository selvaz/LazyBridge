"""Regression coverage for incoming context inherited by Plan steps."""

from __future__ import annotations

import pytest

from lazybridge import Agent, Plan, Step
from lazybridge.envelope import Envelope
from lazybridge.sentinels import from_parallel
from lazybridge.testing import MockAgent


async def _run(plan: Plan, *, context: str | None = None) -> Envelope:
    plan._validate({})
    return await plan.run(
        Envelope(task="incoming task", context=context),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    )


@pytest.mark.asyncio
async def test_nested_plan_first_step_inherits_incoming_context() -> None:
    """An outer step's context survives entry into its Plan target."""
    seen: list[Envelope] = []

    def capture(env: Envelope) -> str:
        seen.append(env)
        return "inner result"

    nested = Agent(
        engine=Plan(Step(MockAgent(capture, name="worker"), name="worker")),
        name="nested",
    )
    result = await _run(
        Plan(Step(nested, name="nested", context="outer context")),
    )

    assert result.ok
    assert seen[0].context == "outer context"


@pytest.mark.asyncio
async def test_explicit_context_follows_inherited_context() -> None:
    """Explicit items extend, rather than replace, incoming context."""
    seen: list[str | None] = []

    def capture(env: Envelope) -> str:
        seen.append(env.context)
        return "done"

    await _run(
        Plan(
            Step(
                MockAgent(capture, name="worker"),
                name="worker",
                context=["explicit one", "explicit two"],
            )
        ),
        context="inherited",
    )

    assert seen == ["inherited\n\nexplicit one\n\nexplicit two"]


@pytest.mark.asyncio
async def test_top_level_plan_without_context_remains_none() -> None:
    """The ordinary top-level Plan path does not invent context."""
    seen: list[str | None] = []

    def capture(env: Envelope) -> str:
        seen.append(env.context)
        return "done"

    await _run(Plan(Step(MockAgent(capture, name="worker"), name="worker")))

    assert seen == [None]


@pytest.mark.asyncio
async def test_parallel_band_steps_inherit_incoming_context() -> None:
    """Every branch in a parallel band receives the Plan's context."""
    seen: dict[str, str | None] = {}

    def capture(name: str):
        def run(env: Envelope) -> str:
            seen[name] = env.context
            return f"{name} result"

        return run

    await _run(
        Plan(
            Step(MockAgent(capture("left"), name="left"), name="left", parallel=True),
            Step(MockAgent(capture("right"), name="right"), name="right", parallel=True),
        ),
        context="shared brief",
    )

    assert seen == {"left": "shared brief", "right": "shared brief"}


@pytest.mark.asyncio
async def test_from_parallel_still_contributes_only_referenced_output() -> None:
    """Inheritance does not make from_parallel repeat a branch's input context."""
    seen: list[str | None] = []

    def capture_join(env: Envelope) -> str:
        seen.append(env.context)
        return "joined"

    await _run(
        Plan(
            Step(MockAgent("left result", name="left"), name="left", parallel=True),
            Step(MockAgent("right result", name="right"), name="right", parallel=True),
            Step(
                MockAgent(capture_join, name="join"),
                name="join",
                context=from_parallel("left"),
            ),
        ),
        context="shared brief",
    )

    assert seen == ["shared brief\n\nleft result"]


@pytest.mark.asyncio
async def test_agent_run_context_reaches_wrapped_plan_steps() -> None:
    """Agent.run's context keyword enters its Plan as envelope context."""
    seen: list[str | None] = []

    def capture(env: Envelope) -> str:
        seen.append(env.context)
        return "done"

    agent = Agent(
        engine=Plan(Step(MockAgent(capture, name="worker"), name="worker")),
        name="pipeline",
    )

    result = await agent.run("incoming task", context="caller context")

    assert result.ok
    assert seen == ["caller context"]
