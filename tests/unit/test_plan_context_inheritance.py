"""Regression coverage for incoming context inherited by Plan steps."""

from __future__ import annotations

import pytest

from lazybridge import Agent, Plan, Step
from lazybridge.envelope import Envelope
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
