"""``Step.context`` must deliver a step's *output*, whatever its type.

Before this fix ``_execute_one`` appended a resolved context envelope's
payload only when it was already a ``str``::

    if ctx_env.context:
        ctx_parts.append(ctx_env.context)
    if ctx_env.payload and isinstance(ctx_env.payload, str):
        ctx_parts.append(ctx_env.payload)

An agent declared with ``output=SomeModel`` therefore contributed *no*
context at all, and the first line quietly substituted the upstream
step's own **input** context in its place.  Both halves were silent: no
error, no warning, and a downstream step that looked wired up while
receiving something else entirely.

Coverage below: the payload types that used to be dropped, the ones that
already worked (non-regression), the payload-less fallback that must
survive, and the two substitution behaviours that must NOT come back.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lazybridge import Plan, Step
from lazybridge.envelope import Envelope
from lazybridge.sentinels import from_start, from_step
from lazybridge.testing import MockAgent


class Note(BaseModel):
    headline: str
    conviction: float


def _capture() -> tuple[list[str], MockAgent]:
    """A terminal agent that records the ``context`` it was handed."""
    seen: list[str] = []

    def record(env: Envelope) -> str:
        seen.append(env.context or "")
        return "done"

    return seen, MockAgent(record, name="sink")


async def _run(plan: Plan, task: str = "initial", context: str | None = None) -> Envelope:
    plan._validate({})
    return await plan.run(
        Envelope(task=task, context=context),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    )


# ---------------------------------------------------------------------------
# The payloads that used to vanish
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pydantic_payload_reaches_downstream_context() -> None:
    """The regression that motivated the fix: ``output=`` model context."""
    upstream = MockAgent(Note(headline="rates repriced", conviction=0.8), name="up")
    seen, sink = _capture()

    result = await _run(
        Plan(
            Step(upstream, name="up"),
            Step(sink, name="sink", task="synthesise", context=from_step("up")),
        )
    )

    assert result.ok
    assert "rates repriced" in seen[0]
    assert "0.8" in seen[0]


@pytest.mark.asyncio
async def test_dict_payload_reaches_downstream_context() -> None:
    """JSON-serialisable non-str payloads were dropped by the same test.

    The payload is handed over inside an ``Envelope`` because a bare dict
    given to ``MockAgent`` is a *response mapping* (task substring → reply),
    not a payload.
    """
    upstream = MockAgent(Envelope(payload={"regime": "risk-off"}), name="up")
    seen, sink = _capture()

    await _run(
        Plan(
            Step(upstream, name="up"),
            Step(sink, name="sink", task="synthesise", context=from_step("up")),
        )
    )

    assert "risk-off" in seen[0]


@pytest.mark.asyncio
async def test_list_context_mixes_typed_and_string_payloads() -> None:
    """A multi-item ``context=`` resolves each item on its own terms."""
    typed = MockAgent(Note(headline="curve steepened", conviction=0.4), name="typed")
    plain = MockAgent("plain-output", name="plain")
    seen, sink = _capture()

    await _run(
        Plan(
            Step(typed, name="typed"),
            Step(plain, name="plain"),
            Step(
                sink,
                name="sink",
                task="synthesise",
                context=[from_step("typed"), from_step("plain")],
            ),
        )
    )

    assert "curve steepened" in seen[0]
    assert "plain-output" in seen[0]
    assert seen[0].index("curve steepened") < seen[0].index("plain-output")


# ---------------------------------------------------------------------------
# Non-regression: what already worked must keep working
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_str_payload_still_reaches_downstream_context() -> None:
    upstream = MockAgent("upstream-output", name="up")
    seen, sink = _capture()

    await _run(
        Plan(
            Step(upstream, name="up"),
            Step(sink, name="sink", task="synthesise", context=from_step("up")),
        )
    )

    assert "upstream-output" in seen[0]


@pytest.mark.asyncio
async def test_literal_string_context_still_appended_verbatim() -> None:
    seen, sink = _capture()

    await _run(Plan(Step(sink, name="sink", task="go", context="fixed instructions")))

    assert seen[0] == "fixed instructions"


@pytest.mark.asyncio
async def test_payloadless_envelope_falls_back_to_its_context() -> None:
    """``from_start`` carries meaning in ``context``, not ``payload`` —
    the ``elif`` branch exists for exactly this case."""
    seen, sink = _capture()

    await _run(
        Plan(Step(sink, name="sink", task="go", context=from_start)),
        context="the original brief",
    )

    assert seen[0] == "the original brief"


# ---------------------------------------------------------------------------
# The two substitutions that must not come back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_input_context_is_not_forwarded_in_place_of_output() -> None:
    """A resolved step contributes its own output — never the context it
    was itself given.  The old code appended both, so a dropped payload
    was silently replaced by the upstream step's input."""
    upstream = MockAgent(Note(headline="the output", conviction=0.1), name="up")
    seen, sink = _capture()

    await _run(
        Plan(
            Step(upstream, name="up", context="the upstream's own briefing"),
            Step(sink, name="sink", task="synthesise", context=from_step("up")),
        )
    )

    assert "the output" in seen[0]
    assert "briefing" not in seen[0]


@pytest.mark.asyncio
async def test_empty_payload_contributes_nothing_and_revives_nothing() -> None:
    """An empty-string payload is a real (if empty) output.  It must not
    fall through to the upstream-context branch — that would reinstate
    the substitution for every step that returns nothing."""
    upstream = MockAgent("", name="up")
    seen, sink = _capture()

    await _run(
        Plan(
            Step(upstream, name="up", context="the upstream's own briefing"),
            Step(sink, name="sink", task="synthesise", context=from_step("up")),
        )
    )

    assert seen[0] == ""
