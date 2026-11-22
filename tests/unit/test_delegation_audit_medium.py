from __future__ import annotations

import pytest

from lazybridge import Agent, Envelope, Plan, Step
from lazybridge.core.types import ImageContent
from lazybridge.ext.delegation.background import ExtraParam, make_background_delegate
from lazybridge.ext.delegation.jobs import JobRegistry
from lazybridge.session import EventType, Session
from lazybridge.store import Store


class _CaptureEngine:
    model = "capture"

    def __init__(self) -> None:
        self.seen: list[Envelope] = []

    async def run(self, env: Envelope, **_: object) -> Envelope:
        self.seen.append(env)
        return Envelope(task=env.task, payload="ok")


@pytest.mark.parametrize("target_kind", ["direct", "named", "alias"])
@pytest.mark.asyncio
async def test_plan_agent_targets_receive_the_same_complete_envelope(target_kind: str) -> None:
    engine = _CaptureEngine()
    child = Agent(engine=engine, name="child")
    if target_kind == "direct":
        target, tools = child, []
    elif target_kind == "named":
        target, tools = "child", [child]
    else:
        target, tools = "alias", [child.as_tool("alias")]

    plan = Agent(
        engine=Plan(Step(target, context="step context")),
        name="plan",
        tools=tools,
    )
    image = ImageContent.from_url("https://example.test/image.png")
    await plan.run(Envelope(task="task", context="outer context", images=[image]))

    assert len(engine.seen) == 1
    assert engine.seen[0].context == "outer context\n\nstep context"
    assert engine.seen[0].images == [image]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "missing required"),
        ({"repo": 7}, "must be of JSON type string"),
        ({"repo": "ok", "attempts": True}, "must be of JSON type integer"),
        ({"repo": "ok", "attempts": "3"}, "must be of JSON type integer"),
    ],
)
async def test_delegate_rejects_invalid_extra_params_before_side_effects(
    tmp_path, arguments: dict[str, object], message: str
) -> None:
    store = Store(db=str(tmp_path / "jobs.db"))
    registry = JobRegistry(store)
    policy_calls: list[dict[str, object]] = []
    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=lambda **_: object(),
        registry=registry,
        background_tasks=set(),
        notify=None,
        doc="delegate",
        extra_params={
            "repo": ExtraParam(json_type="string", required=True),
            "attempts": ExtraParam(json_type="integer"),
        },
        guard=lambda _objective, values: policy_calls.append(values),
    )

    result = await tool.func("work", **arguments)

    assert message in result
    assert policy_calls == []
    assert store.items(prefix="delegation:job:") == []


@pytest.mark.asyncio
async def test_delegate_treats_null_optional_extra_param_as_omitted(tmp_path) -> None:
    store = Store(db=str(tmp_path / "jobs.db"))
    policy_calls: list[dict[str, object]] = []
    tool = make_background_delegate(
        tool_name="delegate",
        label="worker",
        engine_factory=lambda **_: object(),
        registry=JobRegistry(store),
        background_tasks=set(),
        notify=None,
        doc="delegate",
        extra_params={"attempts": ExtraParam(json_type="integer")},
        guard=lambda _objective, values: policy_calls.append(values) or "REJECTED: stop after validation",
    )

    result = await tool.func("work", attempts=None)

    assert result == "REJECTED: stop after validation"
    assert policy_calls == [{}]
    assert store.items(prefix="delegation:job:") == []


@pytest.mark.asyncio
async def test_verified_tool_shares_one_changing_evidence_snapshot_per_attempt() -> None:
    source_reads = 0
    writer_contexts: list[str | None] = []
    reviewer_prompts: list[str] = []

    class ChangingSource:
        def text(self) -> str:
            nonlocal source_reads
            source_reads += 1
            return f"evidence-{source_reads}"

    class Writer:
        model = "writer"

        async def run(self, env: Envelope, **_: object) -> Envelope:
            writer_contexts.append(env.context)
            return Envelope(task=env.task, payload="draft")

    class Reviewer:
        async def run(self, prompt: str) -> Envelope:
            reviewer_prompts.append(prompt)
            return Envelope(payload="approved")

    writer = Agent(engine=Writer(), name="writer", sources=[ChangingSource()])
    tool = writer.as_tool("write", verify=Reviewer(), verify_context=True)
    await tool.func("question")

    assert source_reads == 1
    assert writer_contexts == ["evidence-1"]
    assert "evidence-1" in reviewer_prompts[0]


@pytest.mark.asyncio
async def test_shared_aliased_child_uses_each_callers_session() -> None:
    class EmittingEngine(_CaptureEngine):
        async def run(self, env: Envelope, *, session=None, **kwargs: object) -> Envelope:
            if session is not None:
                session.emit(EventType.AGENT_START, {"agent_name": "child"})
            return await super().run(env, **kwargs)

    child_engine = EmittingEngine()
    child = Agent(engine=child_engine, name="child")
    alias = child.as_tool("alias")
    session_a, session_b = Session(), Session()

    class CallingEngine:
        model = "caller"

        async def run(self, _env: Envelope, *, tools, **_: object) -> Envelope:
            return await tools[0].func("delegated")

    orchestrator_a = Agent(engine=CallingEngine(), name="a", tools=[alias], session=session_a)
    orchestrator_b = Agent(engine=CallingEngine(), name="b", tools=[alias], session=session_b)
    # The alias hides the child from construction-time propagation; model
    # the implicit inheritance produced by mounting the Agent directly.
    child.session = session_a
    child._session_inherited = True

    await orchestrator_a.run("one")
    await orchestrator_b.run("two")

    starts_a = session_a.events.query(event_type=EventType.AGENT_START)
    starts_b = session_b.events.query(event_type=EventType.AGENT_START)
    assert any(event["payload"].get("agent_name") == "child" for event in starts_a)
    assert any(event["payload"].get("agent_name") == "child" for event in starts_b)


@pytest.mark.asyncio
async def test_named_verified_agent_step_runs_judge_and_publishes_alias_output() -> None:
    store = Store()
    judge_prompts: list[str] = []

    class Judge:
        async def run(self, prompt: str) -> Envelope:
            judge_prompts.append(prompt)
            return Envelope(payload="approved")

    child = Agent(engine=_CaptureEngine(), name="child", store=store)
    alias = child.as_tool("alias", verify=Judge())
    plan = Agent(engine=Plan(Step("alias")), name="plan", tools=[alias])

    result = await plan.run("task")

    assert result.ok
    assert judge_prompts
    assert store.read("__agent_output__:alias") == "ok"


@pytest.mark.asyncio
async def test_explicit_session_grandchild_ignores_inherited_override() -> None:
    caller_session, explicit_session = Session(), Session()
    grand_engine = _CaptureEngine()
    grand = Agent(engine=grand_engine, name="grand", session=explicit_session)

    class ChildEngine:
        model = "child"

        async def run(self, env: Envelope, **_: object) -> Envelope:
            return await grand.run(env)

    child = Agent(engine=ChildEngine(), name="child")
    parent = Agent(engine=Plan(Step("alias")), name="parent", tools=[child.as_tool("alias")], session=caller_session)
    child.session = caller_session
    child._session_inherited = True

    await parent.run("task")

    assert grand.session is explicit_session
