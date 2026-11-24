import asyncio
from dataclasses import replace

import pytest

from lazybridge import Agent, Plan, Step
from lazybridge.envelope import Envelope
from lazybridge.policy import (
    PolicyBudgetExceeded,
    PolicyDenied,
    PolicyWidening,
    TaskGrant,
    TreeBudget,
    child_within_parent,
    current_policy_context,
    policy_scope,
    resolve,
)
from lazybridge.testing import MockAgent
from lazybridge.tools import Tool


def policy(tmp_path, *, classes=("read",), tools=()):
    return resolve(
        "edit-and-run",
        TaskGrant(tmp_path, {"tool_classes": list(classes), "tools": list(tools)}, "off", "local"),
    )


class CaptureEngine:
    def __init__(self):
        self.seen = []

    async def run(self, env, *, tools, **kwargs):
        self.seen.append([tool.name for tool in tools])
        return Envelope(task=env.task, payload="ok")


@pytest.mark.asyncio
async def test_tool_discovery_and_direct_boundary(tmp_path):
    read = Tool(lambda: "read", name="read", policy_grant="read")
    edit = Tool(lambda: "edit", name="edit", policy_grant="edit")
    engine = CaptureEngine()
    agent = Agent(engine=engine, name="root", tools=[read, edit], policy=policy(tmp_path))
    assert (await agent.run("work")).text() == "ok"
    assert engine.seen == [["read"]]

    async def bypass():
        with pytest.raises(PolicyDenied):
            await edit.run()

    gate = Tool(bypass, name="gate", policy_grant="read")
    agent = Agent(engine=CaptureEngine(), name="root", tools=[gate], policy=policy(tmp_path))
    with policy_scope(agent.policy):
        await gate.run()


@pytest.mark.asyncio
async def test_nested_mock_sibling_and_grandchild_cannot_widen(tmp_path):
    parent = policy(tmp_path)
    wider = policy(tmp_path, classes=("read", "edit"))
    child = MockAgent("child", name="child", policy=parent)
    assert (await Agent(engine=CaptureEngine(), name="root", tools=[child], policy=parent).run("x")).ok

    with policy_scope(parent):
        with pytest.raises(PolicyWidening):
            await MockAgent("no", name="sibling", policy=wider).run("x")
        with policy_scope(parent), pytest.raises(PolicyWidening):
            await MockAgent("no", name="grandchild", policy=wider).run("x")


def test_grandchild_can_narrow_to_sub_subfolder(tmp_path):
    parent = policy(tmp_path, classes=("read", "edit"))
    child = child_within_parent(parent, TaskGrant(tmp_path / "sub", {"tool_classes": ["read"]}, "off", "local"))
    request = TaskGrant(child.folder / "nested", {"tool_classes": ["read"]}, "off", "local")

    with policy_scope(parent), policy_scope(child), policy_scope(request) as context:
        grandchild = context.effective_policy
        assert grandchild.folder == child.folder / "nested"
        assert grandchild.scratch_folder == child.scratch_folder / "nested"
        assert grandchild.tool_classes == {"read"}

    escaped = replace(grandchild, scratch_folder=child.scratch_folder / ".." / "sibling")
    with (
        policy_scope(parent),
        policy_scope(child),
        pytest.raises(PolicyWidening, match="scratch"),
        policy_scope(escaped),
    ):
        pass


@pytest.mark.asyncio
async def test_mock_agent_tree_budget_reaches_plan_and_grandchildren(tmp_path):
    authority = policy(tmp_path)
    leaf = MockAgent("leaf", name="leaf")

    async def invoke_leaf(_env):
        return await leaf.run("nested")

    child = MockAgent(invoke_leaf, name="child")
    root = Agent(
        engine=Plan(Step(child)),
        name="root",
        policy=authority,
        tree_budget=TreeBudget(max_depth=1),
    )
    result = await root.run("work")
    assert not result.ok
    assert "max_depth" in result.error.message


@pytest.mark.asyncio
async def test_budget_without_policy_enforces_limits_and_allows_tools():
    child = MockAgent("child", name="child")
    tool = Tool(lambda: "allowed", name="untagged")

    async def invoke(_env):
        assert await tool.run() == "allowed"
        with pytest.raises(PolicyBudgetExceeded, match="max_total_children"):
            await child.run("nested")
        return "done"

    root = MockAgent(invoke, tree_budget=TreeBudget(max_total_children=0))
    assert (await root.run("work")).text() == "done"


@pytest.mark.asyncio
async def test_mock_agent_resume_reuses_total_child_slot(tmp_path):
    authority = policy(tmp_path)
    child = MockAgent("done", name="child")

    async def run_twice(_env):
        assert (await child.run("first")).ok
        assert (await child.run("resume")).ok
        return "complete"

    root = MockAgent(
        run_twice,
        name="root",
        policy=authority,
        tree_budget=TreeBudget(max_total_children=1),
    )
    assert (await root.run("work")).text() == "complete"

    with (
        policy_scope(authority, tree_budget=TreeBudget(max_total_children=0)),
        pytest.raises(PolicyBudgetExceeded, match="max_total_children"),
    ):
        await child.run("new")


@pytest.mark.asyncio
async def test_plan_verifier_fallback_and_async_child_keep_context(tmp_path):
    authority = policy(tmp_path)
    observed = []

    def record(env):
        ctx = current_policy_context()
        observed.append(ctx.effective_policy.digest if ctx else None)
        return "approved"

    worker = MockAgent(record, name="worker")
    plan_agent = Agent(engine=Plan(Step(worker)), name="plan", policy=authority)
    await plan_agent.run("plan")

    producer = MockAgent("answer", name="producer", policy=authority)
    verifier = MockAgent(record, name="verifier")
    from lazybridge._verify import verify_with_retry

    with policy_scope(authority):
        await verify_with_retry(producer, Envelope(task="verify"), verifier)

    async def child_task():
        await asyncio.sleep(0)
        ctx = current_policy_context()
        observed.append(ctx.effective_policy.digest if ctx else None)

    async def launch(env):
        task = asyncio.create_task(child_task())
        await task
        return "done"

    await MockAgent(launch, name="background", policy=authority).run("x")
    assert observed and all(item == authority.digest for item in observed)


@pytest.mark.asyncio
async def test_wrapped_tool_preserves_policy_grant(tmp_path):
    original = Tool(lambda: "edited", name="edit", policy_grant="edit")
    alias = Tool.wrap(original, name="alias")
    assert alias.policy_grant == "edit"
    with policy_scope(policy(tmp_path)):
        assert not alias.granted_by_active_policy()
        with pytest.raises(PolicyDenied):
            await alias.run()


def test_resolved_child_checks_scratch_and_mcp_configuration(tmp_path):
    parent = policy(tmp_path)
    with (
        policy_scope(parent),
        pytest.raises(PolicyWidening, match="scratch"),
        policy_scope(replace(parent, scratch_folder=tmp_path / "other")),
    ):
        pass
    server = {"tools": ("read",), "command": "safe"}
    parent = replace(parent, mcp_servers={"server": server}, mcp_inventory=("server",))
    child = replace(parent, mcp_servers={"server": {**server, "command": "unsafe"}})
    with policy_scope(parent), pytest.raises(PolicyWidening, match="configuration"), policy_scope(child):
        pass


@pytest.mark.asyncio
async def test_stream_restores_parent_policy_between_chunks(tmp_path):
    parent = policy(tmp_path)
    child = policy(tmp_path, classes=())
    seen = []

    class StreamEngine(CaptureEngine):
        async def stream(self, env, **kwargs):
            seen.append(current_policy_context())
            yield "one"
            seen.append(current_policy_context())
            yield "two"

    agent = Agent(engine=StreamEngine(), name="child", policy=child)
    read = Tool(lambda: "ok", name="read", policy_grant="read")
    with policy_scope(parent) as parent_context:
        stream = agent.stream("x")
        assert await anext(stream) == "one"
        assert current_policy_context() is parent_context
        assert await read.run() == "ok"
        assert await anext(stream) == "two"
        assert current_policy_context() is parent_context
        await stream.aclose()
        assert current_policy_context() is parent_context
    assert seen[0] is seen[1]
    assert seen[0].effective_policy == child
    assert current_policy_context() is None


@pytest.mark.asyncio
async def test_stream_holds_active_slot_until_closed(tmp_path):
    class StreamEngine(CaptureEngine):
        async def stream(self, env, **kwargs):
            yield "one"
            yield "two"

    child = Agent(engine=StreamEngine(), name="stream")
    other = MockAgent("done", name="other")
    with policy_scope(policy(tmp_path), tree_budget=TreeBudget(max_active_children=1)):
        stream = child.stream("x")
        assert await anext(stream) == "one"
        with pytest.raises(PolicyBudgetExceeded, match="max_active_children"):
            await other.run("x")
        await stream.aclose()
        assert (await other.run("x")).ok


@pytest.mark.asyncio
async def test_mock_subclass_run_override_keeps_policy_in_plan(tmp_path):
    observed = []

    class CustomMock(MockAgent):
        async def run(self, task):
            observed.append("override")
            return await super().run(task)

    def record(env):
        observed.append(current_policy_context().effective_policy.digest)
        return "done"

    authority = policy(tmp_path)
    agent = Agent(engine=Plan(Step(CustomMock(record, name="worker"))), name="plan", policy=authority)
    assert (await agent.run("x")).ok
    assert observed == ["override", authority.digest]


@pytest.mark.asyncio
async def test_fallback_catches_conclude_and_inherits_policy(tmp_path):
    from lazybridge.envelope import ErrorInfo
    from lazybridge.signals import conclude

    seen = []

    class FailedEngine(CaptureEngine):
        async def run(self, env, **kwargs):
            return Envelope(error=ErrorInfo(type="failed", message="failure"))

    class ConcludingEngine(CaptureEngine):
        async def run(self, env, **kwargs):
            ctx = current_policy_context()
            seen.append(ctx.effective_policy.digest if ctx else None)
            conclude("fallback answer")

    authority = policy(tmp_path)
    fallback = Agent(engine=ConcludingEngine(), name="fallback")
    primary = Agent(engine=FailedEngine(), name="primary", fallback=fallback, policy=authority)
    result = await primary._run_as_tool("work")
    assert result.text() == "fallback answer"
    assert seen == [authority.digest]


@pytest.mark.asyncio
async def test_untagged_tools_fail_closed_only_under_policy(tmp_path):
    tool = Tool(lambda: "ok", name="untagged")
    assert await tool.run() == "ok"
    assert tool.granted_by_active_policy()
    engine = CaptureEngine()
    agent = Agent(engine=engine, name="root", tools=[tool], policy=policy(tmp_path))
    assert (await agent.run("work")).ok
    assert engine.seen == [[]]
    with policy_scope(agent.policy), pytest.raises(PolicyDenied):
        await tool.run()
