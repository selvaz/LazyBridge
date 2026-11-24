import asyncio

import pytest

from lazybridge import Agent, Plan, Step
from lazybridge.envelope import Envelope
from lazybridge.policy import PolicyDenied, PolicyWidening, TaskGrant, current_policy_context, policy_scope, resolve
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

    gate = Tool(bypass, name="gate")
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
    from dataclasses import replace

    parent = policy(tmp_path)
    with policy_scope(parent):
        with pytest.raises(PolicyWidening, match="scratch"):
            with policy_scope(replace(parent, scratch_folder=tmp_path / "other")):
                pass
    server = {"tools": ("read",), "command": "safe"}
    parent = replace(parent, mcp_servers={"server": server}, mcp_inventory=("server",))
    child = replace(parent, mcp_servers={"server": {**server, "command": "unsafe"}})
    with policy_scope(parent), pytest.raises(PolicyWidening, match="configuration"):
        with policy_scope(child):
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
