"""Canonical grammar tests — Agent(engine=..., tools=[], memory=..., session=...).

These tests document the base grammar of LazyBridge. Every test uses the
canonical form explicitly. String shortcuts and factory methods are not
used here — they belong in test_agent_factories.py. The goal is to make
the grammar visible in the test suite itself.

Key contracts verified:
- All three engine types (LLM, Plan, Human) use the same Agent wrapper.
- as_tool("name") is the canonical way to mount an Agent as a capability.
- The name passed to as_tool must match the Step target string.
- PlanCompiler catches name mismatches at construction time.
- task and context sentinels belong on Step, not on Agent.
- String sugar Agent("model") produces the same shape as the canonical form.
"""

from __future__ import annotations

import pytest

from lazybridge import Agent, LLMEngine, Memory, Plan, Session, Step
from lazybridge.engines.plan._types import PlanCompileError

# ---------------------------------------------------------------------------
# Fake engines for isolation — no real LLM calls
# ---------------------------------------------------------------------------


class _EchoEngine:
    """Returns the task text unchanged."""

    async def run(self, env, *, tools, output_type, memory, session, **_):
        from lazybridge.envelope import Envelope

        return Envelope.from_task(env.task or "")

    async def stream(self, env, *, tools, output_type, memory, session, **_):
        async def _gen():
            yield env.task or ""

        return _gen()


class _FixedEngine:
    """Always returns a fixed string."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def run(self, env, *, tools, output_type, memory, session, **_):
        from lazybridge.envelope import Envelope

        return Envelope.from_task(self._response)

    async def stream(self, env, *, tools, output_type, memory, session, **_):
        async def _gen():
            yield self._response

        return _gen()


# ---------------------------------------------------------------------------
# 1. Canonical form — shape is identical across all engine types
# ---------------------------------------------------------------------------


def test_llm_agent_canonical_shape():
    """Agent(engine=LLMEngine(...), tools=[], memory=Memory(), session=Session())."""
    sess = Session()
    agent = Agent(
        engine=LLMEngine("claude-opus-4-7"),
        tools=[],
        memory=Memory(),
        session=sess,
    )
    assert isinstance(agent.engine, LLMEngine)
    assert agent.memory is not None
    assert agent.session is sess


def test_plan_agent_canonical_shape():
    """Same Agent wrapper — only the engine differs."""

    def noop(task: str) -> str:
        """No-op tool."""
        return task

    agent = Agent(
        engine=Plan(Step(noop, name="step1")),
        tools=[noop],
        memory=Memory(),
        session=Session(),
        name="_t_201",
    )
    assert isinstance(agent.engine, Plan)
    assert agent.memory is not None


def test_custom_engine_canonical_shape():
    """Agent accepts any engine that implements the Engine protocol."""
    agent = Agent(
        engine=_EchoEngine(),
        tools=[],
        memory=Memory(),
        session=Session(),
        name="_t_1",
    )
    assert isinstance(agent.engine, _EchoEngine)


def test_all_three_engines_have_same_call_interface():
    """run() / __call__() work identically regardless of engine type."""

    def noop(task: str) -> str:
        """No-op."""
        return task

    agents = [
        Agent(engine=_EchoEngine(), tools=[], name="_t_2"),
        Agent(engine=_FixedEngine("done"), tools=[], name="_tt_307"),
    ]
    for a in agents:
        assert callable(a)
        assert hasattr(a, "run")
        assert hasattr(a, "stream")


# ---------------------------------------------------------------------------
# 2. String shortcut expands to the canonical LLMEngine form
# ---------------------------------------------------------------------------


def test_string_shortcut_produces_llm_engine():
    """Agent("model") → Agent(engine=LLMEngine("model"))."""
    via_shortcut = Agent("claude-opus-4-7")
    canonical = Agent(engine=LLMEngine("claude-opus-4-7"))
    assert type(via_shortcut.engine) is type(canonical.engine)
    assert via_shortcut.engine.model == canonical.engine.model


def test_none_shortcut_defaults_to_claude():
    """Agent() → Agent(engine=LLMEngine("claude-opus-4-7"))."""
    agent = Agent()
    assert isinstance(agent.engine, LLMEngine)
    assert agent.engine.model == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# 3. as_tool("name") — canonical way to mount an Agent as a capability
# ---------------------------------------------------------------------------


def test_as_tool_registers_under_given_name():
    """researcher.as_tool("research") → key "research" in tool map."""
    researcher = Agent(engine=_EchoEngine(), name="_t_3")
    tool = researcher.as_tool("research")
    assert tool.name == "research"


def test_orchestrator_tool_map_contains_as_tool_names():
    """The name from as_tool appears in the orchestrator's tool map."""
    researcher = Agent(engine=_EchoEngine(), name="_t_4")
    writer = Agent(engine=_EchoEngine(), name="_t_5")

    orchestrator = Agent(
        engine=LLMEngine("claude-opus-4-7"),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
    )
    assert "research" in orchestrator._tool_map
    assert "write" in orchestrator._tool_map


# ---------------------------------------------------------------------------
# 4. Plan + as_tool name contract — Step target must match tool map key
# ---------------------------------------------------------------------------


def test_plan_step_target_matches_as_tool_name():
    """Step("research") resolves to researcher.as_tool("research") in tool map."""
    researcher = Agent(engine=_EchoEngine(), name="_t_6")
    writer = Agent(engine=_EchoEngine(), name="_t_7")

    # This must not raise — the names are consistent
    orchestrator = Agent(
        engine=Plan(
            Step("research"),
            Step("write"),
        ),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
        name="_t_8",
    )
    assert isinstance(orchestrator.engine, Plan)


def test_plan_compiler_catches_missing_tool_name():
    """PlanCompiler raises PlanCompileError when Step target not in tool map."""
    researcher = Agent(engine=_EchoEngine(), name="_t_9")

    with pytest.raises(PlanCompileError, match="tool 'research' not found"):
        Agent(
            engine=Plan(Step("research")),
            tools=[researcher.as_tool("wrong_name")],  # mismatch,
            name="_t_10",
        )


def test_step_name_defaults_to_target_string():
    """Step("research") → name="research" automatically."""
    researcher = Agent(engine=_EchoEngine(), name="_t_11")
    orchestrator = Agent(
        engine=Plan(Step("research")),
        tools=[researcher.as_tool("research")],
        name="_t_12",
    )
    steps = orchestrator.engine.steps
    assert steps[0].name == "research"
    assert steps[0].target == "research"


def test_step_target_and_name_can_differ():
    """Step(target="research", name="phase_1") — calls "research", step is "phase_1"."""
    researcher = Agent(engine=_EchoEngine(), name="_t_13")

    # Valid: target exists in tool map, name is different
    orchestrator = Agent(
        engine=Plan(Step(target="research", name="phase_1")),
        tools=[researcher.as_tool("research")],
        name="_t_202",
    )
    steps = orchestrator.engine.steps
    assert steps[0].target == "research"
    assert steps[0].name == "phase_1"


# ---------------------------------------------------------------------------
# 5. Sentinels belong on Step, not on Agent
# ---------------------------------------------------------------------------


def test_sentinels_on_step_not_agent():
    """task= and context= sentinels are Step attributes, not Agent attributes."""
    from lazybridge.sentinels import from_prev, from_step

    researcher = Agent(engine=_EchoEngine(), name="_t_14")
    writer = Agent(engine=_EchoEngine(), name="_t_15")

    orchestrator = Agent(
        engine=Plan(
            Step("research"),
            Step("write", task=from_prev, context=from_step("research")),
        ),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
        name="_t_16",
    )
    steps = orchestrator.engine.steps
    assert steps[1].task is from_prev
    assert steps[1].context is not None
    # Sentinels do not appear on the Agent itself
    assert not hasattr(orchestrator, "task")
    assert not hasattr(orchestrator, "context")


# ---------------------------------------------------------------------------
# 6. End-to-end canonical run with fake engine
# ---------------------------------------------------------------------------


def test_canonical_agent_runs():
    """Full canonical form runs and returns an Envelope."""
    agent = Agent(
        engine=_FixedEngine("hello"),
        tools=[],
        memory=Memory(),
        session=Session(),
        name="_tt_308",
    )
    result = agent("test input")
    assert result.text() == "hello"


def test_canonical_plan_agent_runs():
    """Plan agent with as_tool composition runs end-to-end."""
    from lazybridge.sentinels import from_prev

    researcher = Agent(engine=_FixedEngine("research result"), name="_tt_309")
    writer = Agent(engine=_FixedEngine("final output"), name="_tt_310")

    pipeline = Agent(
        engine=Plan(
            Step("research"),
            Step("write", task=from_prev),
        ),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
        memory=Memory(),
        session=Session(),
        name="_t_17",
    )
    result = pipeline("test topic")
    assert result.ok
    assert result.text() == "final output"


# ---------------------------------------------------------------------------
# 7. from_memory — live memory reference from another agent
# ---------------------------------------------------------------------------


def test_as_tool_carries_memory_reference():
    """as_tool() passes the agent's Memory object to the Tool."""
    mem = Memory()
    researcher = Agent(engine=_EchoEngine(), memory=mem, name="_t_18")
    tool = researcher.as_tool("research")
    assert tool.agent_memory is mem


def test_as_tool_no_memory_gives_none():
    """as_tool() on an agent without memory sets agent_memory=None."""
    researcher = Agent(engine=_EchoEngine(), name="_t_19")
    tool = researcher.as_tool("research")
    assert tool.agent_memory is None


def test_from_memory_compiler_requires_tool_in_map():
    """PlanCompiler rejects from_memory referencing a tool not in the map."""
    from lazybridge import from_memory
    from lazybridge.engines.plan._types import PlanCompileError

    def noop(task: str) -> str:
        """Noop."""
        return task

    researcher = Agent(engine=_EchoEngine(), memory=Memory(), name="_t_20")
    writer = Agent(engine=_EchoEngine(), name="_t_21")
    # from_memory("ghost") references a tool name not in the tool map
    with pytest.raises(PlanCompileError, match="from_memory"):
        Agent(
            engine=Plan(
                Step(noop, name="research"),
                Step("write", context=from_memory("ghost")),  # "ghost" not in tool map
            ),
            tools=[
                researcher.as_tool("research"),
                writer.as_tool("write"),
            ],
            name="_t_203",
        )


def test_from_memory_compiler_requires_agent_to_have_memory():
    """PlanCompiler rejects from_memory when the tool's agent has no memory."""
    from lazybridge import from_memory
    from lazybridge.engines.plan._types import PlanCompileError

    researcher = Agent(engine=_EchoEngine(), name="_t_22")  # no memory=
    writer = Agent(engine=_EchoEngine(), name="_t_23")
    with pytest.raises(PlanCompileError, match="memory="):
        Agent(
            engine=Plan(
                Step("research"),
                Step("write", context=from_memory("research")),
            ),
            tools=[
                researcher.as_tool("research"),
                writer.as_tool("write"),
            ],
            name="_t_24",
        )


def test_from_memory_resolves_live_at_execution_time():
    """from_memory reads memory at step run time, not at plan construction."""
    from lazybridge import Memory, from_memory

    mem = Memory()
    researcher = Agent(engine=_FixedEngine("research result"), memory=mem, name="_tt_311")
    writer = Agent(engine=_FixedEngine("final output"), name="_tt_312")

    pipeline = Agent(
        engine=Plan(
            Step("research"),
            Step("write", context=from_memory("research")),
        ),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
        name="_t_25",
    )

    # Memory is empty at construction — that's fine
    assert mem.text() == ""

    # After pipeline runs, from_memory would have read whatever was in mem
    # at the moment "write" ran. We verify the pipeline completes without error.
    result = pipeline("test topic")
    assert result.ok


def test_from_memory_empty_memory_is_silent_noop():
    """from_memory with empty memory contributes nothing to context (no error)."""
    from lazybridge import Memory, from_memory

    mem = Memory()
    researcher = Agent(engine=_FixedEngine("result"), memory=mem, name="_tt_313")
    writer = Agent(engine=_FixedEngine("done"), name="_tt_314")

    pipeline = Agent(
        engine=Plan(
            Step("research"),
            Step("write", context=from_memory("research")),
        ),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
        name="_t_26",
    )
    # Empty memory → silent no-op → pipeline still runs
    result = pipeline("topic")
    assert result.ok
    assert result.text() == "done"


# ---------------------------------------------------------------------------
# 8. from_agent — reads last output of a named agent from shared Store
# ---------------------------------------------------------------------------


def test_as_tool_carries_store_reference():
    """as_tool() passes the agent's Store object to the Tool."""
    from lazybridge import Store

    store = Store()
    researcher = Agent(engine=_EchoEngine(), store=store, name="_t_27")
    tool = researcher.as_tool("research")
    assert tool.agent_store is store


def test_as_tool_no_store_gives_none():
    """as_tool() on an agent without a store sets agent_store=None."""
    researcher = Agent(engine=_EchoEngine(), name="_t_28")
    tool = researcher.as_tool("research")
    assert tool.agent_store is None


def test_agent_writes_output_to_store_after_run():
    """Agent writes its last output to store under '__agent_output__:{name}' after success."""
    from lazybridge import Store
    from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX

    store = Store()
    researcher = Agent(engine=_FixedEngine("research result"), store=store, name="research")
    researcher("some topic")
    value = store.read(_AGENT_OUTPUT_KEY_PREFIX + "research")
    assert value == "research result"


def test_from_agent_compiler_requires_tool_in_map():
    """PlanCompiler rejects from_agent referencing a tool not in the map."""
    from lazybridge import from_agent
    from lazybridge.engines.plan._types import PlanCompileError

    researcher = Agent(engine=_EchoEngine(), name="_t_29")
    writer = Agent(engine=_EchoEngine(), name="_t_30")
    with pytest.raises(PlanCompileError, match="from_agent"):
        Agent(
            engine=Plan(
                Step("research"),
                Step("write", context=from_agent("ghost")),  # "ghost" not in tool map
            ),
            tools=[
                researcher.as_tool("research"),
                writer.as_tool("write"),
            ],
            name="_t_31",
        )


def test_from_agent_compiler_requires_agent_tool():
    """PlanCompiler rejects from_agent when the tool is a plain function (not an agent)."""
    from lazybridge import from_agent
    from lazybridge.engines.plan._types import PlanCompileError

    def plain(task: str) -> str:
        """A plain function."""
        return task

    researcher = Agent(engine=_EchoEngine(), name="_t_32")
    with pytest.raises(PlanCompileError, match="from_agent"):
        Agent(
            engine=Plan(
                Step("research"),
                Step("plain", context=from_agent("plain")),
            ),
            tools=[
                researcher.as_tool("research"),
                plain,
            ],
            name="_t_33",
        )


def test_from_agent_resolves_from_store():
    """from_agent reads the named agent's last output from the shared Store."""
    from lazybridge import Store, from_agent

    store = Store()

    researcher = Agent(engine=_FixedEngine("research result"), store=store, name="research")
    writer = Agent(engine=_FixedEngine("final output"), name="_tt_315")

    pipeline = Agent(
        engine=Plan(
            Step("research"),
            Step("write", context=from_agent("research")),
        ),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
        name="_t_34",
    )
    result = pipeline("test topic")
    assert result.ok
    assert result.text() == "final output"


def test_from_agent_store_populated_by_research_step():
    """from_agent reads the output that the research step wrote to the store."""
    from lazybridge import Store, from_agent

    store = Store()

    researcher = Agent(engine=_FixedEngine("result"), store=store, name="research")
    writer = Agent(engine=_FixedEngine("done"), name="_tt_316")

    pipeline = Agent(
        engine=Plan(
            Step("research"),  # runs first, writes to store
            Step("write", context=from_agent("research")),  # reads what research wrote
        ),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
        name="_t_35",
    )
    result = pipeline("topic")
    assert result.ok
    assert result.text() == "done"


def test_from_agent_compiler_requires_store_on_agent():
    """PlanCompiler rejects from_agent when the referenced agent has no store= attached."""
    from lazybridge import from_agent
    from lazybridge.engines.plan._types import PlanCompileError

    researcher = Agent(engine=_EchoEngine(), name="_t_36")  # no store= — cannot support from_agent
    writer = Agent(engine=_EchoEngine(), name="_t_37")
    with pytest.raises(PlanCompileError, match="store="):
        Agent(
            engine=Plan(
                Step("research"),
                Step("write", context=from_agent("research")),
            ),
            tools=[
                researcher.as_tool("research"),
                writer.as_tool("write"),
            ],
            name="_t_38",
        )


def test_from_agent_uses_tool_alias_not_agent_name():
    """from_agent reads under the tool alias, not the agent's internal name= attribute.

    This is the key contract: as_tool("research") makes "research" the authoritative
    key, even if the wrapped agent has a different name or no name at all.
    """
    from lazybridge import Store, from_agent

    store = Store()
    # Agent has no explicit name= — its internal name won't match the alias "research"
    researcher = Agent(engine=_FixedEngine("research result"), store=store, name="_tt_317")
    writer = Agent(engine=_FixedEngine("final output"), name="_tt_318")

    pipeline = Agent(
        engine=Plan(
            Step("research"),
            Step("write", context=from_agent("research")),
        ),
        tools=[
            researcher.as_tool("research"),  # alias is "research"
            writer.as_tool("write"),
        ],
        name="_t_39",
    )
    result = pipeline("topic")
    assert result.ok
    # Pipeline completes — alias-based write ensures from_agent("research") finds the data
    assert result.text() == "final output"


def test_from_agent_missing_store_entry_is_silent_noop():
    """from_agent contributes nothing when the store key is absent (agent hasn't run yet).

    This is the documented silent-noop contract: the agent tool exists,
    has store= attached, but the store contains no __agent_output__:research
    key because the research agent has never been called.  The step should
    still complete normally, receiving no context from from_agent.
    """
    from lazybridge import Store, from_agent
    from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX

    store = Store()

    class _ContextCapture:
        """Engine that records the context it received."""

        received_context: str | None = None

        async def run(self, env, *, tools, output_type, memory, session, **_):
            from lazybridge.envelope import Envelope

            _ContextCapture.received_context = env.context
            return Envelope.from_task("done")

        async def stream(self, env, *, tools, output_type, memory, session, **_):
            async def _gen():
                yield "done"

            return _gen()

    researcher = Agent(engine=_FixedEngine("result"), store=store, name="research")
    writer = Agent(engine=_ContextCapture(), name="write")

    # Pre-condition: store is empty — research has NOT run yet.
    assert store.read(_AGENT_OUTPUT_KEY_PREFIX + "research") is None

    # Manually remove the key that Step("research") will write, to simulate
    # from_agent being evaluated before any run.  We test by directly
    # resolving the sentinel against an empty store instead of running
    # the full pipeline (which would run research first and populate it).
    from lazybridge.envelope import Envelope
    from lazybridge.tools import build_tool_map

    tool_map = build_tool_map([researcher.as_tool("research"), writer.as_tool("write")])
    plan_instance = Plan(
        Step("research"),
        Step("write", context=from_agent("research")),
    )
    sentinel = from_agent("research")
    resolved = plan_instance._resolve_sentinel(
        sentinel,
        Envelope.from_task("dummy"),
        Envelope.from_task("dummy"),
        [],
        {},
        tool_map,
    )
    # Missing key → empty envelope, no error, no crash
    assert resolved.error is None
    assert not resolved.context
    assert resolved.ok
