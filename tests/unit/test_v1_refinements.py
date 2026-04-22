"""Tests for architectural refinements #2–#6.

#2  as_tool observability — session propagation to nested Agents.
#3  Provider registry — runtime alias/rule registration.
#4  Agent factories — from_model / from_engine / from_provider.
#5  Envelope[T] generic — typed payloads.
#6  Plan.to_dict / Plan.from_dict — topology round-trip.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from lazybridge import (
    Agent,
    Envelope,
    LLMEngine,
    Plan,
    Session,
    Step,
    from_start,
    from_step,
)

# ---------------------------------------------------------------------------
# #2  as_tool observability
# ---------------------------------------------------------------------------


def test_as_tool_propagates_session_to_nested_agent():
    """When Agent A (with session) wraps Agent B (without session) as a
    tool, B must inherit A's session so nested events are observable.
    """
    sess = Session()
    inner = Agent("claude-opus-4-7", name="inner")   # no session
    assert inner.session is None

    Agent("claude-opus-4-7", name="outer", session=sess, tools=[inner])

    # Session pushed down to the inner agent.
    assert inner.session is sess
    # Both agents are registered in the graph.
    names = {n.name for n in sess.graph.nodes()}
    assert names == {"inner", "outer"}
    # An edge outer → inner is recorded with the as_tool label.
    edges = sess.graph.edges()
    assert any(e.from_id == "outer" and e.to_id == "inner" and e.label == "as_tool" for e in edges)


def test_as_tool_does_not_override_existing_session():
    """If the nested agent already has a session, the outer agent must
    NOT steal it — the developer opted into separate observability.
    """
    sess_outer = Session()
    sess_inner = Session()
    inner = Agent("claude-opus-4-7", name="inner", session=sess_inner)
    Agent("claude-opus-4-7", name="outer", session=sess_outer, tools=[inner])
    assert inner.session is sess_inner   # unchanged


# ---------------------------------------------------------------------------
# #3  Provider registry
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_provider_rules():
    """Snapshot and restore LLMEngine's provider tables so tests don't leak."""
    snap_aliases = dict(LLMEngine._PROVIDER_ALIASES)
    snap_rules = list(LLMEngine._PROVIDER_RULES)
    yield
    LLMEngine._PROVIDER_ALIASES = snap_aliases
    LLMEngine._PROVIDER_RULES = snap_rules


def test_register_provider_alias_is_honoured(restore_provider_rules):
    LLMEngine.register_provider_alias("mistral", "mistral")
    assert LLMEngine._infer_provider("mistral") == "mistral"


def test_register_provider_rule_is_honoured(restore_provider_rules):
    LLMEngine.register_provider_rule("llama", "meta")
    assert LLMEngine._infer_provider("llama-3-70b") == "meta"


def test_newly_registered_rule_beats_builtins(restore_provider_rules):
    # Built-in: "claude" → anthropic. User override: "claude" → custom.
    LLMEngine.register_provider_rule("claude", "custom-proxy")
    assert LLMEngine._infer_provider("claude-opus-4-7") == "custom-proxy"


def test_startswith_rule_is_honoured(restore_provider_rules):
    LLMEngine.register_provider_rule("grok", "x-ai", kind="startswith")
    assert LLMEngine._infer_provider("grok-2") == "x-ai"
    # Non-prefix occurrence must NOT match a startswith rule.
    assert LLMEngine._infer_provider("claude-opus-4-7") == "anthropic"


# ---------------------------------------------------------------------------
# #4  Agent factories
# ---------------------------------------------------------------------------


def test_agent_from_model_uses_llm_engine():
    ag = Agent.from_model("claude-opus-4-7", name="x")
    assert ag.engine.__class__.__name__ == "LLMEngine"
    assert ag.engine.model == "claude-opus-4-7"


def test_agent_from_engine_accepts_plan():
    plan = Plan(Step(lambda task: "out", name="s1"))
    ag = Agent.from_engine(plan, name="planned")
    assert ag.engine is plan
    assert ag.name == "planned"


def test_agent_from_provider_applies_tier():
    """Tier must reach the engine.model so BaseProvider.resolve_model_alias
    maps it to the concrete model.  Pre-fix, ``model=tier`` was passed to
    ``Agent.__init__`` which silently ignored it when ``engine=`` was
    already supplied — the tier was lost.
    """
    ag = Agent.from_provider("anthropic", tier="top", name="y")
    assert ag.engine.model == "top"          # tier reaches the engine
    assert ag.engine.provider == "anthropic"  # explicit provider preserved

    ag_cheap = Agent.from_provider("openai", tier="cheap", name="z")
    assert ag_cheap.engine.model == "cheap"
    assert ag_cheap.engine.provider == "openai"


# ---------------------------------------------------------------------------
# #5  Envelope[T]
# ---------------------------------------------------------------------------


class _Payload(BaseModel):
    value: int = 0


def test_envelope_untyped_accepts_any_payload():
    e = Envelope(task="t", payload={"a": 1})
    assert e.payload == {"a": 1}
    assert e.text() == '{"a": 1}'


def test_envelope_generic_pydantic_payload():
    et: Envelope[_Payload] = Envelope(payload=_Payload(value=7))
    assert isinstance(et.payload, _Payload)
    assert et.payload.value == 7
    # text() renders Pydantic payloads as JSON
    assert et.text() == '{"value":7}'


def test_envelope_error_channel_round_trip():
    err = Envelope.error_envelope(RuntimeError("boom"))
    assert not err.ok
    assert err.error.type == "RuntimeError"
    assert err.error.message == "boom"
    assert err.text() == ""


# ---------------------------------------------------------------------------
# #6  Plan serialisation round-trip
# ---------------------------------------------------------------------------


def test_plan_round_trip_preserves_topology():
    def fetch(task: str) -> str:
        return "f"
    def rank(task: str) -> str:
        return "r"

    plan = Plan(
        Step(fetch, writes="fetched"),
        Step(rank, writes="ranked", task=from_start),
        max_iterations=42,
    )

    blob = json.dumps(plan.to_dict())
    restored = Plan.from_dict(json.loads(blob), registry={"fetch": fetch, "rank": rank})

    assert restored.max_iterations == 42
    assert [s.name for s in restored.steps] == ["fetch", "rank"]
    assert restored.steps[1].task is from_start
    assert restored.steps[0].writes == "fetched"


def test_plan_round_trip_preserves_from_step_sentinels():
    def a(task: str) -> str:
        return "a"
    def b(task: str) -> str:
        return "b"

    plan = Plan(
        Step(a, name="alpha"),
        Step(b, name="beta", task=from_step("alpha")),
    )
    restored = Plan.from_dict(plan.to_dict(), registry={"a": a, "b": b})

    task_sentinel = restored.steps[1].task
    # from_step is a factory — instance equality not guaranteed but type and name are.
    assert task_sentinel.__class__.__name__ == "_FromStep"
    assert task_sentinel.name == "alpha"


def test_plan_round_trip_executes_correctly_after_reload():
    def producer(task: str) -> str:
        return f"produced:{task}"
    def consumer(task: str) -> str:
        return f"consumed({task})"

    plan = Plan(
        Step(producer, name="producer"),
        Step(consumer, name="consumer"),
    )

    blob = plan.to_dict()
    restored = Plan.from_dict(blob, registry={"producer": producer, "consumer": consumer})

    out = Agent.from_engine(restored)("hello")
    assert out.payload == "consumed(produced:hello)"


def test_plan_from_dict_raises_on_missing_registry_entry():
    def fn(task: str) -> str:
        return "x"

    plan = Plan(Step(fn, name="fn"))
    blob = plan.to_dict()
    with pytest.raises(KeyError, match="fn"):
        Plan.from_dict(blob, registry={})   # no 'fn' → should raise


def test_plan_tool_name_targets_survive_round_trip_without_registry():
    # Targets referenced by tool name (string) don't need a registry entry;
    # they stay strings and are resolved by the caller's tool_map.
    def mytool(x: str) -> str:
        return x

    plan = Plan(Step(target="mytool", name="step1"))
    blob = plan.to_dict()
    restored = Plan.from_dict(blob)   # no registry needed
    assert restored.steps[0].target == "mytool"
