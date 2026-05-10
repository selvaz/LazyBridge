"""Agent constructor / factory surface (post-0.8.0 simplification).

Every Agent is ``Container(engine, tools, state)``.  The engine decides
HOW the agent behaves; everything else (memory, session, guard, verify,
fallback, output, name) is uniform across every engine.  0.8.0 deleted
the five pure-alias factories (``from_model`` / ``from_engine`` /
``from_plan`` / ``from_chain`` / ``from_parallel``) — anything that
just renamed the canonical form is gone.

What stays:

- ``Agent("model")`` — pure-alias string positional shortcut.
- ``Agent(engine=...)`` — canonical ctor for every other engine kind.
- ``Agent.from_provider(provider, tier=...)`` — non-trivial: resolves a
  tier alias to the provider's current model.
- ``Agent.chain(*agents)`` — non-trivial: builds the Plan + Step graph.
- ``Agent.parallel(*agents)`` — non-trivial: returns ``_ParallelAgent``
  whose ``__call__`` yields ``list[Envelope]``.
"""

from __future__ import annotations

import pytest

from lazybridge import Agent, Memory, Session, Step
from lazybridge.engines.llm import LLMEngine
from lazybridge.engines.plan import Plan
from lazybridge.envelope import Envelope


def _stub(name: str = "stub") -> Agent:
    """Build an Agent without firing a real provider — bypasses __init__'s
    LLMEngine construction by using ``__new__`` and stamping the minimum
    surface ``Plan`` / chain / parallel need.  Tests that need a fully
    functional Agent use ``MockAgent`` instead.
    """
    a = Agent.__new__(Agent)
    a._is_lazy_agent = True
    a.name = name
    a.description = None
    a.session = None
    a.engine = LLMEngine("claude-opus-4-7")
    return a


# ---------------------------------------------------------------------------
# Discoverability — every documented factory is on the class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["from_provider", "chain", "parallel"],
)
def test_factory_is_classmethod_on_agent(method):
    bound = getattr(Agent, method, None)
    assert callable(bound), f"Agent.{method} should exist and be callable"


@pytest.mark.parametrize(
    "removed",
    ["from_model", "from_engine", "from_plan", "from_chain", "from_parallel"],
)
def test_deleted_factories_are_actually_gone(removed):
    """0.8.0 removed five pure-alias factories.  The deletion is part of the
    LLM-friendliness contract — sugar that just renames the canonical form
    is more concept for an LLM to learn, not less."""
    assert not hasattr(Agent, removed), (
        f"Agent.{removed} was deleted in 0.8.0; use the canonical form (see docs/concepts/canonical-vs-sugar.md)."
    )


# ---------------------------------------------------------------------------
# String-positional shortcut — explicit string-to-LLMEngine (alias for canonical)
# ---------------------------------------------------------------------------


def test_string_shortcut_returns_agent_with_llm_engine():
    a = Agent("claude-opus-4-7")
    assert isinstance(a, Agent)
    assert isinstance(a.engine, LLMEngine)
    assert a.engine.model == "claude-opus-4-7"


def test_string_shortcut_passes_kwargs_through():
    sess = Session()
    mem = Memory()
    a = Agent("claude-opus-4-7", memory=mem, session=sess, name="custom")
    assert a.memory is mem
    assert a.session is sess
    assert a.name == "custom"


# ---------------------------------------------------------------------------
# from_provider — tier-aliased LLM
# ---------------------------------------------------------------------------


def test_from_provider_returns_agent_with_llm_engine():
    a = Agent.from_provider("anthropic", tier="top")
    assert isinstance(a, Agent)
    assert isinstance(a.engine, LLMEngine)


# ---------------------------------------------------------------------------
# Canonical engine= keyword — escape hatch for ANY engine
# ---------------------------------------------------------------------------


def test_engine_kwarg_wraps_any_engine_unchanged():
    plan = Plan(Step(target=lambda t: f"out:{t}", name="step1"))
    a = Agent(engine=plan)
    assert isinstance(a, Agent)
    assert a.engine is plan


def test_engine_kwarg_passes_kwargs():
    sess = Session()
    plan = Plan(Step(target=lambda t: f"out:{t}", name="s"))
    a = Agent(engine=plan, session=sess, name="pipeline")
    assert a.session is sess
    assert a.name == "pipeline"


# ---------------------------------------------------------------------------
# Plan engine — declarative DAG (canonical form: Agent(engine=Plan(...)))
# ---------------------------------------------------------------------------


def test_plan_engine_builds_from_steps():
    s1 = Step(target=lambda t: "a", name="s1")
    s2 = Step(target=lambda t: "b", name="s2")
    a = Agent(engine=Plan(s1, s2))
    assert isinstance(a, Agent)
    assert isinstance(a.engine, Plan)
    assert [s.name for s in a.engine.steps] == ["s1", "s2"]


def test_plan_engine_threads_plan_kwargs():
    """Plan-specific kwargs (max_iterations, on_concurrent) reach the engine."""
    a = Agent(
        engine=Plan(
            Step(target=lambda t: "x", name="s"),
            max_iterations=42,
            on_concurrent="fork",
        )
    )
    assert a.engine.max_iterations == 42
    assert a.engine.on_concurrent == "fork"


def test_plan_engine_threads_agent_kwargs():
    """Non-Plan kwargs (memory, session, name) flow to the unified Agent ctor."""
    sess = Session()
    a = Agent(
        engine=Plan(Step(target=lambda t: "x", name="s")),
        session=sess,
        name="pipeline",
    )
    assert a.session is sess
    assert a.name == "pipeline"


def test_plan_engine_executes_end_to_end():
    """A sentinel-free linear plan runs and produces the last step's output."""

    def step_a(task: str) -> str:
        return "A-out"

    def step_b(task: str) -> str:
        return f"B-saw:{task}"

    a = Agent(
        engine=Plan(
            Step(target=step_a, name="a"),
            Step(target=step_b, name="b"),
        )
    )
    env = a("hello")
    assert isinstance(env, Envelope)
    assert "B-saw" in env.text()


# ---------------------------------------------------------------------------
# Agent.chain — sequential agents (real behaviour: builds Plan + Step graph)
# ---------------------------------------------------------------------------


def test_chain_returns_agent_with_plan_engine():
    a, b = _stub("a"), _stub("b")
    chained = Agent.chain(a, b)
    assert isinstance(chained, Agent)
    assert isinstance(chained.engine, Plan)
    assert [s.name for s in chained.engine.steps] == ["a", "b"]


# ---------------------------------------------------------------------------
# Agent.parallel — scripted fan-out (intentional list[Envelope] return)
# ---------------------------------------------------------------------------


def test_parallel_returns_parallel_agent():
    """Documented asymmetry: parallel returns _ParallelAgent (whose
    __call__ yields list[Envelope]) — scripted fan-out has no single
    "result" to wrap into one envelope."""
    from lazybridge.agent import _ParallelAgent

    a, b, c = _stub("a"), _stub("b"), _stub("c")
    p = Agent.parallel(a, b, c)
    assert isinstance(p, _ParallelAgent)
    assert len(p.agents) == 3


def test_parallel_threads_concurrency_kwargs():
    a, b = _stub("a"), _stub("b")
    p = Agent.parallel(a, b, concurrency_limit=2, step_timeout=5.0)
    assert p.concurrency_limit == 2
    assert p.step_timeout == 5.0


# ---------------------------------------------------------------------------
# Cross-factory invariant — every factory honours the same uniform kwargs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        # Canonical Agent(model_string) shortcut
        lambda **kw: Agent("claude-opus-4-7", **kw),
        # Tier-aliased provider factory (the only non-pure-alias factory left)
        lambda **kw: Agent.from_provider("anthropic", tier="top", **kw),
        # Canonical Plan engine
        lambda **kw: Agent(engine=Plan(Step(target=lambda t: "x", name="s")), **kw),
    ],
)
def test_uniform_kwargs_pass_through(builder):
    """name= / session= / memory= reach the constructed Agent regardless of
    which entry path built it.  This is the unified-surface invariant: the
    same uniform kwargs apply to every engine kind."""
    sess = Session()
    mem = Memory()
    a = builder(name="uniform", session=sess, memory=mem)
    assert a.name == "uniform"
    assert a.session is sess
    assert a.memory is mem
