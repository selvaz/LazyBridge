"""Wave: unified ``Agent.from_<engine_kind>`` factory surface.

Every Agent is ``Container(engine, tools, state)``.  The engine
decides HOW the agent behaves; everything else (memory, session,
guard, verify, fallback, output, name) is uniform across every
engine.  These factories are sugar that build the right engine and
forward shared kwargs through to the unified Agent constructor — so
reading any ``Agent.from_X(...)`` call site tells you immediately
which engine is in there.

Core factories tested here (ext-engine factories live module-side
in their respective ext packages — see
:mod:`lazybridge.ext.hil` / :mod:`lazybridge.ext.planners` — to
respect the core/ext import boundary).
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
    ["from_model", "from_provider", "from_engine", "from_plan", "from_chain", "from_parallel"],
)
def test_factory_is_classmethod_on_agent(method):
    bound = getattr(Agent, method, None)
    assert callable(bound), f"Agent.{method} should exist and be callable"


# ---------------------------------------------------------------------------
# from_model — explicit string-to-LLMEngine
# ---------------------------------------------------------------------------


def test_from_model_returns_agent_with_llm_engine():
    a = Agent.from_model("claude-opus-4-7")
    assert isinstance(a, Agent)
    assert isinstance(a.engine, LLMEngine)
    assert a.engine.model == "claude-opus-4-7"


def test_from_model_passes_kwargs_through():
    sess = Session()
    mem = Memory()
    a = Agent.from_model("claude-opus-4-7", memory=mem, session=sess, name="custom")
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
# from_engine — escape hatch for ANY engine
# ---------------------------------------------------------------------------


def test_from_engine_wraps_any_engine_unchanged():
    plan = Plan(Step(target=lambda t: f"out:{t}", name="step1"))
    a = Agent.from_engine(plan)
    assert isinstance(a, Agent)
    assert a.engine is plan


def test_from_engine_passes_kwargs():
    sess = Session()
    plan = Plan(Step(target=lambda t: f"out:{t}", name="s"))
    a = Agent.from_engine(plan, session=sess, name="pipeline")
    assert a.session is sess
    assert a.name == "pipeline"


# ---------------------------------------------------------------------------
# from_plan — declarative DAG
# ---------------------------------------------------------------------------


def test_from_plan_builds_plan_engine_from_steps():
    s1 = Step(target=lambda t: "a", name="s1")
    s2 = Step(target=lambda t: "b", name="s2")
    a = Agent.from_plan(s1, s2)
    assert isinstance(a, Agent)
    assert isinstance(a.engine, Plan)
    assert [s.name for s in a.engine.steps] == ["s1", "s2"]


def test_from_plan_threads_plan_kwargs():
    """Plan-specific kwargs (max_iterations, on_concurrent) reach the engine."""
    a = Agent.from_plan(
        Step(target=lambda t: "x", name="s"),
        max_iterations=42,
        on_concurrent="fork",
    )
    assert a.engine.max_iterations == 42
    assert a.engine.on_concurrent == "fork"


def test_from_plan_threads_agent_kwargs():
    """Non-Plan kwargs (memory, session, name) flow to the unified Agent ctor."""
    sess = Session()
    a = Agent.from_plan(
        Step(target=lambda t: "x", name="s"),
        session=sess,
        name="pipeline",
    )
    assert a.session is sess
    assert a.name == "pipeline"


def test_from_plan_executes_end_to_end():
    """A sentinel-free linear plan runs and produces the last step's output."""

    def step_a(task: str) -> str:
        return "A-out"

    def step_b(task: str) -> str:
        return f"B-saw:{task}"

    a = Agent.from_plan(
        Step(target=step_a, name="a"),
        Step(target=step_b, name="b"),
    )
    env = a("hello")
    assert isinstance(env, Envelope)
    assert "B-saw" in env.text()


# ---------------------------------------------------------------------------
# from_chain — sequential agents (linear Plan sugar)
# ---------------------------------------------------------------------------


def test_from_chain_returns_agent_with_plan_engine():
    a, b = _stub("a"), _stub("b")
    chained = Agent.from_chain(a, b)
    assert isinstance(chained, Agent)
    assert isinstance(chained.engine, Plan)
    assert [s.name for s in chained.engine.steps] == ["a", "b"]


def test_from_chain_is_alias_for_chain():
    """Backward-compat: same shape, identical engine type."""
    a, b = _stub("a"), _stub("b")
    by_chain = Agent.chain(a, b)
    by_from = Agent.from_chain(a, b)
    assert isinstance(by_chain.engine, type(by_from.engine))
    assert [s.name for s in by_chain.engine.steps] == [s.name for s in by_from.engine.steps]


# ---------------------------------------------------------------------------
# from_parallel — scripted fan-out (intentional list[Envelope] return)
# ---------------------------------------------------------------------------


def test_from_parallel_returns_parallel_agent():
    """Documented asymmetry: from_parallel returns _ParallelAgent (whose
    __call__ yields list[Envelope]) — scripted fan-out has no single
    "result" to wrap into one envelope."""
    from lazybridge.agent import _ParallelAgent

    a, b, c = _stub("a"), _stub("b"), _stub("c")
    p = Agent.from_parallel(a, b, c)
    assert isinstance(p, _ParallelAgent)
    assert len(p.agents) == 3


def test_from_parallel_threads_concurrency_kwargs():
    a, b = _stub("a"), _stub("b")
    p = Agent.from_parallel(a, b, concurrency_limit=2, step_timeout=5.0)
    assert p.concurrency_limit == 2
    assert p.step_timeout == 5.0


def test_from_parallel_is_alias_for_parallel():
    a, b = _stub("a"), _stub("b")
    by_parallel = Agent.parallel(a, b, concurrency_limit=4)
    by_from = Agent.from_parallel(a, b, concurrency_limit=4)
    assert type(by_parallel) is type(by_from)
    assert by_parallel.concurrency_limit == by_from.concurrency_limit


# ---------------------------------------------------------------------------
# Cross-factory invariant — every factory honours the same uniform kwargs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory_kwargs",
    [
        ("from_model", ("claude-opus-4-7",), {}),
        ("from_provider", ("anthropic",), {"tier": "top"}),
        ("from_plan", (Step(target=lambda t: "x", name="s"),), {}),
    ],
)
def test_uniform_kwargs_pass_through(factory_kwargs):
    """name= / session= / memory= reach the constructed Agent regardless of
    which factory built it.  This is the unified-surface invariant: the
    same uniform kwargs apply to every engine kind."""
    method, args, extra = factory_kwargs
    sess = Session()
    mem = Memory()
    a = getattr(Agent, method)(*args, **extra, name="uniform", session=sess, memory=mem)
    assert a.name == "uniform"
    assert a.session is sess
    assert a.memory is mem
