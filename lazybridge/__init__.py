"""lazybridge — Agent = Engine + Tools + State.

Every Agent has the same shape.  Only the engine changes::

    from lazybridge import Agent, LLMEngine, Plan, Step, Memory, Session, Store

    # --- Build sub-agents first ---

    researcher = Agent(
        engine=LLMEngine("claude-opus-4-7", system="You are a research expert."),
        tools=[search],          # search is a plain Python function
        store=store,             # required if other steps use from_agent("research")
    )
    writer = Agent(
        engine=LLMEngine("gpt-4o", system="You are a concise technical writer."),
    )

    # --- Deterministic orchestrator: Plan engine ---

    pipeline = Agent(
        engine=Plan(
            Step("research"),                                  # calls researcher tool
            Step("write", task=from_prev, context=from_step("research")),
        ),
        tools=[
            researcher.as_tool("research"),   # alias "research" = Step target + sentinel key
            writer.as_tool("write"),
        ],
        memory=Memory(strategy="summary"),
        session=Session(),
    )

    # --- Dynamic orchestrator: LLM engine ---

    orchestrator = Agent(
        engine=LLMEngine("claude-opus-4-7"),
        tools=[
            researcher.as_tool("research"),
            writer.as_tool("write"),
        ],
        memory=Memory(),
        session=Session(),
    )

    result = pipeline("AI trends 2026").text()

**String shortcut** — ``Agent("claude-opus-4-7")`` expands to
``Agent(engine=LLMEngine("claude-opus-4-7"))``.  Use the explicit
``LLMEngine(...)`` form when you need ``system=``, ``max_turns=``,
``thinking=``, or other engine-level config.

**The name chain** — the alias passed to ``as_tool("name")`` is the
authoritative key that connects every part of the system::

    researcher.as_tool("research")  →  key in tool map: "research"
    Step("research")                →  looks up "research" in tool map ✓
    routes={"research": predicate}  →  routes to the step named "research" ✓
    from_step("research")           →  reads output of step "research" (in-Plan) ✓
    from_agent("research")          →  reads last stored output of "research" (cross-run) ✓
    from_memory("research")         →  reads live memory of "research" ✓

**Choosing between sentinels** — inside a single Plan, ``from_step`` is
the standard choice: it reads from in-memory step history with no
external dependency.  ``from_agent`` is for cross-run or cross-plan
data: last known output persisted in a shared Store.
``from_memory`` reads the agent's live conversation history.
"""

__version__ = "0.7.0"
__stability__ = "alpha"

# Public API.  Symbols a user constructs, passes as a kwarg, or catches as
# an exception are re-exported from this top-level module.  Internals
# (provider request/response types, dispatcher helpers like wrap_tool /
# build_tool_map, attribute-only types like EnvelopeMetadata / StoreEntry,
# rarely-subclassed Protocols) live under their submodules and stay
# importable as ``from lazybridge.X import Y`` for advanced callers.

# Primary surface
from lazybridge.agent import Agent
from lazybridge.agent import _ParallelAgent as _ParallelAgent

# Provider entry points
from lazybridge.core.providers import BaseProvider
from lazybridge.core.types import (
    AgentRuntimeConfig,
    CacheConfig,
    NativeTool,
    ObservabilityConfig,
    ResilienceConfig,
)

# Engines (HumanEngine, SupervisorEngine, eval helpers, and OTelExporter
# live under ``lazybridge.ext.{hil,evals,otel}``).
from lazybridge.engines.llm import LLMEngine, StreamStallError, ToolTimeoutError
from lazybridge.engines.plan import Plan, PlanCompileError, Step
from lazybridge.envelope import Envelope

# Exporters (core).  ``OTelExporter`` lives in ``lazybridge.ext.otel``.
from lazybridge.exporters import (
    CallbackExporter,
    ConsoleExporter,
    EventExporter,
    FilteredExporter,
    JsonFileExporter,
    StructuredLogExporter,
)

# Graph
from lazybridge.graph import GraphSchema
from lazybridge.guardrails import ContentGuard, Guard, GuardAction, GuardChain, GuardError, LLMGuard
from lazybridge.memory import Memory
from lazybridge.predicates import when
from lazybridge.sentinels import (
    from_agent,
    from_memory,
    from_parallel,
    from_parallel_all,
    from_prev,
    from_start,
    from_step,
)
from lazybridge.session import EventLog, EventType, Session
from lazybridge.store import Store
from lazybridge.testing import MockAgent
from lazybridge.tools import Tool, ToolProvider

__all__ = [
    # Primary API
    "Agent",
    # Envelope
    "Envelope",
    # Sentinels
    "from_prev",
    "from_start",
    "from_step",
    "from_parallel",
    "from_parallel_all",
    "from_memory",
    "from_agent",
    # Tools
    "Tool",
    "ToolProvider",
    # Native tools (provider-hosted, e.g. web search)
    "NativeTool",
    # Predicates DSL (for Step.routes)
    "when",
    # State
    "Memory",
    "Store",
    # Session / Observability
    "Session",
    "EventLog",
    "EventType",
    # Guardrails
    "Guard",
    "GuardAction",
    "GuardError",
    "ContentGuard",
    "GuardChain",
    "LLMGuard",
    # Engines (HumanEngine, SupervisorEngine in lazybridge.ext.hil)
    "LLMEngine",
    "Plan",
    "Step",
    "PlanCompileError",
    "ToolTimeoutError",
    "StreamStallError",
    # Graph
    "GraphSchema",
    # Exporters
    "EventExporter",
    "CallbackExporter",
    "ConsoleExporter",
    "FilteredExporter",
    "JsonFileExporter",
    "StructuredLogExporter",
    # Provider entry point (custom adapters)
    "BaseProvider",
    # Shareable runtime configs
    "AgentRuntimeConfig",
    "CacheConfig",
    "ObservabilityConfig",
    "ResilienceConfig",
    # Testing
    "MockAgent",
]
