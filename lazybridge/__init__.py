"""lazybridge — zero-boilerplate multi-provider LLM agent framework.

Tier 1 — 2 lines::

    from lazybridge import Agent
    Agent("claude-opus-4-7")("hello").text()

Tier 2 — with tools::

    Agent("claude-opus-4-7", tools=[search, calculator])("find AI news").text()

Tier 3 — structured output::

    class Summary(BaseModel):
        title: str
        bullets: list[str]

    Agent("claude-opus-4-7", output=Summary)("summarize...").payload.title

Tier 4 — multi-agent chain / parallel::

    researcher = Agent("claude-opus-4-7", tools=[search])
    writer     = Agent("claude-opus-4-7")
    Agent.chain(researcher, writer)("AI trends").text()

    Agent.parallel(fact_checker, sentiment_analyzer, summarizer)("article text")

Tier 5 — structured plan with routing::

    from lazybridge import Agent, Plan, Step
    Agent(engine=Plan(Step("search", output=SearchResult), Step("rank")))

Tier 6 — full config::

    from lazybridge import Agent, LLMEngine, Memory, Session
    from lazybridge.ext.otel import OTelExporter
    Agent(
        engine=LLMEngine("claude-opus-4-7", thinking=True, max_turns=20),
        tools=[search],
        output=Report,
        memory=Memory(strategy="auto"),
        session=Session(exporters=[OTelExporter(endpoint="http://jaeger:4318")]),
    )
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
from lazybridge.sentinels import from_parallel, from_parallel_all, from_prev, from_start, from_step
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
