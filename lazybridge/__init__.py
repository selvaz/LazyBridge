"""lazybridge v1.0 — zero-boilerplate multi-provider LLM agent framework.

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

# Core public API
from lazybridge.agent import Agent
from lazybridge.agent import _ParallelAgent as _ParallelAgent

# Core types (re-exported for convenience)
from lazybridge.core.providers import BaseProvider
from lazybridge.core.types import (
    AgentRuntimeConfig,
    CacheConfig,
    CompletionRequest,
    CompletionResponse,
    Message,
    NativeTool,
    ObservabilityConfig,
    ResilienceConfig,
    Role,
    StreamChunk,
    StructuredOutputConfig,
    ThinkingConfig,
    ToolCall,
    ToolDefinition,
    UsageStats,
)

# Engines.  HumanEngine, SupervisorEngine, eval helpers, and OTelExporter
# are extension surface — import them from ``lazybridge.ext.{hil,evals,otel}``.
from lazybridge.engines.base import Engine
from lazybridge.engines.llm import LLMEngine, StreamStallError, ToolTimeoutError
from lazybridge.engines.plan import Plan, PlanCompileError, PlanState, Step, StepResult
from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo

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
from lazybridge.graph import EdgeType, GraphSchema, NodeType
from lazybridge.guardrails import ContentGuard, Guard, GuardAction, GuardChain, GuardError, LLMGuard
from lazybridge.memory import Memory
from lazybridge.predicates import when
from lazybridge.sentinels import from_parallel, from_parallel_all, from_prev, from_start, from_step
from lazybridge.session import EventLog, EventType, Session
from lazybridge.store import Store, StoreEntry
from lazybridge.tools import Tool

__all__ = [
    # Primary API
    "Agent",
    # Envelope
    "Envelope",
    "EnvelopeMetadata",
    "ErrorInfo",
    # Sentinels
    "from_prev",
    "from_start",
    "from_step",
    "from_parallel",
    "from_parallel_all",
    # Tools
    "Tool",
    # Predicates DSL (for Step.routes)
    "when",
    # Memory & Store
    "Memory",
    "Store",
    "StoreEntry",
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
    # Engines
    "Engine",
    "LLMEngine",
    "Plan",
    "Step",
    "PlanState",
    "StepResult",
    "PlanCompileError",
    "ToolTimeoutError",
    "StreamStallError",
    # Graph
    "GraphSchema",
    "NodeType",
    "EdgeType",
    # Exporters
    "EventExporter",
    "CallbackExporter",
    "ConsoleExporter",
    "FilteredExporter",
    "JsonFileExporter",
    "StructuredLogExporter",
    # Core types
    "BaseProvider",
    "CompletionRequest",
    "CompletionResponse",
    "Message",
    "NativeTool",
    "Role",
    "StreamChunk",
    "StructuredOutputConfig",
    "ThinkingConfig",
    "ToolCall",
    "ToolDefinition",
    "UsageStats",
    # Config objects
    "AgentRuntimeConfig",
    "CacheConfig",
    "ObservabilityConfig",
    "ResilienceConfig",
]
