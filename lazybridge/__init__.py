"""lazybridge — zero-boilerplate multi-provider LLM agent framework.

Quick start::

    from lazybridge import LazyAgent

    ai = LazyAgent("anthropic")
    print(ai.chat("hello").content)

With tools::

    from lazybridge import LazyAgent, LazyTool

    def search(query: str) -> str:
        \"\"\"Search the web.\"\"\"
        ...

    tool = LazyTool.from_function(search)
    ai = LazyAgent("anthropic")
    resp = ai.loop("find the latest AI news", tools=[tool])

Multi-agent session::

    from lazybridge import LazyAgent, LazySession, LazyContext

    sess = LazySession(tracking="basic")

    researcher = LazyAgent("anthropic", name="researcher", session=sess)
    writer     = LazyAgent("openai",    name="writer",     session=sess)

    researcher.loop("find top 3 AI papers this week", tools=[search])
    writer.chat(
        "write a blog post",
        context=LazyContext.from_agent(researcher),
    )

    print(sess.store.read_all())
    print(sess.graph.to_json())

Pipeline as tool (expose a session to an orchestrator)::

    from lazybridge import LazyAgent, LazySession

    sess = LazySession()
    researcher = LazyAgent("anthropic", name="researcher", session=sess)
    writer     = LazyAgent("openai",    name="writer",     session=sess)

    pipeline = sess.as_tool("research_pipeline", "Run the pipeline", mode="chain")
    orchestrator = LazyAgent("anthropic")
    orchestrator.loop("coordinate the work", tools=[pipeline])
"""

__version__ = "0.6.0"

from lazybridge.core.providers import BaseProvider
from lazybridge.core.structured import StructuredOutputError
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    NativeTool,
    Role,
    StreamChunk,
    StructuredOutputConfig,
    ThinkingConfig,
    ToolCall,
    ToolDefinition,
    UsageStats,
    Verifier,
)
from lazybridge.evals import EvalCase, EvalReport, EvalSuite
from lazybridge.exporters import (
    CallbackExporter,
    EventExporter,
    FilteredExporter,
    JsonFileExporter,
    OTelExporter,
    StructuredLogExporter,
)
from lazybridge.graph.schema import EdgeType, GraphSchema, NodeType
from lazybridge.guardrails import ContentGuard, Guard, GuardAction, GuardChain, GuardError, LLMGuard
from lazybridge.human import HumanAgent
from lazybridge.lazy_agent import LazyAgent
from lazybridge.lazy_context import LazyContext
from lazybridge.lazy_router import LazyRouter
from lazybridge.lazy_run import run_async
from lazybridge.lazy_session import Event, LazySession, TrackLevel
from lazybridge.lazy_store import LazyStore, StoreEntry
from lazybridge.lazy_tool import (
    LazyTool,
    NormalizedToolSet,
    ToolArgumentValidationError,
    ToolSchemaBuilder,
    ToolSchemaBuildError,
    ToolSchemaMode,
)
from lazybridge.memory import Memory
from lazybridge.supervisor import SupervisorAgent

__all__ = [
    # Main classes
    "LazyAgent",
    "HumanAgent",
    "SupervisorAgent",
    "Memory",
    "LazySession",
    "LazyContext",
    "LazyStore",
    "StoreEntry",
    "LazyTool",
    "LazyRouter",
    "run_async",
    # Guardrails
    "Guard",
    "GuardAction",
    "GuardError",
    "ContentGuard",
    "GuardChain",
    # Evals
    "EvalSuite",
    "EvalCase",
    "EvalReport",
    "LLMGuard",
    # Tool schema
    "ToolSchemaMode",
    "ToolSchemaBuilder",
    "ToolArgumentValidationError",
    "ToolSchemaBuildError",
    "StructuredOutputError",
    "NormalizedToolSet",
    # Session / tracking
    "TrackLevel",
    "Event",
    # Exporters
    "EventExporter",
    "CallbackExporter",
    "FilteredExporter",
    "JsonFileExporter",
    "StructuredLogExporter",
    "OTelExporter",
    # Graph
    "GraphSchema",
    "NodeType",
    "EdgeType",
    # Core types (re-exported for convenience)
    "CompletionRequest",
    "CompletionResponse",
    "Message",
    "NativeTool",
    "Role",
    "StreamChunk",
    "StructuredOutputConfig",
    "ThinkingConfig",
    "ToolCall",
    "Verifier",
    "ToolDefinition",
    "UsageStats",
    "BaseProvider",
]
