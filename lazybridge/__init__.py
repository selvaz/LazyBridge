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

    pipeline = sess.as_tool("research_pipeline", "Run the pipeline", entry_agent=researcher)
    orchestrator = LazyAgent("anthropic")
    orchestrator.loop("coordinate the work", tools=[pipeline])
"""

__version__ = "0.3.1"

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
)
from lazybridge.graph.schema import EdgeType, GraphSchema, NodeType
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

__all__ = [
    # Main classes
    "LazyAgent",
    "Memory",
    "LazySession",
    "LazyContext",
    "LazyStore",
    "StoreEntry",
    "LazyTool",
    "LazyRouter",
    "run_async",
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
    "ToolDefinition",
    "UsageStats",
    "BaseProvider",
]
