"""lazybridgeframework — zero-boilerplate multi-provider LLM agent framework.

Quick start::

    from lazybridgeframework import LazyAgent

    ai = LazyAgent("anthropic")
    print(ai.chat("hello").content)

With tools::

    from lazybridgeframework import LazyAgent, LazyTool

    def search(query: str) -> str:
        \"\"\"Search the web.\"\"\"
        ...

    tool = LazyTool.from_function(search)
    ai = LazyAgent("anthropic")
    resp = ai.loop("find the latest AI news", tools=[tool])

Multi-agent session::

    from lazybridgeframework import LazyAgent, LazySession, LazyContext

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

    from lazybridgeframework import LazyAgent, LazySession

    sess = LazySession()
    researcher = LazyAgent("anthropic", name="researcher", session=sess)
    writer     = LazyAgent("openai",    name="writer",     session=sess)

    pipeline = sess.as_tool("research_pipeline", "Run the pipeline", entry_agent=researcher)
    orchestrator = LazyAgent("anthropic")
    orchestrator.loop("coordinate the work", tools=[pipeline])
"""

from lazybridgeframework.core.providers import BaseProvider
from lazybridgeframework.core.structured import StructuredOutputError
from lazybridgeframework.core.types import (
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
from lazybridgeframework.graph.schema import EdgeType, GraphSchema, NodeType
from lazybridgeframework.lazy_agent import LazyAgent
from lazybridgeframework.lazy_context import LazyContext
from lazybridgeframework.lazy_router import LazyRouter
from lazybridgeframework.lazy_run import run_async
from lazybridgeframework.lazy_session import Event, LazySession, TrackLevel
from lazybridgeframework.lazy_store import LazyStore, StoreEntry
from lazybridgeframework.lazy_tool import (
    LazyTool,
    NormalizedToolSet,
    ToolArgumentValidationError,
    ToolSchemaBuilder,
    ToolSchemaBuildError,
    ToolSchemaMode,
)
from lazybridgeframework.memory import Memory

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
