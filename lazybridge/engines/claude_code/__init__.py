"""``ClaudeCodeEngine`` — a standard LazyBridge ``Engine`` backed by the
locally authenticated Claude Code runtime (Claude Agent SDK).

Requires the optional ``claude-agent-sdk``/``mcp`` dependencies:
``pip install "lazybridge[claude-code]"``. See
:doc:`/guides/full/claude-code-engine` for setup, usage, and configuration.
"""

from .engine import ClaudeCodeEngine
from .protocol import ClaudeSdkClient, ClaudeSdkOptions, ClaudeSdkResult, ClaudeSdkStreamEvent, McpTool
from .sdk_client import AgentSdkClient

__all__ = [
    "AgentSdkClient",
    "ClaudeCodeEngine",
    "ClaudeSdkClient",
    "ClaudeSdkOptions",
    "ClaudeSdkResult",
    "ClaudeSdkStreamEvent",
    "McpTool",
]
