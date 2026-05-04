"""Model Context Protocol (MCP) integration for LazyBridge.

MCP is a JSON-RPC protocol for exposing tools, resources, and prompts to
LLM clients. LazyBridge integrates MCP at the **tool boundary** — an
``MCPServer`` acts as a *tool provider* that expands into a list of
:class:`lazybridge.Tool` entries when added to ``Agent(tools=[...])``.

Phase 1 (this release) ships **tools only**. Resources and prompts are
planned for later phases.

Quick start
-----------
::

    from lazybridge import Agent
    from lazybridge.ext.mcp import MCP

    fs = MCP.stdio(
        "fs",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    )

    agent = Agent("claude-opus-4-7", tools=[fs])
    print(agent("Read README.md and summarise the install steps").text())

Multiple servers, mixing transports::

    github = MCP.http("github", "https://example.com/mcp",
                      headers={"Authorization": "Bearer …"})
    db     = MCP.stdio("db", command="uvx", args=["mcp-postgres"])

    agent = Agent("gpt-5.5", tools=[github, db])

Install with::

    pip install lazybridge[mcp]
"""

#: Stability tag — see ``docs/guides/core-vs-ext.md``.
__stability__ = "stable"
__lazybridge_min__ = "1.0.0"

from lazybridge.ext.mcp.server import MCP, MCPServer

__all__ = ["MCP", "MCPServer"]
