---
title: Python AI agent framework documentation
description: Primary documentation for Python agent and workflow frameworks, and the current LazyBridge engine contract.
---

# Python AI agent framework documentation

Start with the projects' primary documentation and validate a versioned task
against your own state, tools and acceptance checks. There is no universal
ranking or comparative benchmark in this documentation.

| Project | Primary documentation |
| --- | --- |
| LangGraph | [Overview](https://docs.langchain.com/oss/python/langgraph/overview) |
| CrewAI | [Documentation](https://docs.crewai.com/) |
| Pydantic AI | [Documentation](https://ai.pydantic.dev/) |
| LlamaIndex | [Python documentation](https://developers.llamaindex.ai/python/framework/) |
| Haystack | [Documentation](https://docs.haystack.deepset.ai/) |
| Microsoft Agent Framework | [Documentation](https://learn.microsoft.com/en-us/agent-framework/) |
| Google ADK | [Documentation](https://google.github.io/adk-docs/) |
| AutoGen | [Documentation](https://microsoft.github.io/autogen/stable/) |

## LazyBridge's workflow contract

Claude Code and Codex act as interchangeable engines behind a Python Agent.
Tools and application Memory/Store/Session references can remain the same;
native transcripts and permissions remain runtime-specific. Wire an
opposite-engine reviewer through verify, and use deterministic acceptance checks
before shipping. Named native sessions/threads support later continuation.

[LazyTools](https://tools.lazybridge.com/) supplies MCP/code support, repository
delegation, durable tickets, project records and quota routing. The quota
routing benchmark is in progress; there are no measured quality, speed or
subscription savings claims here.

Read the [comparison criteria](comparison.md),
[coding engine guides](guides/full/coding-agents.md),
[collaboration guide](guides/full/claude-writes-codex-reviews.md) and
[maturity table](index.md#maturity). The
[quickstart](quickstart.md) demonstrates the contract.
