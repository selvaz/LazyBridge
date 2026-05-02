# Recipes

End-to-end, copy-paste examples organised by task.
Each recipe is self-contained: it shows the imports, the agents, the call, and what to look for in the result.

| Recipe | Tier | What it covers |
|---|---|---|
| [Tool calling](tool-calling.md) | Basic | Function → tool → agent; multi-tool; session inspection |
| [Structured output](structured-output.md) | Basic | Pydantic model output; `.payload` vs `.text()`; with tools |
| [Plan with resume](plan-with-resume.md) | Full | Typed steps; conditional routing; checkpoint/resume after crash |
| [Human-in-the-loop](human-in-the-loop.md) | Mid/Full | Approval gate (HumanEngine) and REPL supervision (SupervisorEngine) |
| [Orchestration tools](orchestration-tools.md) | Full | Outer agent composes work over a sub-agent registry via chain / parallel / plan tools |
| [MCP integration](mcp.md) | Ext | Connect to any Model Context Protocol server (stdio or Streamable HTTP) as a tool collection |

Not sure which recipe fits? → [Decision trees](../decisions/index.md)
