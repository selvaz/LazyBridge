# Tool family

Wrap any callable as a `Tool` for an `Agent`. The `Tool.wrap()`
classmethod is the canonical multi-input factory (callable / `Agent`
/ existing `Tool`); `Tool(...)` is the explicit constructor used when
you want to set every field by hand. `ToolProvider` is the protocol
for expandable tool catalogues (MCP servers etc.). `NativeTool`
enumerates provider-hosted server-side tools.

The module-level `lazybridge.tool` (lowercase) is a thin
backwards-compat alias for `Tool.wrap` — existing imports keep
working, new code should prefer the classmethod.

For narrative usage see [Guides → Basic → Tool](../guides/basic/tool.md)
and [Guides → Basic → Native tools](../guides/basic/native-tools.md).

## Timeouts

`Tool(timeout=N)` bounds one tool; `Agent(tool_timeout=N)` supplies a
default to every tool that sets none of its own; `LLMEngine(tool_timeout=N)`
does the same at engine level. The most specific one set wins, and a tool
that exceeds its bound raises `ToolTimeoutError`, which the engine reports
to the model as a failed tool result rather than aborting the run.

Bound the tool, not just the run. `Agent(timeout=N)` can only fire at an
`await`, and a **synchronous** tool never yields one — a blocking
`time.sleep`/`requests.get` inside a tool will run past the agent deadline
indefinitely. `Tool(timeout=N)` instead runs the call on a daemon thread and
**abandons** it when the time is out: the caller is freed immediately, the
work itself keeps running until it returns on its own, and its result is
discarded. Anything with a side effect may therefore still complete after
the timeout — for work that must actually stop, give the underlying library
its own deadline (`requests.get(..., timeout=)`) or run it in a subprocess.

::: lazybridge.Tool

::: lazybridge.tool

::: lazybridge.ToolTimeoutError

::: lazybridge.ToolProvider

::: lazybridge.NativeTool
