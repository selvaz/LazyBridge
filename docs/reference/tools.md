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

An **async** tool is cancelled rather than abandoned, but cancelling is a
request and not a guarantee: a coroutine may catch `CancelledError` and carry
on, or spend a long time in cleanup. It gets `Tool.cancel_grace_seconds`
(1.0) to unwind, after which it too is abandoned.

One case no deadline can reach: a coroutine that blocks the event loop —
CPU-bound work or a synchronous call inside `async def`, whether in the body
or in cancellation cleanup. Nothing else runs while it does, including the
clock that would end it. That is a property of `asyncio`, not of this bound;
the fix is to keep blocking work out of `async def` (declare the tool `def`
and let `Tool(timeout=)` put it on its own thread, or use
`run_in_executor`).

The bound is on the **call**, not on process exit. An abandoned task still
belongs to its event loop, and `asyncio.run` cancels *and gathers* every
pending task on the way out — so an async tool that swallows `CancelledError`
outright can delay shutdown even though the call itself returned on time.
`run_sync()` is unaffected: LazyBridge owns that loop and skips draining what
it has already abandoned.

::: lazybridge.Tool

::: lazybridge.tool

::: lazybridge.ToolTimeoutError

::: lazybridge.ToolProvider

::: lazybridge.NativeTool
