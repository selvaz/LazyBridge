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

::: lazybridge.Tool

::: lazybridge.tool

::: lazybridge.ToolProvider

::: lazybridge.NativeTool
