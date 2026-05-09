# Tool family

Wrap any callable as a `Tool` for an `Agent`. The `tool(...)` factory
is the canonical entry point; `Tool` exposes the explicit constructor
and the `from_schema` path. `ToolProvider` is the protocol for
expandable tool catalogues (MCP servers etc.). `NativeTool` enumerates
provider-hosted server-side tools.

For narrative usage see [Guides → Basic → Tool](../guides/basic/tool.md)
and [Guides → Basic → Native tools](../guides/basic/native-tools.md).

::: lazybridge.tool

::: lazybridge.Tool

::: lazybridge.ToolProvider

::: lazybridge.NativeTool
