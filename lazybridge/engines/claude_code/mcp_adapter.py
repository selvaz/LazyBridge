"""Bridge LazyBridge-style tools into the Agent SDK MCP tool contract.

This module intentionally depends only on the small public shape of a
LazyBridge Tool: ``name``, ``description``, ``definition()`` and async
``run(**kwargs)``.  The eventual upstream implementation can therefore use
the real ``lazybridge.Tool`` class without a second tool registry.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from .protocol import McpTool


class ToolLike(Protocol):
    name: str
    description: str | None

    def definition(self) -> Any: ...

    async def run(self, **kwargs: Any) -> Any: ...


def _serialise_result(value: Any) -> tuple[str, dict[str, Any] | None]:
    """Return human-readable MCP content plus optional structured content."""
    if hasattr(value, "text") and callable(value.text):
        return str(value.text()), None
    if isinstance(value, str):
        return value, None
    if hasattr(value, "model_dump") and callable(value.model_dump):
        data = value.model_dump(mode="json")
        return json.dumps(data, ensure_ascii=False), data
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False, default=str), value if isinstance(value, dict) else None
    return str(value), None


def _schema(tool: ToolLike) -> dict[str, Any]:
    definition = tool.definition()
    params = getattr(definition, "parameters", None)
    if not isinstance(params, dict) or params.get("type") != "object":
        raise ValueError(f"Tool {tool.name!r} must expose a JSON-Schema object root")
    return params


def to_mcp_tools(
    tools: Iterable[ToolLike],
    *,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
    tool_timeout: float | None = None,
) -> tuple[McpTool, ...]:
    """Adapt normalised LazyBridge tools to in-process MCP tools.

    Failures from a tool are returned as MCP ``isError`` payloads, allowing
    Claude to recover in its own agent loop instead of terminating the whole
    LazyBridge run.  ``tool_timeout``, when set, wraps each ``tool.run()``
    in ``asyncio.wait_for`` — mirroring ``LLMEngine.tool_timeout`` — so one
    hanging LazyBridge tool cannot block the whole Claude Code turn.
    """
    adapted: list[McpTool] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            raise ValueError(f"Duplicate LazyBridge tool name {tool.name!r}")
        seen.add(tool.name)

        async def handler(arguments: dict[str, Any], *, _tool: ToolLike = tool) -> dict[str, Any]:
            if observer:
                observer("call", {"tool_name": _tool.name, "arguments": arguments})
            try:
                if tool_timeout is not None:
                    result = await asyncio.wait_for(_tool.run(**arguments), timeout=tool_timeout)
                else:
                    result = await _tool.run(**arguments)
                text, structured = _serialise_result(result)
                if observer:
                    observer("result", {"tool_name": _tool.name, "result": text})
                response: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
                if structured is not None:
                    response["structuredContent"] = structured
                return response
            except TimeoutError:
                message = f"Tool {_tool.name!r} timed out after {tool_timeout}s"
                if observer:
                    observer("timeout", {"tool_name": _tool.name, "error": message, "timeout_s": tool_timeout})
                return {
                    "content": [{"type": "text", "text": message}],
                    "isError": True,
                }
            except Exception as exc:
                if observer:
                    observer("error", {"tool_name": _tool.name, "error": str(exc)})
                return {
                    "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                    "isError": True,
                }

        adapted.append(
            McpTool(
                name=tool.name,
                description=tool.description or f"LazyBridge tool {tool.name}",
                input_schema=_schema(tool),
                handler=handler,
            )
        )
    return tuple(adapted)
