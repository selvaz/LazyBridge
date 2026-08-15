"""Adapt normal LazyBridge tools to Codex App Server dynamic tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from typing import Any, Protocol


class ToolLike(Protocol):
    name: str
    description: str | None

    def definition(self) -> Any: ...

    async def run(self, **kwargs: Any) -> Any: ...


def _text(value: Any) -> str:
    if hasattr(value, "text") and callable(value.text):
        return str(value.text())
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def definitions(tools: Iterable[ToolLike]) -> list[dict[str, Any]]:
    """Return App Server function declarations from LazyBridge tool schemas."""
    output: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if tool.name in names:
            raise ValueError(f"Duplicate LazyBridge tool name {tool.name!r}")
        names.add(tool.name)
        schema = getattr(tool.definition(), "parameters", None)
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"Tool {tool.name!r} must expose an object JSON Schema")
        output.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or tool.name,
                "inputSchema": schema,
            }
        )
    return output


def dispatcher(
    tools: Iterable[ToolLike],
    observer: Callable[[str, dict[str, Any]], None] | None = None,
    *,
    tool_timeout: float | None = None,
) -> Callable[[str, dict[str, Any]], Any]:
    """Create the callback App Server invokes for one dynamic tool call.

    ``tool_timeout``, when set, wraps each ``tool.run()`` in
    ``asyncio.wait_for`` — mirroring ``LLMEngine.tool_timeout`` — so one
    hanging LazyBridge tool cannot block the whole Codex turn.
    """
    by_name = {tool.name: tool for tool in tools}

    async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = by_name.get(name)
        if tool is None:
            return {"success": False, "contentItems": [{"type": "inputText", "text": f"Unknown tool: {name}"}]}
        if observer:
            observer("call", {"tool_name": name, "arguments": arguments})
        try:
            if tool_timeout is not None:
                result = await asyncio.wait_for(tool.run(**arguments), timeout=tool_timeout)
            else:
                result = await tool.run(**arguments)
            text = _text(result)
            if observer:
                observer("result", {"tool_name": name, "result": text})
            return {"success": True, "contentItems": [{"type": "inputText", "text": text}]}
        except TimeoutError:
            text = f"Tool {name!r} timed out after {tool_timeout}s"
            if observer:
                observer("timeout", {"tool_name": name, "error": text, "timeout_s": tool_timeout})
            return {"success": False, "contentItems": [{"type": "inputText", "text": text}]}
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            if observer:
                observer("error", {"tool_name": name, "error": text})
            return {"success": False, "contentItems": [{"type": "inputText", "text": text}]}

    return call
