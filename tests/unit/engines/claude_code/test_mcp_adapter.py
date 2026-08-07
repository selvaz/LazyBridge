from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from lazybridge.engines.claude_code.mcp_adapter import to_mcp_tools


@dataclass
class _Definition:
    parameters: dict


class _Tool:
    name = "get_quote"
    description = "Return a deterministic quote."

    def definition(self):
        return _Definition(
            {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            }
        )

    async def run(self, **kwargs):
        return {"symbol": kwargs["symbol"], "price": 123.45}


def test_adapter_exposes_schema_and_structured_result():
    (mcp_tool,) = to_mcp_tools([_Tool()])

    assert mcp_tool.name == "get_quote"
    assert mcp_tool.input_schema["required"] == ["symbol"]
    result = asyncio.run(mcp_tool.handler({"symbol": "AMZN"}))

    assert result["structuredContent"] == {"symbol": "AMZN", "price": 123.45}
    assert result["content"][0]["type"] == "text"


def test_adapter_returns_tool_error_as_mcp_error():
    class BrokenTool(_Tool):
        async def run(self, **kwargs):
            raise RuntimeError("upstream unavailable")

    (mcp_tool,) = to_mcp_tools([BrokenTool()])
    result = asyncio.run(mcp_tool.handler({"symbol": "AMZN"}))

    assert result["isError"] is True
    assert "upstream unavailable" in result["content"][0]["text"]


def test_adapter_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate"):
        to_mcp_tools([_Tool(), _Tool()])
