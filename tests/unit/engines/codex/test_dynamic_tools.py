from __future__ import annotations

import asyncio
from dataclasses import dataclass

from lazybridge.engines.codex.dynamic_tools import definitions, dispatcher


@dataclass
class Definition:
    parameters: dict


class QuoteTool:
    name = "get_quote"
    description = "Return a quote."

    def definition(self):
        return Definition({"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]})

    async def run(self, **kwargs):
        return {"symbol": kwargs["symbol"], "price": 123.45}


def test_lazybridge_tool_becomes_a_dynamic_function_and_dispatches():
    tool = QuoteTool()
    declaration = definitions([tool])[0]
    assert declaration["name"] == "get_quote"
    assert declaration["inputSchema"]["required"] == ["symbol"]

    result = asyncio.run(dispatcher([tool])("get_quote", {"symbol": "AMZN"}))
    assert result["success"] is True
    assert "AMZN" in result["contentItems"][0]["text"]
