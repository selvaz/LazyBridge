"""Live, opt-in validation of Claude subscription auth and in-process MCP.

Run only from a terminal already authenticated with Claude Code:

    .venv\\Scripts\\python.exe examples\\live_mcp_smoke.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from lazybridge.engines.claude_code.mcp_adapter import to_mcp_tools
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions
from lazybridge.engines.claude_code.sdk_client import AgentSdkClient


@dataclass
class _Definition:
    parameters: dict


class _ProbeTool:
    name = "lazybridge_probe"
    description = "Return a deterministic probe string. Use it when asked to validate the bridge."

    def definition(self) -> _Definition:
        return _Definition({"type": "object", "properties": {}, "additionalProperties": False})

    async def run(self, **kwargs: object) -> str:
        return "lazybridge-mcp-ok"


async def main() -> None:
    client = AgentSdkClient()
    result = await client.run(
        "Call the lazybridge_probe tool exactly once, then reply with its result and nothing else.",
        options=ClaudeSdkOptions(
            cwd=None,
            system_prompt="You are validating an MCP integration. Follow the user request exactly.",
            mcp_tools=to_mcp_tools([_ProbeTool()]),
        ),
    )
    print(f"session_id={result.session_id}")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
