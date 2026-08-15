"""Live, opt-in validation of Codex App Server auth and the JSON-RPC bridge.

Run only from a terminal with a working, authenticated ``codex`` CLI:

    .venv\\Scripts\\python.exe examples\\codex\\live_app_server_smoke.py

It also prints the token usage the client parsed out of the
``thread/tokenUsage/updated`` notifications. ``cost_usd`` is expected to be
``0.0``: the App Server reports no per-turn price under ChatGPT-plan auth
(see docs/guides/full/codex-engine.md).
"""

from __future__ import annotations

import asyncio
import json

from lazybridge.engines.codex.app_server import CodexAppServerClient


async def _probe_tool(name: str, arguments: dict) -> dict:
    assert name == "lazybridge_probe"
    return {"success": True, "contentItems": [{"type": "inputText", "text": "lazybridge-app-server-ok"}]}


async def main() -> None:
    client = CodexAppServerClient()
    result = await client.run(
        prompt="Call the lazybridge_probe tool exactly once, then reply with its result and nothing else.",
        model=None,
        cwd=None,
        dynamic_tools=[
            {
                "type": "function",
                "name": "lazybridge_probe",
                "description": "Return a deterministic probe string.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            }
        ],
        on_tool_call=_probe_tool,
    )
    print(result.text)
    print(
        json.dumps(
            {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
            }
        )
    )
    if result.input_tokens == 0 and result.output_tokens == 0:
        print(
            "NOTE: token usage came back 0 — the App Server stopped emitting "
            "thread/tokenUsage/updated in the shape app_server.py reads "
            "(params.tokenUsage.total.inputTokens/outputTokens). Log the raw "
            "notifications to see what replaced it."
        )


if __name__ == "__main__":
    asyncio.run(main())
