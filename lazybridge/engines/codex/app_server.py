"""Small JSON-RPC client for one ephemeral Codex App Server run.

The wire format here was verified against a live ``codex app-server``
(codex-cli 0.148.0); the authoritative schema comes from
``codex app-server generate-json-schema --out <dir>``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ToolCallback = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def codex_executable() -> str:
    """Locate the ``codex`` CLI.

    ``CODEX_BIN`` wins, then ``PATH`` (npm/global installs), then the Codex
    desktop app's versioned install directory — the latter is *not* added to
    ``PATH``, so its ``bin/<hash>/codex.exe`` layout would otherwise be
    unreachable from ``create_subprocess_exec``.
    """
    if override := os.environ.get("CODEX_BIN"):
        return override
    if found := shutil.which("codex"):
        return found
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "OpenAI" / "Codex" / "bin",
        Path.home() / ".local" / "share" / "OpenAI" / "Codex" / "bin",
    ]
    candidates = [
        exe
        for root in roots
        if root.is_dir()
        for exe in root.glob("*/codex*")
        if exe.is_file() and exe.suffix in ("", ".exe")
    ]
    if candidates:
        return str(max(candidates, key=lambda p: p.stat().st_mtime))
    raise FileNotFoundError(
        "codex CLI not found on PATH or in the Codex app install directory — "
        "install it (`npm install -g @openai/codex`) or set CODEX_BIN to its full path."
    )


@dataclass(frozen=True)
class CodexRunResult:
    """Final result of one Codex App Server turn.

    ``cost_usd`` is always ``0.0``: under ChatGPT-plan auth the App Server
    reports plan rate-limit percentages, never a per-turn price. The field
    exists so ``Envelope.metadata`` stays uniform across engines.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class CodexAppServerClient:
    """One ``codex app-server`` subprocess per run, torn down in ``finally``."""

    def __init__(self, command: tuple[str, ...] | None = None) -> None:
        #: Resolved lazily so importing the package never touches the
        #: filesystem and a missing CLI surfaces at run() time, as a normal
        #: engine error, not at construction.
        self.command = command

    async def run(
        self,
        *,
        prompt: str,
        model: str | None,
        cwd: str | None,
        dynamic_tools: list[dict[str, Any]],
        on_tool_call: ToolCallback,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        effort: str | None = None,
    ) -> CodexRunResult:
        command = self.command or (codex_executable(), "app-server")
        # stderr is DEVNULL, not PIPE: nothing ever reads it here, and an
        # unread PIPE deadlocks the App Server once its stderr buffer fills.
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert process.stdin and process.stdout
        pending: dict[int, asyncio.Future[Any]] = {}
        completed: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        usage: dict[str, Any] = {}
        counter = 0

        async def send(message: dict[str, Any]) -> None:
            assert process.stdin
            process.stdin.write((json.dumps(message) + "\n").encode())
            await process.stdin.drain()

        async def request(method: str, params: dict[str, Any]) -> Any:
            nonlocal counter
            counter += 1
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            pending[counter] = future
            await send({"method": method, "id": counter, "params": params})
            return await future

        async def read_loop() -> None:
            nonlocal usage
            assert process.stdout
            while line := await process.stdout.readline():
                message = json.loads(line)
                method = message.get("method")
                if method is None:
                    # A response to one of our own requests. Dispatching on
                    # the absence of "method" (rather than on "id" alone)
                    # matters: the App Server numbers its requests to us from
                    # 0 with a separate counter, so an ``item/tool/call`` id
                    # can collide with a still-pending client request id.
                    future = pending.pop(message.get("id"), None)
                    if future is not None and not future.done():
                        if "error" in message:
                            future.set_exception(
                                RuntimeError(message["error"].get("message", "Codex App Server error"))
                            )
                        else:
                            future.set_result(message.get("result", {}))
                elif method == "item/tool/call":
                    params = message["params"]
                    try:
                        result = await on_tool_call(params["tool"], params.get("arguments", {}))
                    except Exception as exc:  # defensive protocol response
                        result = {"success": False, "contentItems": [{"type": "inputText", "text": str(exc)}]}
                    await send({"id": message["id"], "result": result})
                elif method == "item/agentMessage/delta":
                    delta = message.get("params", {}).get("delta", "")
                    if delta and on_text:
                        # Awaited so a slow consumer (bounded queue in
                        # CodexEngine.stream()) applies backpressure all the
                        # way back to this reader, same as the sink-based
                        # streaming path in LLMEngine / ClaudeCodeEngine.
                        await on_text(delta)
                elif method == "thread/tokenUsage/updated":
                    # The only place the App Server reports usage — the
                    # ``turn/completed`` payload carries none. ``total`` (not
                    # ``last``) is the whole turn: the thread is ephemeral and
                    # single-turn, and ``last`` would drop the model calls made
                    # before each tool round-trip.
                    total = message.get("params", {}).get("tokenUsage", {}).get("total")
                    if isinstance(total, dict):
                        usage = total
                elif method == "error" and not completed.done():
                    params = message.get("params", {})
                    if not params.get("willRetry"):
                        completed.set_exception(
                            RuntimeError(params.get("error", {}).get("message", "Codex App Server error"))
                        )
                elif method == "turn/completed" and not completed.done():
                    completed.set_result(message["params"]["turn"])

        reader = asyncio.create_task(read_loop())
        try:
            await request(
                "initialize",
                {
                    "clientInfo": {"name": "lazybridge", "title": "LazyBridge", "version": "0.1.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            await send({"method": "initialized", "params": {}})
            # ``sandbox`` is the CLI's kebab-case ``SandboxMode`` enum
            # (read-only / workspace-write / danger-full-access); "readOnly"
            # is rejected outright with "unknown variant".
            thread = await request(
                "thread/start",
                {
                    "model": model,
                    "cwd": cwd,
                    "ephemeral": True,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "dynamicTools": dynamic_tools,
                },
            )
            # ``input`` is the App Server's UserInput union: the text turn plus
            # any image attachments the engine converted.
            turn_params: dict[str, Any] = {
                "threadId": thread["thread"]["id"],
                "input": [{"type": "text", "text": prompt}, *(attachments or [])],
            }
            if effort is not None:
                turn_params["effort"] = effort
            await request("turn/start", turn_params)
            turn = await completed
            if turn.get("status") != "completed":
                raise RuntimeError(turn.get("error", {}).get("message", f"Codex turn {turn.get('status')}"))
            input_tokens = int(usage.get("inputTokens") or 0)
            output_tokens = int(usage.get("outputTokens") or 0)
            text = ""
            for item in reversed(turn.get("items", [])):
                if item.get("type") == "agentMessage":
                    text = item.get("text", "")
                    break
            return CodexRunResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
        finally:
            reader.cancel()
            process.terminate()
            await process.wait()
