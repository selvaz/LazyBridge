"""Scripted loopback Messages endpoint for Claude's native tool loop."""

from __future__ import annotations

import gzip
import json
from http.server import BaseHTTPRequestHandler
from typing import Any

from ._codex_probe_model import CanaryHandler


class ProbeModel:
    def __init__(self, calls: list[tuple[str, str, dict[str, Any]]]) -> None:
        self.calls = calls
        self.available: set[str] = set()
        self.results: dict[str, dict[str, Any]] = {}
        self.requests = 0

    def handler(self) -> type[BaseHTTPRequestHandler]:
        fixture = self

        class Handler(CanaryHandler):
            def do_POST(self) -> None:
                if not self.path.split("?", 1)[0].endswith("/v1/messages"):
                    self.send_error(404)
                    return
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                body = json.loads(raw)
                # Ignore auxiliary requests such as title generation. Only
                # the turn carrying the fixture marker should get tool calls.
                main = "LAZYBRIDGE_RELEASE_FIXTURE" in json.dumps(body.get("messages", []))
                for message in body.get("messages", []):
                    content = message.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if part.get("type") == "tool_result":
                                fixture.results[part["tool_use_id"]] = part
                content = [{"type": "text", "text": "fixture-ok"}]
                if main:
                    fixture.available.update(tool["name"] for tool in body.get("tools", []) if "name" in tool)
                    if fixture.requests == 0 and fixture.calls:
                        content = [
                            {"type": "tool_use", "id": call_id, "name": tool, "input": arguments}
                            for call_id, tool, arguments in fixture.calls
                        ]
                    fixture.requests += 1
                reason = "tool_use" if content[0]["type"] == "tool_use" else "end_turn"
                message = {
                    "id": "msg_release_fixture",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": content,
                    "stop_reason": reason,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                }
                if body.get("stream"):
                    events: list[dict[str, Any]] = [
                        {
                            "type": "message_start",
                            "message": {**message, "content": [], "stop_reason": None},
                        },
                    ]
                    for index, part in enumerate(content):
                        if part["type"] == "tool_use":
                            start = {**part, "input": {}}
                            delta = {"type": "input_json_delta", "partial_json": json.dumps(part["input"])}
                        else:
                            start = {**part, "text": ""}
                            delta = {"type": "text_delta", "text": part["text"]}
                        events.extend(
                            [
                                {"type": "content_block_start", "index": index, "content_block": start},
                                {"type": "content_block_delta", "index": index, "delta": delta},
                                {"type": "content_block_stop", "index": index},
                            ]
                        )
                    events.extend(
                        [
                            {
                                "type": "message_delta",
                                "delta": {"stop_reason": reason, "stop_sequence": None},
                                "usage": {"output_tokens": 10},
                            },
                            {"type": "message_stop"},
                        ]
                    )
                    data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
                    content_type = "text/event-stream"
                else:
                    data = json.dumps(message)
                    content_type = "application/json"
                payload = data.encode()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return Handler
