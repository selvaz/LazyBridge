"""A deterministic loopback Responses endpoint for the native release probes."""

from __future__ import annotations

import gzip
import json
from http.server import BaseHTTPRequestHandler
from typing import Any


class CanaryHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"fixture-ok")

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _tools(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for tool in items:
        if tool.get("type") == "namespace":
            result.update(_tools(tool.get("tools", [])))
        elif isinstance(tool.get("name"), str):
            result[tool["name"]] = tool
    return result


class ProbeModel:
    """Serve scripted calls and capture the native model-facing inventory and outputs."""

    def __init__(self, calls: list[tuple[str, str | dict[str, Any]]]) -> None:
        self.calls = calls
        self.available: dict[str, dict[str, Any]] = {}
        self.outputs: dict[str, Any] = {}
        self.requests = 0

    def handler(self) -> type[BaseHTTPRequestHandler]:
        fixture = self

        class Handler(CanaryHandler):
            def do_POST(self) -> None:
                if not self.path.endswith("/responses"):
                    self.send_error(404)
                    return
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                body = json.loads(raw)
                if fixture.requests == 0:
                    fixture.available = _tools(body.get("tools", []))
                for item in body.get("input", []):
                    if item.get("type") in {"function_call_output", "custom_tool_call_output"}:
                        fixture.outputs[item["call_id"]] = item.get("output")
                response_id = f"response-{fixture.requests}"
                if fixture.requests < len(fixture.calls):
                    name, arguments = fixture.calls[fixture.requests]
                    item = {"call_id": f"call-{fixture.requests}", "name": name}
                    if isinstance(arguments, str):
                        item.update(type="custom_tool_call", input=arguments)
                    else:
                        item.update(type="function_call", arguments=json.dumps(arguments))
                else:
                    item = {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "fixture-ok"}],
                    }
                fixture.requests += 1
                events = (
                    {"type": "response.created", "response": {"id": response_id}},
                    {"type": "response.output_item.done", "item": item},
                    {"type": "response.completed", "response": {"id": response_id}},
                )
                data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler
