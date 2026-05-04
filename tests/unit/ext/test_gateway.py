from __future__ import annotations

import pytest

from lazybridge.ext.gateway import ExternalToolProvider, ExternalToolSpec
from lazybridge.tools import build_tool_map


class FakeExternalClient:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            {
                "name": "search_mail",
                "description": "Search the user's email.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "send_message",
                "description": "Send a user-approved message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "body"],
                },
            },
        ]

    def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return {"tool": name, "arguments": dict(arguments), "ok": True}


def test_external_tool_spec_from_mapping_accepts_direct_shape():
    spec = ExternalToolSpec.from_mapping(
        {
            "name": "lookup",
            "description": "Lookup a record.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
            "strict": True,
        }
    )

    assert spec.name == "lookup"
    assert spec.description == "Lookup a record."
    assert spec.parameters["properties"]["id"]["type"] == "string"
    assert spec.strict is True


def test_external_tool_spec_from_mapping_accepts_openai_function_shape():
    spec = ExternalToolSpec.from_mapping(
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup a record.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )

    assert spec.name == "lookup"
    assert spec.description == "Lookup a record."


def test_external_tool_provider_expands_into_lazybridge_tools_and_invokes_remote_tool():
    client = FakeExternalClient()
    provider = ExternalToolProvider(client)

    tools = build_tool_map([provider])

    assert set(tools) == {"search_mail", "send_message"}
    definition = tools["search_mail"].definition()
    assert definition.name == "search_mail"
    assert definition.parameters["required"] == ["query"]

    result = tools["search_mail"].run_sync(query="from:alice")

    assert result == {"tool": "search_mail", "arguments": {"query": "from:alice"}, "ok": True}
    assert client.calls == [("search_mail", {"query": "from:alice"})]


def test_external_tool_provider_filters_and_prefixes_tools():
    client = FakeExternalClient()
    provider = ExternalToolProvider(client, include={"send_message"}, name_prefix="pd_")

    tools = build_tool_map([provider])

    assert set(tools) == {"pd_send_message"}
    result = tools["pd_send_message"].run_sync(to="user@example.com", body="hello")

    assert result["tool"] == "send_message"
    assert client.calls == [("send_message", {"to": "user@example.com", "body": "hello"})]


def test_external_tool_provider_exclude_removes_tools():
    client = FakeExternalClient()
    provider = ExternalToolProvider(client, exclude={"send_message"})

    tools = build_tool_map([provider])

    assert set(tools) == {"search_mail"}


def test_external_tool_spec_requires_name():
    with pytest.raises(ValueError, match="non-empty string name"):
        ExternalToolSpec.from_mapping({"description": "missing name"})
