from __future__ import annotations

import asyncio

import pytest

from lazybridge.engines.claude_code.mcp_adapter import to_mcp_tools
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions
from lazybridge.engines.claude_code.sdk_client import AgentSdkClient
from lazybridge.engines.coding import ApprovalDecision

# Both tests below call AgentSdkClient._sdk_options() directly, which builds
# a real claude_agent_sdk.ClaudeAgentOptions — needs lazybridge[claude-code].
# CI's standard test job does not install that optional extra.
pytest.importorskip("claude_agent_sdk", reason="needs lazybridge[claude-code]")


class _Definition:
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}


class _Tool:
    name = "ping"
    description = "Return pong."

    def definition(self):
        return _Definition()

    async def run(self, **kwargs):
        return "pong"


def test_real_sdk_options_are_strict_and_mcp_only():
    options = ClaudeSdkOptions(model="sonnet", mcp_tools=to_mcp_tools([_Tool()]))
    sdk_options = AgentSdkClient._sdk_options(options)

    assert sdk_options.tools == []
    assert sdk_options.strict_mcp_config is True
    assert sdk_options.setting_sources == []
    assert sdk_options.allowed_tools == ["mcp__lazybridge__ping"]
    assert sdk_options.model == "sonnet"
    assert sdk_options.max_turns == 20
    assert set(sdk_options.mcp_servers) == {"lazybridge"}


def test_readonly_builtins_are_configured_separately_from_application_tools():
    options = ClaudeSdkOptions(
        cwd="C:\\workspace",
        builtin_tools=("Read", "Glob", "Grep", "WebSearch", "WebFetch"),
        file_roots=("C:\\workspace",),
    )
    sdk_options = AgentSdkClient._sdk_options(options)

    assert sdk_options.tools == ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]
    assert sdk_options.permission_mode == "default"


def test_output_format_reaches_the_sdk_options():
    schema = {"type": "object", "properties": {"symbol": {"type": "string"}}}
    options = ClaudeSdkOptions(output_format={"type": "json_schema", "schema": schema})

    sdk_options = AgentSdkClient._sdk_options(options)

    # The SDK turns this into the CLI's --json-schema flag; dropping it here
    # would silently downgrade structured output to "hope it answers JSON".
    assert sdk_options.output_format == {"type": "json_schema", "schema": schema}


def test_stream_options_preserve_structured_output_configuration():
    schema = {"type": "object", "properties": {"symbol": {"type": "string"}}}
    options = ClaudeSdkOptions(
        model="sonnet",
        output_format={"type": "json_schema", "schema": schema},
    )

    stream_options = AgentSdkClient._stream_options(options)

    assert stream_options.include_partial_messages is True
    assert stream_options.output_format == options.output_format
    assert stream_options.model == options.model


def test_gated_application_tool_is_not_shadowed_by_allowed_tools():
    seen = []

    async def gate(request):
        seen.append(request)
        return ApprovalDecision.allow()

    options = ClaudeSdkOptions(
        cwd="C:\\workspace",
        mcp_tools=to_mcp_tools([_Tool()]),
        preapprove_application_tools=False,
        approval_gate=gate,
        permission_mode="default",
    )
    sdk_options = AgentSdkClient._sdk_options(options)

    assert sdk_options.allowed_tools == []
    assert sdk_options.can_use_tool is not None
    result = asyncio.run(sdk_options.can_use_tool("mcp__lazybridge__ping", {}, object()))
    assert result.behavior == "allow"
    assert seen[0].provider == "claude-code"
    assert seen[0].name == "mcp__lazybridge__ping"


def test_unapproved_application_tool_without_gate_fails_closed():
    options = ClaudeSdkOptions(
        mcp_tools=to_mcp_tools([_Tool()]),
        preapprove_application_tools=False,
        permission_mode="default",
    )
    sdk_options = AgentSdkClient._sdk_options(options)

    assert sdk_options.can_use_tool is not None
    result = asyncio.run(sdk_options.can_use_tool("mcp__lazybridge__ping", {}, object()))
    assert result.behavior == "deny"
