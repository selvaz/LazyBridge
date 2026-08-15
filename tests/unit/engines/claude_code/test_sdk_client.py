from __future__ import annotations

import pytest

from lazybridge.engines.claude_code.mcp_adapter import to_mcp_tools
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions
from lazybridge.engines.claude_code.sdk_client import AgentSdkClient

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
