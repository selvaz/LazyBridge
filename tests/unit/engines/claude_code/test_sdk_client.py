from __future__ import annotations

import asyncio
import dataclasses

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


def test_permission_mode_is_chosen_per_run_when_the_policy_leaves_it_open():
    """A fully pre-approved, tool-only agent must not be put in prompting mode.

    ``permission_mode`` defaulting to a literal ``"default"`` in the policy
    would make the engine's own choice unreachable: nothing can answer a
    prompt when no ``can_use_tool`` callback is configured.
    """
    tool_only = ClaudeSdkOptions(mcp_tools=to_mcp_tools([_Tool()]))
    with_builtins = ClaudeSdkOptions(builtin_tools=("Read",), file_roots=(r"C:\workspace",))

    assert AgentSdkClient._sdk_options(tool_only).permission_mode == "dontAsk"
    assert AgentSdkClient._sdk_options(with_builtins).permission_mode == "default"


def test_an_explicit_permission_mode_still_wins():
    options = ClaudeSdkOptions(mcp_tools=to_mcp_tools([_Tool()]), permission_mode="acceptEdits")

    assert AgentSdkClient._sdk_options(options).permission_mode == "acceptEdits"


def _unmatched_hook(sdk_options, event: str):
    """The all-tools hook, whose matcher intentionally is omitted."""
    return next(entry.hooks[0] for entry in sdk_options.hooks[event] if entry.matcher is None)


def test_native_tool_observer_receives_calls_and_results_without_observing_mcp_tools():
    seen = []
    sdk_options = AgentSdkClient._sdk_options(
        ClaudeSdkOptions(tool_observer=lambda kind, payload: seen.append((kind, payload)))
    )

    assert set(sdk_options.hooks) >= {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
    call_hook = _unmatched_hook(sdk_options, "PreToolUse")
    result_hook = _unmatched_hook(sdk_options, "PostToolUse")
    failure_hook = _unmatched_hook(sdk_options, "PostToolUseFailure")
    assert asyncio.run(call_hook({"tool_name": "Bash", "tool_input": {"command": "git status"}}, "t1", object())) == {}
    assert asyncio.run(result_hook({"tool_name": "Bash", "tool_response": {"stdout": "x"}}, "t1", object())) == {}
    assert asyncio.run(failure_hook({"tool_name": "Bash", "error": "interrupted"}, "t1", object())) == {}
    assert seen == [
        ("call", {"tool_name": "Bash", "arguments": {"command": "git status"}, "tool_use_id": "t1", "native": True}),
        ("result", {"tool_name": "Bash", "result": {"stdout": "x"}, "tool_use_id": "t1", "native": True}),
        ("error", {"tool_name": "Bash", "error": "interrupted", "tool_use_id": "t1", "native": True}),
    ]

    assert asyncio.run(call_hook({"tool_name": "mcp__lazybridge__foo", "tool_input": {}}, "t2", object())) == {}
    assert len(seen) == 3


def test_native_tool_observer_errors_cannot_break_a_tool_call():
    def broken(kind, payload):
        raise ValueError("observer unavailable")

    sdk_options = AgentSdkClient._sdk_options(ClaudeSdkOptions(tool_observer=broken))
    call_hook = _unmatched_hook(sdk_options, "PreToolUse")

    with pytest.warns(UserWarning, match="native tool observer raised ValueError"):
        assert asyncio.run(call_hook({"tool_name": "Bash", "tool_input": {}}, "t1", object())) == {}


def test_no_native_observer_leaves_post_tool_use_unregistered():
    sdk_options = AgentSdkClient._sdk_options(ClaudeSdkOptions())

    assert "PostToolUse" not in (sdk_options.hooks or {})


# --------------------------------------------------------------------------- #
# Streaming input follows the SDK hooks (release/1.4.0 regression)
#
# ``_sdk_options`` registers the native ``tool_observer`` hooks whenever
# ``options.tool_observer is not None`` — regardless of ``builtin_tools`` or
# ``attachments``. But the Agent SDK only delivers hook callbacks (and
# ``can_use_tool``) over STREAMING input; a plain string prompt is answered
# without ever consulting them. A ``ClaudeCodeEngine`` used as a pure LLM
# (no file_roots, no web, no extra_tools) but with a ``Session`` to observe
# used to register the hooks in ``_sdk_options`` while ``run``/``stream``
# still chose the plain-string prompt — hooks with nothing to invoke them,
# which fails the SDK call before the model is even queried.
# --------------------------------------------------------------------------- #
def test_an_observer_alone_requires_streaming_input():
    """No builtin tools, no attachments — only a tool_observer.

    This is exactly the case from the fix spec's reproduction: a bare
    ``ClaudeCodeEngine`` whose only reason to register hooks is the
    (session-backed) observer. Streaming input must still be selected, or
    the hooks ``_sdk_options`` just registered fire for nobody.
    """
    options = ClaudeSdkOptions(tool_observer=lambda kind, payload: None)

    assert AgentSdkClient._needs_streaming_input(options, ()) is True


def test_no_observer_no_builtins_no_attachments_stays_on_the_plain_prompt():
    """The baseline case: nothing here needs the richer message shape."""
    assert AgentSdkClient._needs_streaming_input(ClaudeSdkOptions(), ()) is False


def test_builtin_tools_alone_still_require_streaming_input():
    """Already-working case (e.g. ``file_roots`` or ``web``): unchanged."""
    options = ClaudeSdkOptions(builtin_tools=("Read", "Glob", "Grep"))

    assert AgentSdkClient._needs_streaming_input(options, ()) is True


def test_attachments_alone_still_require_streaming_input():
    """Already-working case: unaffected by the tool_observer addition."""
    assert AgentSdkClient._needs_streaming_input(ClaudeSdkOptions(), ({"type": "image"},)) is True


def test_builtin_tools_and_observer_together_still_require_streaming_input():
    """The combined case must not regress either: still exactly one decision."""
    options = ClaudeSdkOptions(
        builtin_tools=("WebSearch", "WebFetch"),
        tool_observer=lambda kind, payload: None,
    )

    assert AgentSdkClient._needs_streaming_input(options, ()) is True


def _pre_tool_use_hook(sdk_options, matcher_contains: str):
    """The PreToolUse callback whose matcher covers ``matcher_contains``."""
    for entry in sdk_options.hooks["PreToolUse"]:
        if entry.matcher and matcher_contains in entry.matcher:
            return entry.hooks[0]
    raise AssertionError(f"no PreToolUse matcher covering {matcher_contains!r}")


class TestFileConfinement:
    """``file_roots`` is enforced by a PreToolUse hook, not by can_use_tool.

    The SDK evaluates hooks first and the callback last, and a tool approved
    by an allow rule or a permissive mode never reaches the callback at all.
    Confinement that only lived there was bypassable by configuration.
    """

    @staticmethod
    def _options(tmp_path, **kwargs):
        return ClaudeSdkOptions(
            model="sonnet",
            cwd=str(tmp_path),
            file_roots=(str(tmp_path),),
            builtin_tools=("Read", "Glob", "Grep"),
            mcp_tools=to_mcp_tools([_Tool()]),
            **kwargs,
        )

    def test_a_hook_guards_the_file_tools(self, tmp_path):
        sdk_options = AgentSdkClient._sdk_options(self._options(tmp_path))

        hook = _pre_tool_use_hook(sdk_options, "Read")
        # Edit/Write are covered too although this profile does not grant
        # them: a settings-added writer must not slip past the check.
        matcher = next(e.matcher for e in sdk_options.hooks["PreToolUse"] if e.matcher)
        assert "Edit" in matcher and "Write" in matcher

        inside = asyncio.run(hook({"tool_input": {"file_path": str(tmp_path / "a.py")}}, None, object()))
        assert inside == {}

    def test_a_path_outside_the_roots_is_denied(self, tmp_path):
        outside = tmp_path.parent / "elsewhere.env"
        sdk_options = AgentSdkClient._sdk_options(self._options(tmp_path))

        hook = _pre_tool_use_hook(sdk_options, "Read")
        result = asyncio.run(hook({"tool_input": {"file_path": str(outside)}}, None, object()))

        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "file_roots" in result["hookSpecificOutput"]["permissionDecisionReason"]

    def test_confinement_survives_a_permissive_mode(self, tmp_path):
        # bypassPermissions auto-approves everything that reaches the mode
        # step, so can_use_tool is never consulted — the hook is the only
        # layer left, and a hook deny holds even there.
        sdk_options = AgentSdkClient._sdk_options(self._options(tmp_path, permission_mode="bypassPermissions"))

        hook = _pre_tool_use_hook(sdk_options, "Read")
        result = asyncio.run(hook({"tool_input": {"file_path": str(tmp_path.parent / "x")}}, None, object()))

        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_confinement_survives_an_allow_rule(self, tmp_path):
        # allowed_tools=["Read"] auto-approves Read before the callback.
        sdk_options = AgentSdkClient._sdk_options(self._options(tmp_path, allowed_tools=("Read",)))

        hook = _pre_tool_use_hook(sdk_options, "Read")
        result = asyncio.run(hook({"tool_input": {"file_path": str(tmp_path.parent / "x")}}, None, object()))

        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_no_roots_means_no_confinement_hook(self, tmp_path):
        sdk_options = AgentSdkClient._sdk_options(ClaudeSdkOptions(model="sonnet", mcp_tools=to_mcp_tools([_Tool()])))

        matchers = (sdk_options.hooks or {}).get("PreToolUse", [])
        assert not [e for e in matchers if e.matcher]


# --------------------------------------------------------------------------- #
# Per-agent auto-compaction window
# --------------------------------------------------------------------------- #
def test_the_compaction_window_travels_as_an_env_var_not_a_settings_file():
    """The env var is what an agent can actually be given privately.

    Settings files are shared and, worse, an agent reads none of them by
    default (``setting_sources`` is empty), so a value written there would
    reach this agent only if it were also told to inherit a human's personal
    settings wholesale.
    """
    sdk_options = AgentSdkClient._sdk_options(ClaudeSdkOptions(auto_compact_window=140_000))

    assert sdk_options.env == {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "140000"}
    assert sdk_options.setting_sources == []


def test_no_window_means_no_environment_of_our_own():
    """An empty dict, not a stray variable: the CLI keeps its tuned default."""
    assert AgentSdkClient._sdk_options(ClaudeSdkOptions()).env == {}


def test_the_policy_reaches_the_options_the_engine_builds():
    """The field is useless if the engine does not carry it across."""
    from lazybridge.engines.claude_code import ClaudeCodeEngine
    from lazybridge.engines.coding import ClaudeCodePolicy, CodingAgentConfig

    engine = ClaudeCodeEngine(config=CodingAgentConfig(claude=ClaudeCodePolicy(auto_compact_window=90_000)))
    assert engine._options([], None).auto_compact_window == 90_000


def test_a_new_policy_field_never_displaces_an_existing_positional_argument():
    """These dataclasses accept positional construction, so field ORDER is API.

    The first version of the compaction field was inserted in the middle:
    `ClaudeCodePolicy(None, True, (), (), (), ("Write",))` then bound
    `("Write",)` to the window and left `extra_tools` empty, which would have
    shipped `CLAUDE_CODE_AUTO_COMPACT_WINDOW="('Write',)"` into a subprocess
    without a word. Nothing raised. This pins the arrangement that made that
    impossible: additions go last.
    """
    from lazybridge.engines.coding import ClaudeCodePolicy

    policy = ClaudeCodePolicy(None, True, (), (), (), ("Write",))
    assert policy.extra_tools == ("Write",)
    assert policy.auto_compact_window is None

    from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions

    assert [f.name for f in dataclasses.fields(ClaudeSdkOptions)][-2:] == ["auto_compact_window", "tool_observer"]
