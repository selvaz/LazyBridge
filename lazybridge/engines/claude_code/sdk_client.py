"""Real Claude Agent SDK client adapter.

The rest of the prototype depends only on ``ClaudeSdkClient``.  This module is
the sole optional-import boundary for the external SDK.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from lazybridge.engines.coding import ApprovalRequest, ask_approval

from .protocol import ClaudeSdkClient, ClaudeSdkOptions, ClaudeSdkResult, ClaudeSdkStreamEvent, McpTool

#: Set once ``CanUseToolShadowedWarning`` has been added to the process-wide
#: warnings filter (see ``_sdk_options``). A plain flag, not a per-call
#: ``warnings.catch_warnings()`` context: that context manager mutates
#: global filter state without coroutine isolation, so two concurrent
#: ``AgentSdkClient`` calls (e.g. via ``asyncio.gather``) could clobber each
#: other's filters. Registering the ignore rule once, idempotently, avoids
#: that hazard entirely.
_shadow_warning_filtered = False


class ClaudeSdkRequestError(RuntimeError):
    """Raised when the Claude Agent SDK reports a failed run.

    Carries the CLI's ``api_error_status`` (HTTP status of the failing API
    call — e.g. 429/500/529 — when the CLI version reports one; ``None``
    otherwise) so ``ClaudeCodeEngine``'s retry classifier can distinguish a
    transient provider failure from a permanent one instead of guessing
    from the error string.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class AgentSdkClient(ClaudeSdkClient):
    """Run Claude Code through the installed Agent SDK.

    The SDK uses the same local Claude Code authentication as the CLI.  No
    credential is passed to, read by, or persisted by this adapter.
    """

    @staticmethod
    async def _prompt_stream(
        prompt: str, attachments: tuple[dict[str, Any], ...] = ()
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield one user message for SDK permission-callback mode.

        The Agent SDK requires an async prompt stream whenever ``can_use_tool``
        is configured.  Keeping this conversion here lets the engine retain
        its simple string-in/string-out contract.

        It is also the only way to send attachments: a plain-string prompt has
        nowhere to put an image block, so ``content`` becomes the block list
        ``[text, image...]`` whenever the run carries any.
        """
        content: str | list[dict[str, Any]] = (
            [{"type": "text", "text": prompt}, *attachments] if attachments else prompt
        )
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
        }

    @staticmethod
    def _sdk_options(options: ClaudeSdkOptions) -> Any:
        global _shadow_warning_filtered
        try:
            from claude_agent_sdk import (
                CanUseToolShadowedWarning,
                ClaudeAgentOptions,
                HookMatcher,
                PermissionResultAllow,
                PermissionResultDeny,
                create_sdk_mcp_server,
                tool,
            )
        except ImportError as exc:  # pragma: no cover - exercised by users without the extra
            raise ImportError(
                "Claude Code support requires the optional dependencies. "
                'Install with: pip install "lazybridge[claude-code]"'
            ) from exc

        if not _shadow_warning_filtered:
            # See can_use_tool() below: MCP tool calls are intentionally
            # pre-approved via allowed_tools and therefore always shadow
            # that callback. The SDK's own docstring for this warning calls
            # that case "advisory only... shadowing can be intentional" —
            # which is exactly our design, not a misconfiguration to fix.
            warnings.filterwarnings("ignore", category=CanUseToolShadowedWarning)
            _shadow_warning_filtered = True

        sdk_tools = []
        for mcp_tool in options.mcp_tools:
            # ``tool`` accepts a full JSON Schema dict, preserving the schema
            # LazyBridge already compiled rather than re-inferring it.
            @tool(mcp_tool.name, mcp_tool.description, mcp_tool.input_schema)
            async def _handler(args: dict[str, Any], *, _mcp_tool: McpTool = mcp_tool) -> dict[str, Any]:
                # McpTool.handler is Callable[..., Awaitable[Any]] — the result is
                # a dict by MCP tool-result convention, but not statically known.
                return cast("dict[str, Any]", await _mcp_tool.handler(args))

            sdk_tools.append(_handler)

        server = create_sdk_mcp_server(
            name=options.mcp_server_name,
            version="0.1.0",
            tools=sdk_tools,
        )
        allowed = list(options.allowed_tools)
        if options.preapprove_application_tools:
            allowed.extend(f"mcp__{options.mcp_server_name}__{item.name}" for item in options.mcp_tools)

        roots = tuple(Path(root).resolve() for root in options.file_roots)

        def allowed_path(value: object) -> bool:
            if not isinstance(value, str) or not roots:
                return False
            path = Path(value)
            resolved = (Path(options.cwd) / path if not path.is_absolute() and options.cwd else path).resolve()
            return any(resolved.is_relative_to(root) for root in roots)

        async def can_use_tool(name: str, arguments: dict[str, Any], context: Any) -> Any:
            # In the compatibility profile MCP tools are pre-approved and do
            # not reach this callback. Gated profiles deliberately omit those
            # allow rules, so application tools and unmatched built-ins share
            # this one normalized approval path.
            if options.approval_gate is not None:
                decision = await ask_approval(
                    options.approval_gate,
                    ApprovalRequest(
                        provider="claude-code",
                        kind="tool",
                        name=name,
                        arguments=arguments,
                        reason=getattr(context, "decision_reason", None),
                        cwd=options.cwd,
                        raw={
                            "tool_use_id": getattr(context, "tool_use_id", None),
                            "title": getattr(context, "title", None),
                            "description": getattr(context, "description", None),
                        },
                    ),
                )
                if decision.action in {"allow", "allow_session"}:
                    updates = getattr(context, "suggestions", None) if decision.action == "allow_session" else None
                    return PermissionResultAllow(updated_permissions=updates or None)
                return PermissionResultDeny(
                    message=decision.message or "Action denied by approval gate",
                    interrupt=decision.action == "cancel",
                )
            if name in {"WebSearch", "WebFetch"}:
                return PermissionResultAllow()
            if name in {"Read", "Glob", "Grep"}:
                path = arguments.get("file_path") or arguments.get("path") or options.cwd
                if allowed_path(path):
                    return PermissionResultAllow()
                return PermissionResultDeny(message="File access is outside ClaudeCodeEngine.file_roots")
            # Defensive fail-closed path: either an application tool has no
            # gate, or an undeclared built-in reached the callback.
            return PermissionResultDeny(message=f"Tool {name!r} is not pre-approved and no approval gate allowed it")

        async def keep_permission_stream_open(
            input_data: dict[str, Any], tool_use_id: str | None, context: Any
        ) -> dict[str, Any]:
            return {"continue_": True}

        use_callback = (
            bool(options.builtin_tools) or options.approval_gate is not None or not options.preapprove_application_tools
        )
        return ClaudeAgentOptions(
            model=options.model,
            fallback_model=options.fallback_model,
            # ``ClaudeSdkOptions.reasoning_effort``/``.thinking`` stay loosely
            # typed (str / dict) in protocol.py so it never imports the SDK;
            # ``ClaudeCodeEngine.__init__`` already validates the literal
            # values/shape at runtime before they reach here.
            effort=options.reasoning_effort,  # type: ignore[arg-type]
            thinking=options.thinking,  # type: ignore[arg-type]
            cwd=options.cwd,
            system_prompt=options.system_prompt,
            max_turns=options.max_turns,
            resume=options.resume,
            tools=list(options.builtin_tools),
            allowed_tools=allowed,
            disallowed_tools=list(options.disallowed_tools),
            mcp_servers={options.mcp_server_name: server},
            strict_mcp_config=True,
            permission_mode=options.permission_mode or ("default" if use_callback else "dontAsk"),
            can_use_tool=can_use_tool if use_callback else None,
            # ``keep_permission_stream_open`` intentionally uses the generic
            # (dict, str | None, Any) shape rather than the SDK's per-hook
            # TypedDict union — this hook only ever fires for PreToolUse.
            hooks={"PreToolUse": [HookMatcher(hooks=[keep_permission_stream_open])]} if use_callback else None,  # type: ignore[list-item]
            setting_sources=list(options.setting_sources),
            include_partial_messages=options.include_partial_messages,
            output_format=options.output_format,
        )

    async def run(
        self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict[str, Any], ...] = ()
    ) -> ClaudeSdkResult:
        try:
            from claude_agent_sdk import ResultMessage, query
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Claude Agent SDK is not installed") from exc

        final: Any | None = None
        sdk_options = self._sdk_options(options)
        sdk_prompt: str | AsyncIterator[dict[str, Any]] = (
            self._prompt_stream(prompt, attachments) if options.builtin_tools or attachments else prompt
        )
        async for message in query(prompt=sdk_prompt, options=sdk_options):
            if isinstance(message, ResultMessage):
                final = message
        if final is None:
            raise RuntimeError("Claude Agent SDK ended without a ResultMessage")
        if final.is_error:
            detail = "; ".join(final.errors or []) or final.result or final.subtype
            raise ClaudeSdkRequestError(f"Claude Agent SDK failed: {detail}", status=final.api_error_status)

        usage = final.usage or {}
        return ClaudeSdkResult(
            # With ``output_format`` set, the CLI returns the schema-validated
            # object on ``structured_output``; ``result`` holds the same JSON
            # as text on current versions but is not guaranteed to. Serialising
            # the parsed object keeps this adapter's string contract while
            # giving ``Agent._validate_and_retry`` something that always parses.
            text=json.dumps(final.structured_output) if final.structured_output is not None else (final.result or ""),
            session_id=final.session_id,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cost_usd=float(final.total_cost_usd or 0.0),
            model=None,
        )

    @staticmethod
    def _stream_options(options: ClaudeSdkOptions) -> ClaudeSdkOptions:
        """Enable partial events without dropping any per-run SDK option."""
        return replace(options, include_partial_messages=True)

    async def stream(
        self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict[str, Any], ...] = ()
    ) -> AsyncIterator[ClaudeSdkStreamEvent]:
        try:
            from claude_agent_sdk import ResultMessage, StreamEvent, query
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Claude Agent SDK is not installed") from exc

        saw_text = False
        saw_result = False
        stream_options = self._stream_options(options)
        sdk_prompt: str | AsyncIterator[dict[str, Any]] = (
            self._prompt_stream(prompt, attachments) if stream_options.builtin_tools or attachments else prompt
        )
        async for message in query(
            prompt=sdk_prompt,
            options=self._sdk_options(stream_options),
        ):
            if isinstance(message, StreamEvent):
                event = message.event
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                        saw_text = True
                        yield ClaudeSdkStreamEvent(text=delta["text"])
            elif isinstance(message, ResultMessage):
                saw_result = True
                if message.is_error:
                    detail = "; ".join(message.errors or []) or message.result or message.subtype
                    raise ClaudeSdkRequestError(f"Claude Agent SDK failed: {detail}", status=message.api_error_status)
                # Some SDK/CLI versions do not emit partial text for a short
                # response. Yield the final response exactly once in that case.
                if not saw_text:
                    final_text = (
                        json.dumps(message.structured_output)
                        if message.structured_output is not None
                        else (message.result or "")
                    )
                    if final_text:
                        yield ClaudeSdkStreamEvent(text=final_text)
                usage = message.usage or {}
                yield ClaudeSdkStreamEvent(
                    session_id=message.session_id,
                    input_tokens=int(usage.get("input_tokens", 0) or 0),
                    output_tokens=int(usage.get("output_tokens", 0) or 0),
                    cost_usd=float(message.total_cost_usd or 0.0),
                    final=True,
                )
        if not saw_result:
            # Mirrors AgentSdkClient.run(): the SDK iterator ended without a
            # ResultMessage. Silently returning here would let the engine
            # record an empty/partial stream as a successful turn (stale
            # memory write, false AGENT_FINISH, and — in session_mode=
            # "runtime" — a resume ID that was never actually confirmed).
            raise RuntimeError("Claude Agent SDK stream ended without a ResultMessage")
