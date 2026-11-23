"""Real Claude Agent SDK client adapter.

The rest of the prototype depends only on ``ClaudeSdkClient``.  This module is
the sole optional-import boundary for the external SDK.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import subprocess
import tempfile
import warnings
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

from lazybridge._asyncbridge import abandoned_tasks
from lazybridge.engines._tool_clock import ToolClock, wait_with_tool_clock
from lazybridge.engines.coding import ApprovalRequest, ask_approval, closing_iter

from .protocol import ClaudeSdkClient, ClaudeSdkOptions, ClaudeSdkResult, ClaudeSdkStreamEvent, McpTool

logger = logging.getLogger(__name__)

# The SDK passes a string system prompt on the CLI command line, and Windows
# caps a whole command line at 32,767 characters: past that, CreateProcess
# fails with WinError 206, which surfaces as ``FileNotFoundError`` and then as
# the SDK's misleading "Claude Code not found". Long prompts (LazyCommittee's
# world-document writer, ~30k+ characters) therefore go through
# ``--system-prompt-file``. The threshold leaves room for every other argument
# (MCP config, JSON schema, tool lists) on the same line.
SYSTEM_PROMPT_FILE_THRESHOLD = 8_000

_CLOSE_GRACE = 0.5
_KILL_TIMEOUT = 10.0


def _kill_tree(process: Any) -> None:
    """Force-stop the SDK child and, on Windows, its descendants."""
    pid = getattr(process, "pid", None)
    if pid is None or getattr(process, "returncode", None) is not None:
        return
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                timeout=_KILL_TIMEOUT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode:
                logger.warning("taskkill failed for Claude process pid=%s", pid)
                with contextlib.suppress(OSError):
                    process.kill()
        else:
            # The SDK does not launch the child in a separate process group.
            process.kill()
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("Failed to kill Claude process tree (pid=%s)", pid, exc_info=True)
        with contextlib.suppress(OSError):
            process.kill()


async def _bounded_query(
    prompt: Any,
    sdk_options: Any,
    initialize_timeout: float,
    idle_timeout: float | None,
    tool_clock: ToolClock | None = None,
    transport_factory: Callable[..., Any] | None = None,
    query_factory: Callable[[Any, Any, Any, float], AsyncIterator[Any]] | None = None,
) -> AsyncIterator[Any]:
    """Consume the SDK independently so cancellation cannot pin the caller in SDK close."""
    if transport_factory is None:
        from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

        transport_factory = SubprocessCLITransport
    if query_factory is None:
        query_factory = _query_with_explicit_initialize_timeout

    # query() normally constructs this itself. Supplying it explicitly lets us
    # retain the child handle even when cancellation is stuck in query.close().
    transport_options = sdk_options
    if getattr(sdk_options, "can_use_tool", None):
        transport_options = replace(sdk_options, permission_prompt_tool_name="stdio")
    transport = transport_factory(prompt=prompt, options=transport_options)
    messages = query_factory(prompt, sdk_options, transport, initialize_timeout)
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def consume() -> None:
        try:
            async for message in messages:
                await queue.put(("message", message))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(("error", exc))
        else:
            await queue.put(("end", None))

    task = asyncio.create_task(consume())
    try:
        while True:
            try:
                kind, value = await wait_with_tool_clock(queue.get(), idle_timeout, tool_clock, idle=True)
            except TimeoutError as exc:
                raise TimeoutError(f"Claude Code stream went idle for {idle_timeout}s without a message") from exc
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        if not task.done():
            # Run the cleanup in its own task: a second cancellation (e.g. the
            # overall request_timeout firing while an idle-timeout cleanup is
            # in its grace period) must not stop the kill or leave the consumer
            # unregistered. If we are cancelled while waiting, kill immediately.
            cleanup = asyncio.ensure_future(_cleanup_consumer(task, transport))
            _cleanup_tasks.add(cleanup)
            cleanup.add_done_callback(_cleanup_tasks.discard)
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                process = getattr(transport, "_process", None)
                if process is not None and getattr(process, "returncode", None) is None:
                    _kill_tree(process)
                _detach(task)
                abandoned_tasks.add(cleanup)
                raise
        if task.done():
            with contextlib.suppress(BaseException):
                task.result()


def _build_explicit_timeout_query(transport: Any, options: Any, initialize_timeout: float) -> Any:
    """Build the SDK's protocol query with an explicit initialize deadline.

    claude-agent-sdk 0.2.x has no public timeout argument. Its public
    ``query()`` reads the parent environment before constructing this class,
    so putting the variable in ``ClaudeAgentOptions.env`` affects only the CLI
    child. Keep the necessary private integration isolated here and fail
    clearly if a future SDK changes the constructor contract.
    """
    version = importlib.metadata.version("claude-agent-sdk")
    from claude_agent_sdk._internal.query import Query

    required = {
        "transport",
        "is_streaming_mode",
        "can_use_tool",
        "hooks",
        "sdk_mcp_servers",
        "initialize_timeout",
        "agents",
        "exclude_dynamic_sections",
        "skills",
    }
    if not version.startswith("0.2.") or not required <= set(inspect.signature(Query).parameters):
        raise RuntimeError(
            f"claude-agent-sdk {version} is incompatible with LazyBridge's explicit initialize-timeout adapter"
        )

    hooks = None
    if options.hooks:
        hooks = {
            event: [
                {
                    "matcher": getattr(matcher, "matcher", None),
                    "hooks": getattr(matcher, "hooks", []),
                    **({"timeout": matcher.timeout} if getattr(matcher, "timeout", None) is not None else {}),
                }
                for matcher in matchers
            ]
            for event, matchers in options.hooks.items()
        }
    sdk_mcp_servers = {
        name: config["instance"]
        for name, config in (options.mcp_servers or {}).items()
        if isinstance(config, dict) and config.get("type") == "sdk"
    }
    agents = (
        {
            name: {key: value for key, value in asdict(agent).items() if value is not None}
            for name, agent in options.agents.items()
        }
        if options.agents
        else None
    )
    system_prompt = options.system_prompt
    exclude_dynamic_sections = (
        system_prompt.get("exclude_dynamic_sections")
        if isinstance(system_prompt, dict)
        and system_prompt.get("type") == "preset"
        and isinstance(system_prompt.get("exclude_dynamic_sections"), bool)
        else None
    )
    return Query(
        transport=transport,
        is_streaming_mode=True,
        can_use_tool=options.can_use_tool,
        hooks=hooks,
        sdk_mcp_servers=sdk_mcp_servers,
        initialize_timeout=initialize_timeout,
        agents=agents,
        exclude_dynamic_sections=exclude_dynamic_sections,
        skills=options.skills,
    )


async def _query_with_explicit_initialize_timeout(
    prompt: Any, options: Any, transport: Any, initialize_timeout: float
) -> AsyncIterator[Any]:
    """Run the SDK protocol query while owning its private Query lifecycle."""
    from claude_agent_sdk._internal.message_parser import parse_message

    await transport.connect()
    query = _build_explicit_timeout_query(transport, options, initialize_timeout)
    try:
        await query.start()
        await query.initialize()
        if isinstance(prompt, str):
            user_message = {
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
            }
            await transport.write(json.dumps(user_message) + "\n")
            query.spawn_task(query.wait_for_result_and_end_input())
        else:
            query.spawn_task(query.stream_input(prompt))
        async for data in query.receive_messages():
            message = parse_message(data)
            if message is not None:
                yield message
    finally:
        await query.close()
        query.close_receive_stream()


#: Strong references to in-flight cleanup tasks (``abandoned_tasks`` is weak).
_cleanup_tasks: set[asyncio.Future[None]] = set()


def _retrieve_exception(future: asyncio.Future[Any]) -> None:
    """Mark a detached task's exception as retrieved so asyncio does not log it."""
    if not future.cancelled():
        future.exception()


def _detach(task: asyncio.Task[Any]) -> None:
    """Leave a stuck SDK consumer behind without blocking loop teardown."""
    if task.done():
        return
    logger.debug("Claude SDK cleanup did not finish after process termination")
    abandoned_tasks.add(task)
    # Deliberately detached; its pending destruction at loop shutdown is expected.
    task._log_destroy_pending = False  # type: ignore[attr-defined]
    task.add_done_callback(_retrieve_exception)


async def _cleanup_consumer(task: asyncio.Task[Any], transport: Any) -> None:
    """Cancel the consumer, kill the CLI if it does not stop, then detach it."""
    process = getattr(transport, "_process", None)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=_CLOSE_GRACE)
    process = process or getattr(transport, "_process", None)
    if process is not None and getattr(process, "returncode", None) is None:
        await asyncio.to_thread(_kill_tree, process)
    if not done:
        done, _ = await asyncio.wait({task}, timeout=_CLOSE_GRACE)
    if not done:
        _detach(task)


@contextmanager
def _system_prompt_via_file(options: ClaudeSdkOptions) -> Iterator[ClaudeSdkOptions]:
    """Yield options whose long string system prompt is read from a temp file."""
    prompt = options.system_prompt
    if not isinstance(prompt, str) or len(prompt) <= SYSTEM_PROMPT_FILE_THRESHOLD:
        yield options
        return
    fd, path = tempfile.mkstemp(prefix="lazybridge-system-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(prompt)
        yield replace(options, system_prompt=cast(Any, {"type": "file", "path": path}))
    finally:
        try:
            os.unlink(path)
        except OSError:
            logger.warning("could not remove system prompt file %s", path, exc_info=True)


def _log_cli_stderr(line: str) -> None:
    """Surface the underlying ``claude`` CLI subprocess's own stderr.

    Without this, ``options.stderr`` stays unset, the SDK transport leaves
    the child's stderr fd on ``None`` (inherit-from-parent) instead of
    piping it, and a non-zero exit only ever surfaces as the SDK's generic
    ``ProcessError("Command failed with exit code N", stderr="Check stderr
    output for details")`` -- found live: inheritance did not reliably carry
    the real message through on Windows (a redirected/piped parent stderr,
    e.g. under a background-launched service), so a real crash left nothing
    to diagnose beyond the exit code. Piping + logging here is the only way
    to actually see why the CLI died, at the cost of these lines going
    through Python logging instead of straight to the console.
    """
    logger.debug("claude CLI stderr: %s", line.rstrip())


_STDERR_TAIL_LINES = 20


def _stderr_detail(message: str, lines: deque[str]) -> str:
    if not lines:
        return message
    logger.warning("Claude CLI stderr tail on failure:\n%s", "\n".join(lines))
    return f"{message}\nClaude CLI stderr (last {len(lines)} lines):\n" + "\n".join(lines)


def _error_with_stderr(exc: Exception, lines: deque[str]) -> Exception:
    """Attach the bounded CLI stderr tail without hiding the original cause."""
    if not lines:
        return exc
    wrapped = RuntimeError(_stderr_detail(str(exc), lines))
    wrapped.__cause__ = exc
    return wrapped


async def _with_stderr_tail(source: AsyncIterator[Any], lines: deque[str]) -> AsyncIterator[Any]:
    """Preserve iterator cleanup while enriching any SDK/transport failure."""
    try:
        async with closing_iter(source) as messages:
            async for message in messages:
                yield message
    except Exception as exc:
        raise _error_with_stderr(exc, lines) from exc


def _capture_stderr(options: ClaudeSdkOptions) -> tuple[ClaudeSdkOptions, deque[str]]:
    """Install the shared bounded stderr collector used by run and stream."""
    lines: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)

    def stderr(line: str) -> None:
        clean = line.rstrip()
        lines.append(clean)
        _log_cli_stderr(clean)

    return replace(options, stderr=stderr), lines


if TYPE_CHECKING:  # pragma: no cover - typing only
    # Type-only: the SDK stays an optional runtime dependency (it is imported
    # lazily inside the methods below). Under ``ignore_missing_imports`` these
    # degrade to ``Any`` when the extra is absent, so a checkout without
    # ``lazybridge[claude-code]`` still type-checks — it just checks less.
    from claude_agent_sdk import HookContext, HookInput, HookJSONOutput
    from claude_agent_sdk import HookMatcher as SdkHookMatcher

#: Set once ``CanUseToolShadowedWarning`` has been added to the process-wide
#: warnings filter (see ``_sdk_options``). A plain flag, not a per-call
#: ``warnings.catch_warnings()`` context: that context manager mutates
#: global filter state without coroutine isolation, so two concurrent
#: ``AgentSdkClient`` calls (e.g. via ``asyncio.gather``) could clobber each
#: other's filters. Registering the ignore rule once, idempotently, avoids
#: that hazard entirely.
_shadow_warning_filtered = False


def _report_permission_denials(options: ClaudeSdkOptions, message: Any) -> None:
    """Surface CLI permission denials through the tool observer as errors.

    The CLI answers a denied tool call with an error tool result the model can
    paraphrase away, and the turn still ends successfully. The authoritative
    list is ``ResultMessage.permission_denials``; reporting it keeps a run that
    silently lost a tool from looking clean.
    """
    observer = options.tool_observer
    if observer is None:
        return
    for denial in getattr(message, "permission_denials", None) or ():
        data = denial if isinstance(denial, dict) else {}
        name = data.get("tool_name") or "unknown"
        try:
            observer(
                "error",
                {
                    "tool_name": name,
                    "error": f"Permission denied for tool {name!r} (permission_mode={options.permission_mode or 'default'})",
                    "arguments": data.get("tool_input") or {},
                    "tool_use_id": data.get("tool_use_id"),
                    "native": True,
                    "permission_denied": True,
                },
            )
        except Exception:  # observability must never fail the turn
            logger.warning("tool observer raised on permission denial", exc_info=True)


class ClaudeSdkRequestError(RuntimeError):
    """Raised when the Claude Agent SDK reports a failed run.

    Carries the CLI's ``api_error_status`` (HTTP status of the failing API
    call — e.g. 429/500/529 — when the CLI version reports one; ``None``
    otherwise) so ``ClaudeCodeEngine``'s retry classifier can distinguish a
    transient provider failure from a permanent one instead of guessing
    from the error string.
    """

    def __init__(self, message: str, *, status: int | None = None, session_id: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        #: The session the failed turn ran in, when the CLI reported one; a
        #: failed turn is still a resumable session.
        self.session_id = session_id


class AgentSdkClient(ClaudeSdkClient):
    """Run Claude Code through the installed Agent SDK.

    The SDK uses the same local Claude Code authentication as the CLI.  No
    credential is passed to, read by, or persisted by this adapter.

    The SDK's ``session_store`` and resume handling are not used by
    LazyBridge and are intentionally not wired here; the engine owns session
    continuity. ``transport_factory`` and ``query_factory`` are the supported
    injection seam for tests and alternate transports, so tests never need to
    patch SDK internals or launch the real CLI.
    """

    def __init__(
        self,
        *,
        transport_factory: Callable[..., Any] | None = None,
        query_factory: Callable[[Any, Any, Any, float], AsyncIterator[Any]] | None = None,
    ) -> None:
        self._transport_factory = transport_factory
        self._query_factory = query_factory

    @staticmethod
    def _message_model(message: Any, current: str | None) -> str | None:
        from claude_agent_sdk import AssistantMessage, SystemMessage

        # Assistant responses take precedence over init, including a fallback
        # model selected after the session was initialized. Ignore subagents.
        if isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
            model = message.model
            return model if isinstance(model, str) and model.strip() else current
        if current is None and isinstance(message, SystemMessage) and message.subtype == "init":
            init_model = message.data.get("model")
            return init_model if isinstance(init_model, str) and init_model.strip() else None
        return current

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
    def _report_session_id(options: ClaudeSdkOptions, message: Any) -> None:
        """Tell ``options.on_session_id`` which session a message belongs to.

        Result/assistant/stream messages carry ``session_id``; the ``init``
        ``SystemMessage`` (first message of a turn) carries it in ``data``. A
        misbehaving callback never affects the turn.
        """
        callback = options.on_session_id
        if callback is None:
            return
        sid = getattr(message, "session_id", None)
        if not sid:
            data = getattr(message, "data", None)
            sid = data.get("session_id") if isinstance(data, dict) else None
        if isinstance(sid, str) and sid:
            try:
                callback(sid)
            except Exception:
                logger.warning("on_session_id callback raised", exc_info=True)

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
        if options.permission_mode == "dontAsk":
            # ``dontAsk`` never consults ``can_use_tool``: anything not matched
            # by an allow rule is denied outright. ``builtin_tools`` only makes
            # a tool *available*, so without these rules every granted native
            # tool (WebSearch, Read, ...) would be silently denied. Tools that
            # need a human decision (anything beyond the read-only set) are
            # preapproved only when no approval gate is attached; ``file_roots`` still confine them.
            safe = {"Read", "Glob", "Grep", "WebSearch", "WebFetch"}
            narrowed = {entry.split("(", 1)[0] for entry in allowed if "(" in entry}
            for name in options.builtin_tools:
                if options.approval_gate is not None and name not in safe:
                    continue  # anything beyond the read-only set keeps going through the gate
                if name in allowed or name in narrowed:
                    continue  # never broaden a caller-supplied narrower rule
                allowed.append(name)
        if options.preapprove_application_tools:
            allowed.extend(f"mcp__{options.mcp_server_name}__{item.name}" for item in options.mcp_tools)

        roots = tuple(Path(root).resolve() for root in options.file_roots)

        def allowed_path(value: object) -> bool:
            if not isinstance(value, str) or not roots:
                return False
            path = Path(value)
            resolved = (Path(options.cwd) / path if not path.is_absolute() and options.cwd else path).resolve()
            return any(resolved.is_relative_to(root) for root in roots)

        async def confine_file_access(
            input_data: HookInput, tool_use_id: str | None, context: HookContext
        ) -> HookJSONOutput:
            """Enforce ``file_roots`` on every file access, always.

            This lives in a ``PreToolUse`` hook rather than in ``can_use_tool``
            because the SDK evaluates hooks *first* and consults the callback
            *last* — a tool auto-approved by an allow rule, by ``acceptEdits``
            or by ``bypassPermissions`` never reaches the callback at all
            (documented in the Agent SDK permission flow). Confinement that
            only ran there was therefore bypassable by configuration: any
            caller passing ``allowed_tools=["Read"]`` silently removed it.
            A hook deny holds even under ``bypassPermissions``.
            """
            if not roots:
                return {}
            # ``HookInput`` is the union of every hook event's payload, so the
            # value reads back as ``object``. This hook is registered under a
            # ``PreToolUse`` matcher only, where ``tool_input`` is always the
            # tool's argument mapping.
            arguments = cast("dict[str, Any]", input_data.get("tool_input") or {})
            path = arguments.get("file_path") or arguments.get("notebook_path") or arguments.get("path") or options.cwd
            if allowed_path(path):
                return {}
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"{path!r} is outside ClaudeCodeEngine.file_roots ({', '.join(str(root) for root in roots)})"
                    ),
                }
            }

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
                        name=name.removeprefix(native_tool_prefix),
                        arguments=arguments,
                        reason=getattr(context, "decision_reason", None),
                        cwd=options.cwd,
                        raw={
                            "provider_tool_name": name,
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
            input_data: HookInput, tool_use_id: str | None, context: HookContext
        ) -> HookJSONOutput:
            return {"continue_": True}

        native_tool_prefix = f"mcp__{options.mcp_server_name}__"
        observer_warning_types: set[type[BaseException]] = set()

        def observe_native(kind: str, payload: dict[str, Any]) -> None:
            """Notify observability without ever affecting a CLI tool call."""
            if options.tool_observer is None:
                return
            try:
                options.tool_observer(kind, payload)
            except Exception as exc:
                exception_type = type(exc)
                if exception_type not in observer_warning_types:
                    observer_warning_types.add(exception_type)
                    warnings.warn(
                        f"Claude Code native tool observer raised {exception_type.__name__}: {exc}. "
                        "Further exceptions of this type will not be re-warned.",
                        stacklevel=2,
                    )

        def is_native(input_data: HookInput) -> bool:
            return not str(input_data.get("tool_name", "")).startswith(native_tool_prefix)

        async def observe_native_call(
            input_data: HookInput, tool_use_id: str | None, context: HookContext
        ) -> HookJSONOutput:
            if is_native(input_data):
                arguments = cast("dict[str, Any]", input_data.get("tool_input") or {})
                observe_native(
                    "call",
                    {
                        "tool_name": input_data.get("tool_name"),
                        "arguments": dict(arguments),
                        "tool_use_id": tool_use_id or input_data.get("tool_use_id"),
                        "native": True,
                    },
                )
            return {}

        async def observe_native_result(
            input_data: HookInput, tool_use_id: str | None, context: HookContext
        ) -> HookJSONOutput:
            if is_native(input_data):
                observe_native(
                    "result",
                    {
                        "tool_name": input_data.get("tool_name"),
                        "result": input_data.get("tool_response"),
                        "tool_use_id": tool_use_id or input_data.get("tool_use_id"),
                        "native": True,
                    },
                )
            return {}

        async def observe_native_failure(
            input_data: HookInput, tool_use_id: str | None, context: HookContext
        ) -> HookJSONOutput:
            if is_native(input_data):
                observe_native(
                    "error",
                    {
                        "tool_name": input_data.get("tool_name"),
                        "error": input_data.get("error"),
                        "tool_use_id": tool_use_id or input_data.get("tool_use_id"),
                        "native": True,
                    },
                )
            return {}

        use_callback = (
            bool(options.builtin_tools) or options.approval_gate is not None or not options.preapprove_application_tools
        )
        matchers: list[SdkHookMatcher] = []
        if roots:
            # Scoped to the tools that touch files — including ones this
            # profile does not grant, so a future/settings-added Edit cannot
            # slip past the confinement it was never checked against.
            matchers.append(
                HookMatcher(
                    matcher="Read|Glob|Grep|Edit|Write|NotebookEdit",
                    hooks=[confine_file_access],
                )
            )
        if use_callback:
            matchers.append(HookMatcher(hooks=[keep_permission_stream_open]))
        sdk_hooks: Any = {"PreToolUse": matchers} if matchers else None
        if options.tool_observer is not None:
            matchers.append(HookMatcher(hooks=[observe_native_call]))
            sdk_hooks = {
                "PreToolUse": matchers,
                "PostToolUse": [HookMatcher(hooks=[observe_native_result])],
            }
            try:
                from claude_agent_sdk.types import HookEvent

                has_post_tool_use_failure = any(
                    event == "PostToolUseFailure" for literal in get_args(HookEvent) for event in get_args(literal)
                )
            except (ImportError, AttributeError):
                has_post_tool_use_failure = False
            if has_post_tool_use_failure:
                sdk_hooks["PostToolUseFailure"] = [HookMatcher(hooks=[observe_native_failure])]
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
            # Both hooks intentionally use the generic (dict, str | None, Any)
            # shape rather than the SDK's per-hook TypedDict union — they only
            # ever fire for PreToolUse.
            #
            # The confinement hook is registered whenever file_roots exist,
            # independently of ``use_callback``: it is the layer that survives
            # an allow rule or a permissive mode, so it must not be tied to
            # whether the callback is in play. It is scoped to the file-reading
            # tools by matcher, so other tools pay nothing for it.
            # Built inline so the key types as the SDK's ``HookEvent`` literal
            # rather than a plain ``str`` (dict key types are invariant, so a
            # pre-built ``dict[str, ...]`` would not satisfy the parameter).
            hooks=sdk_hooks,
            stderr=options.stderr or _log_cli_stderr,
            setting_sources=list(options.setting_sources),
            # The env var, not the settings file: it takes precedence over
            # every settings source, so it says the same thing whether or not
            # this agent was told to read the user's settings. The SDK merges
            # this dict OVER the inherited environment rather than replacing
            # it, so naming one variable does not strip the rest.
            env={
                **(
                    {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(options.auto_compact_window)}
                    if options.auto_compact_window is not None
                    else {}
                ),
            },
            include_partial_messages=options.include_partial_messages,
            output_format=options.output_format,
        )

    async def run(
        self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict[str, Any], ...] = ()
    ) -> ClaudeSdkResult:
        try:
            from claude_agent_sdk import ResultMessage
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Claude Agent SDK is not installed") from exc

        options, stderr_lines = _capture_stderr(options)
        final: Any | None = None
        model: str | None = None
        sdk_prompt: str | AsyncIterator[dict[str, Any]] = (
            self._prompt_stream(prompt, attachments) if self._needs_streaming_input(options, attachments) else prompt
        )
        with _system_prompt_via_file(options) as cli_options:
            sdk_options = self._sdk_options(cli_options)
            async with closing_iter(
                _with_stderr_tail(
                    _bounded_query(
                        sdk_prompt,
                        sdk_options,
                        options.initialize_timeout,
                        options.stream_idle_timeout,
                        options.tool_clock,
                        self._transport_factory,
                        self._query_factory,
                    ),
                    stderr_lines,
                )
            ) as messages:
                async for message in messages:
                    self._report_session_id(options, message)
                    model = self._message_model(message, model)
                    if isinstance(message, ResultMessage):
                        final = message
                        _report_permission_denials(options, message)
        if final is None:
            raise RuntimeError("Claude Agent SDK ended without a ResultMessage")
        if final.is_error:
            detail = "; ".join(final.errors or []) or final.result or final.subtype
            raise ClaudeSdkRequestError(
                _stderr_detail(f"Claude Agent SDK failed: {detail}", stderr_lines),
                status=final.api_error_status,
                session_id=final.session_id,
            )

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
            model=model,
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        )

    @staticmethod
    def _stream_options(options: ClaudeSdkOptions) -> ClaudeSdkOptions:
        """Enable partial events without dropping any per-run SDK option."""
        return replace(options, include_partial_messages=True)

    @staticmethod
    def _needs_streaming_input(options: ClaudeSdkOptions, attachments: tuple[dict[str, Any], ...]) -> bool:
        """Whether this run must send the SDK a streaming (async-iterator) prompt.

        The Agent SDK only supports ``can_use_tool``/permission negotiation and
        hook callbacks — including the native ``tool_observer`` hooks
        registered in ``_sdk_options`` — over streaming input; a plain string
        prompt fails before the model is even queried. ``builtin_tools`` and
        ``attachments`` already require the richer message shape for their own
        reasons, and ``tool_observer`` joins them here so the two can never
        drift apart again: whenever ``_sdk_options`` would register the
        observer hooks, this must say streaming input is needed too.
        """
        return bool(options.builtin_tools) or bool(attachments) or options.tool_observer is not None

    async def stream(
        self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict[str, Any], ...] = ()
    ) -> AsyncIterator[ClaudeSdkStreamEvent]:
        try:
            from claude_agent_sdk import ResultMessage, StreamEvent
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Claude Agent SDK is not installed") from exc

        options, stderr_lines = _capture_stderr(options)
        saw_text = False
        saw_result = False
        model: str | None = None
        stream_options = self._stream_options(options)
        sdk_prompt: str | AsyncIterator[dict[str, Any]] = (
            self._prompt_stream(prompt, attachments)
            if self._needs_streaming_input(stream_options, attachments)
            else prompt
        )
        with _system_prompt_via_file(stream_options) as cli_options:
            async with closing_iter(
                _with_stderr_tail(
                    _bounded_query(
                        sdk_prompt,
                        self._sdk_options(cli_options),
                        stream_options.initialize_timeout,
                        stream_options.stream_idle_timeout,
                        stream_options.tool_clock,
                        self._transport_factory,
                        self._query_factory,
                    ),
                    stderr_lines,
                )
            ) as messages:
                async for message in messages:
                    self._report_session_id(stream_options, message)
                    model = self._message_model(message, model)
                    if isinstance(message, StreamEvent):
                        event = message.event
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                                saw_text = True
                                yield ClaudeSdkStreamEvent(text=delta["text"])
                    elif isinstance(message, ResultMessage):
                        saw_result = True
                        _report_permission_denials(stream_options, message)
                        if message.is_error:
                            detail = "; ".join(message.errors or []) or message.result or message.subtype
                            raise ClaudeSdkRequestError(
                                _stderr_detail(f"Claude Agent SDK failed: {detail}", stderr_lines),
                                status=message.api_error_status,
                                session_id=message.session_id,
                            )
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
                            model=model,
                            cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
                            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
                        )
        if not saw_result:
            # Mirrors AgentSdkClient.run(): the SDK iterator ended without a
            # ResultMessage. Silently returning here would let the engine
            # record an empty/partial stream as a successful turn (stale
            # memory write, false AGENT_FINISH, and — in session_mode=
            # "runtime" — a resume ID that was never actually confirmed).
            raise RuntimeError("Claude Agent SDK stream ended without a ResultMessage")
