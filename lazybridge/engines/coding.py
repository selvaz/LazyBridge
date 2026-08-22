"""Shared permission and approval configuration for coding engines.

The provider runtimes keep enforcing their native sandbox and permission
models.  This module supplies one application-facing approval contract so a
CLI, web UI, queue, or chat integration can answer requests from either
Claude Code or Codex without provider-specific branching.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from weakref import WeakKeyDictionary

ApprovalAction = Literal["allow", "allow_session", "deny", "cancel"]


@dataclass(frozen=True)
class ApprovalRequest:
    """A normalized request emitted before a coding agent performs an action."""

    provider: Literal["claude-code", "codex"]
    kind: Literal["tool", "command", "file_change", "permissions", "user_input"]
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None
    cwd: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ApprovalDecision:
    """Provider-neutral approval result returned by an :class:`ApprovalGate`."""

    action: ApprovalAction
    message: str = ""
    # Used by Codex's request_permissions tool.  A gate may grant a subset of
    # the requested permissions instead of accepting the complete request.
    permissions: Mapping[str, Any] | None = None

    @classmethod
    def allow(cls) -> ApprovalDecision:
        return cls("allow")

    @classmethod
    def allow_for_session(cls) -> ApprovalDecision:
        return cls("allow_session")

    @classmethod
    def deny(cls, message: str = "Action denied by approval gate") -> ApprovalDecision:
        return cls("deny", message)

    @classmethod
    def cancel(cls, message: str = "Action cancelled by approval gate") -> ApprovalDecision:
        return cls("cancel", message)


class ApprovalGate(Protocol):
    """Interface implemented by approval UIs and policy services."""

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision | Awaitable[ApprovalDecision]: ...


ApprovalCallback = Callable[[ApprovalRequest], ApprovalDecision | Awaitable[ApprovalDecision]]


async def ask_approval(gate: ApprovalGate | None, request: ApprovalRequest) -> ApprovalDecision:
    """Invoke ``gate`` and fail closed when no gate is configured."""

    if gate is None:
        return ApprovalDecision.deny("No approval gate is configured")
    result = gate(request)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, ApprovalDecision):
        raise TypeError(f"approval gate must return ApprovalDecision, got {type(result).__name__}")
    return result


#: Per-event-loop lock registries, keyed weakly so a finished loop's locks are
#: collected with it. An ``asyncio.Lock`` binds to the loop that first waits on
#: it and raises ``RuntimeError: bound to a different event loop`` anywhere
#: else — and every synchronous ``Agent.__call__`` runs on a *fresh* loop, so a
#: process-wide lock cache would break on the second such call.
_LOOP_LOCKS: MutableMapping[Any, dict[str, asyncio.Lock]] = WeakKeyDictionary()


def loop_scoped_lock(key: str) -> asyncio.Lock:
    """A lock shared by everything using ``key`` **on the running loop**.

    Serialisation across loops (or processes) is not offered and cannot be:
    two event loops are two independent schedulers. Callers that need it must
    coordinate outside the process.
    """
    loop = asyncio.get_running_loop()
    return _LOOP_LOCKS.setdefault(loop, {}).setdefault(key, asyncio.Lock())


#: Attribute used to park approval caches on a LazyBridge ``Session``.
_APPROVALS_SLOT = "_lazybridge_tool_approvals"


def session_approvals(session: Any, provider: str, agent_name: str) -> set[tuple[str, str]]:
    """Return the ``allow_session`` cache for one agent inside one Session.

    ``allow_session`` has to mean *session*, not *turn*: the cache is keyed by
    ``(provider, agent_name)`` and parked on the ``Session`` object, so a
    grant survives across runs of that agent while a second agent sharing the
    same engine instance — engines are shared freely, which is why
    ``resolve_agent_name`` exists — still gets asked separately.

    Without a ``Session`` there is nowhere to persist the grant, so a fresh
    per-run set is returned and ``allow_session`` degrades to "for the rest of
    this run".
    """
    if session is None:
        return set()
    state = getattr(session, _APPROVALS_SLOT, None)
    if state is None:
        state = {}
        try:
            setattr(session, _APPROVALS_SLOT, state)
        except AttributeError:  # pragma: no cover - exotic Session stand-ins
            return set()
    cache: set[tuple[str, str]] = state.setdefault((provider, agent_name), set())
    return cache


def remembering_gate(gate: ApprovalGate | None, approved: set[tuple[str, str]]) -> ApprovalGate:
    """Wrap ``gate`` so ``allow_session`` grants are not asked again.

    Applies to every request kind — tool, command, file_change, permissions —
    so one wrapper covers both the LazyBridge-side dynamic tool gate and the
    provider-side approval requests arriving over the wire.
    """

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        key = (request.kind, request.name)
        if key in approved:
            return ApprovalDecision.allow_for_session()
        decision = await ask_approval(gate, request)
        if decision.action == "allow_session":
            approved.add(key)
        return decision

    return ask


class TerminalApprovalGate:
    """Small interactive gate suitable for local development.

    ``input`` runs in a worker thread so it does not block the agent's asyncio
    loop. Production applications should normally pass their own callback
    backed by a web UI, message queue, or chat connector.
    """

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        summary = request.reason or f"{request.kind}: {request.name}"
        if request.arguments:
            summary += f"\nArguments: {dict(request.arguments)!r}"
        answer = (
            (
                await asyncio.to_thread(
                    input,
                    f"\n[{request.provider}] {summary}\nApprove? [y]es / [s]ession / [n]o / [c]ancel: ",
                )
            )
            .strip()
            .lower()
        )
        if answer in {"y", "yes"}:
            return ApprovalDecision.allow()
        if answer in {"s", "session"}:
            return ApprovalDecision.allow_for_session()
        if answer in {"c", "cancel"}:
            return ApprovalDecision.cancel()
        return ApprovalDecision.deny()


@dataclass(frozen=True)
class ClaudeCodePolicy:
    """Claude Agent SDK-specific controls."""

    #: ``None`` lets the engine pick per run: ``"default"`` when something
    #: actually needs gating (built-in tools, an approval gate, un-preapproved
    #: application tools) and ``"dontAsk"`` when nothing does. Pinning a value
    #: here overrides that choice — note that a hardcoded ``"default"`` would
    #: put a fully pre-approved, tool-only agent into prompting mode with no
    #: callback able to answer.
    permission_mode: Literal["default", "dontAsk", "acceptEdits", "bypassPermissions", "plan", "auto"] | None = None
    preapprove_application_tools: bool = True
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    setting_sources: tuple[Literal["user", "project", "local"], ...] = ()
    #: Extra built-in tool names ADDED to the engine's derived set (Read/Glob/
    #: Grep from ``file_roots``, WebSearch/WebFetch from ``web=``). This is
    #: what lets a gated agent be granted ``Write``/``Edit``/``Bash``: the
    #: SDK's ``tools=`` option controls which built-ins the model can call at
    #: all, and the engine used to hardcode the read-only set — so no approval
    #: gate could ever be *asked* about a write, because the model never had
    #: the tool. Granting a name here does not pre-approve it: unless it is
    #: also in ``allowed_tools``, every call still routes through
    #: ``can_use_tool`` (the approval gate, or the fail-closed default).
    #:
    #: Confinement caveat — ``file_roots`` is enforced by a hook that matches
    #: the FILE tools (Read/Glob/Grep/Edit/Write/NotebookEdit). ``Bash`` is
    #: NOT path-confinable that way: an approved command can touch any path
    #: its process can. The shell's only boundary is the approval gate's
    #: policy, so the engine REFUSES to grant ``Bash`` (or any name outside
    #: the hook-confined set and the web pair) unless an ``approval_gate`` is
    #: configured — fail closed at construction, not at the first escape.
    extra_tools: tuple[str, ...] = ()
    #: How much context this agent may fill before Claude Code compacts it,
    #: in tokens. ``None`` leaves the CLI on its own tuned default.
    #:
    #: Two things to know before setting it. The number is a WINDOW, not a
    #: trigger: Claude Code compacts when usage approaches it, and the
    #: effective threshold is the minimum of this value and the model's real
    #: context window — so it can bring compaction forward, never push it
    #: past what the model allows. And an agent does not otherwise inherit
    #: this from your own ``settings.json`` at all, because
    #: ``setting_sources`` is empty by default; this is the per-agent way to
    #: say it, without borrowing the rest of a human's personal settings.
    auto_compact_window: int | None = None


@dataclass(frozen=True)
class CodexPolicy:
    """Codex App Server-specific controls."""

    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    approval_policy: Literal["untrusted", "on-request", "never"] = "never"
    preapprove_dynamic_tools: bool = True
    #: Token count at which Codex starts compacting this agent's history.
    #: ``None`` leaves the CLI's own default. Forwarded as the App Server's
    #: ``-c model_auto_compact_token_limit=<n>`` override, so it applies to
    #: this agent's subprocess and to nothing else.
    #:
    #: Its companion ``model_context_window`` is deliberately NOT exposed:
    #: setting it is reported upstream to break auto-compaction outright
    #: (openai/codex#16068), and it describes the budget rather than
    #: enlarging the model's real limit. Leave the window to Codex and move
    #: only the point at which it summarises.
    auto_compact_token_limit: int | None = None


@dataclass(frozen=True)
class CodingAgentConfig:
    """Complete provider configuration plus a shared approval gate."""

    claude: ClaudeCodePolicy = field(default_factory=ClaudeCodePolicy)
    codex: CodexPolicy = field(default_factory=CodexPolicy)
    approval_gate: ApprovalGate | None = None

    @classmethod
    def reviewer(cls) -> CodingAgentConfig:
        """Read-only, non-interactive profile for code review agents."""

        return cls(
            claude=ClaudeCodePolicy(preapprove_application_tools=False),
            codex=CodexPolicy(preapprove_dynamic_tools=False),
        )

    @classmethod
    def writer(cls, approval_gate: ApprovalGate) -> CodingAgentConfig:
        """Workspace writer profile: mutations require the supplied gate."""

        return cls(
            claude=ClaudeCodePolicy(
                permission_mode="default",
                preapprove_application_tools=False,
            ),
            codex=CodexPolicy(
                sandbox="workspace-write",
                approval_policy="on-request",
                preapprove_dynamic_tools=False,
            ),
            approval_gate=approval_gate,
        )


__all__ = [
    "ApprovalAction",
    "ApprovalCallback",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalRequest",
    "ClaudeCodePolicy",
    "CodexPolicy",
    "CodingAgentConfig",
    "TerminalApprovalGate",
    "remembering_gate",
    "session_approvals",
]
