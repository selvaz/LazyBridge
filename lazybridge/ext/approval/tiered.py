"""TieredGate — a declarative, per-agent approval policy over tool calls.

The coding engines (:class:`~lazybridge.engines.claude_code.ClaudeCodeEngine`,
:class:`~lazybridge.engines.codex.CodexEngine`) consult one
:class:`~lazybridge.engines.coding.ApprovalGate` for every action that is not
pre-approved. This module turns that single callback into a **policy table**:
each agent declares, per tool (and optionally per argument pattern), which of
four tiers governs it:

- ``allow``   — runs without asking (the agent's job, e.g. reads, web);
- ``session`` — the FIRST call asks a human; approval sticks for the rest of
  the process, scoped to the exact tool + cwd + policy that was approved;
- ``ask``     — every call asks a human (irreversible/outward actions);
- ``deny``    — never runs, no human can approve it here (wrong tool for the
  agent; use a propose-only flow instead — see the LazyPortfolio advisor
  pattern for LLM output that must mutate state).

First matching rule wins; anything unmatched is denied — a gate that
fail-opens on the rule you forgot to write is not a gate.

Matching: ``name_pattern`` is an ``fnmatch`` pattern over the tool name
(``Write``, ``Bash``, ``web_search``, …). ``arg_pattern``, when given, is an
``fnmatch`` pattern applied to the *command-like* argument (``command`` for
Bash, else the JSON of all arguments) — this is what lets one rule say
"``git push`` asks, plain ``git diff`` runs".

Compound Bash commands (``a && b``, ``a; b``, pipes) are split on shell
separators and every segment is matched independently; the segment with the
MOST SEVERE tier decides the whole call. **This is a name/pattern heuristic
over command text, not a sandbox.** It closes the specific hole of
``git add x && git commit`` riding the blander tier of its first segment; it
does not parse shell syntax, does not see quoting/subshells/`$( )`
substitution, and cannot confine anything to a directory. Path confinement is
a *separate* concern owned by the engine's own ``file_roots`` hook (matches
Read/Glob/Grep/Edit/Write/NotebookEdit — see the ``extra_tools`` docstring on
``ClaudeCodePolicy`` in :mod:`lazybridge.engines.coding`): ``Bash`` in
particular is NOT path-confinable that way, because an approved command can
touch any path its process can. Do not present this splitter as a security
boundary for Bash — it only decides which tier's human-approval rule applies.

Promoted from the ``approval-lab`` prototype (live-verified against a real
Claude Code "Code Assistant" agent). Three gaps found during that prototype
phase are fixed here, not carried forward:

1. **Session-grant scoping.** The prototype indexed a ``session`` grant by
   ``(kind, name)`` alone, so approving ``Write`` once in one ``cwd`` under
   one policy silently reused that approval for a different ``cwd`` or a
   different (looser) policy sharing the same gate. Grants are now scoped to
   ``(provider, kind, name, canonical cwd, policy fingerprint)``.
2. **Structured audit log.** Every decision — including who answered a human
   prompt and how — is appended to :attr:`TieredGate.log` as an
   :class:`AuditRecord`, not just collected as a loose ``dict`` for prototype
   inspection.
3. **Default deny, confirmed not assumed.** Unmatched requests were already
   denied in the prototype; :mod:`tests.unit.ext.approval.test_tiered` pins
   that behaviour with a regression test rather than taking it on faith.

**Known limitation, found in review, not yet closed: the engine's own outer
cache can widen a ``session`` grant past what this module scopes it to.**
``ClaudeCodeEngine``/``CodexEngine`` wrap *any* configured ``approval_gate``
with :func:`lazybridge.engines.coding.remembering_gate`, whose own cache key
is ``(request.kind, request.name)`` — no cwd, no policy fingerprint, no agent
identity. That wrapper's cache is checked *before* the request ever reaches
:class:`TieredGate`, so once it catches a hit, neither this gate's finer
scoping nor its audit log sees the call. Two things bound the practical
blast radius today, but neither is a fix:

- The outer cache is per-``Session``+agent (:func:`session_approvals`); with
  no ``Session`` attached (the common case for a short-lived, one-shot
  worker), it starts empty on every ``run()`` call and only persists for the
  *rest of that one run* — so it cannot leak a grant across separate agent
  invocations, only across calls *within* one run.
- Within one run, a caller whose cwd and rule never change for a given tool
  name (e.g. one worktree per dispatch, one ``session`` rule for ``Write``)
  sees no *scope* widening in practice, because there was only one scope to
  begin with — the real cost is an **audit gap**: the 2nd+ approved call in
  that run is allowed by the outer cache and never appended to
  :attr:`TieredGate.log`.
- A caller that DOES vary cwd, policy, or agent identity within one run (or
  shares a ``Session`` across agents) gets the scope-widening bug this
  module's docstring otherwise claims to have fixed. Prefer the ``ask`` tier
  over ``session`` for any rule where per-call audit completeness matters
  more than avoiding repeat prompts, until the outer wrapper is made scope-
  aware.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from typing import Any, Literal, Protocol

from lazybridge.engines.coding import ApprovalDecision, ApprovalRequest

Tier = Literal["allow", "session", "ask", "deny"]

_SEVERITY = {"allow": 0, "session": 1, "ask": 2, "deny": 3}


class Channel(Protocol):
    """A human-facing yes/no prompt. Implementations: terminal, chat, queue.

    Only the ``ask`` coroutine is part of the structural contract. A channel
    MAY additionally expose a ``name: str`` attribute identifying itself
    (e.g. ``"terminal"``, ``"telegram"``) — :class:`TieredGate` reads it via
    ``getattr`` for the audit trail and falls back to the class name when
    absent, so plain implementations need not declare it.
    """

    async def ask(self, prompt: str) -> bool: ...


class TerminalChannel:
    """Ask on stdin. ``y``/``yes``/``si``/``s``/``ok`` approve; anything else denies.

    Generic and dependency-free — useful for local development and for
    exercising :class:`TieredGate` in tests without a real chat backend.
    Production agents that run unattended (scheduled jobs, always-on
    processes) need a different channel; this one blocks on a human being at
    the terminal.
    """

    name = "terminal"

    async def ask(self, prompt: str) -> bool:
        answer = await asyncio.to_thread(input, f"{prompt}\n  approve? [y/N] ")
        return answer.strip().lower() in {"y", "yes", "si", "s", "ok"}


@dataclass(frozen=True)
class Rule:
    tier: Tier
    name_pattern: str
    arg_pattern: str | None = None

    def matches(self, request: ApprovalRequest) -> bool:
        if not fnmatch(request.name, self.name_pattern):
            return False
        if self.arg_pattern is None:
            return True
        command = request.arguments.get("command")
        haystack = command if isinstance(command, str) else json.dumps(dict(request.arguments), default=str)
        return fnmatch(haystack, self.arg_pattern)

    def fingerprint(self) -> str:
        """Short stable hash of this rule's shape, used to scope session grants.

        Two rules that look identical fingerprint identically even if they
        are different ``Rule`` instances (e.g. rebuilt on every process
        start) — the fingerprint is over content, not identity.
        """
        payload = f"{self.tier}|{self.name_pattern}|{self.arg_pattern or ''}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AuditRecord:
    """One structured audit entry — every decision the gate makes, allowed or not.

    ``arguments_hash``/``arguments_preview`` stand in for the raw arguments so
    the audit log stays safe to retain and to display even when a call
    carries large payloads or sensitive-looking values (file contents,
    tokens pasted into a command). The hash is over the exact JSON the rule
    matched against, so two records with equal hashes really did see equal
    arguments.
    """

    timestamp: float
    provider: str
    kind: str
    tool_name: str
    cwd: str | None
    tier: str
    action: str
    message: str
    arguments_hash: str
    arguments_preview: str
    #: Identity of the channel that answered a human prompt, if the decision
    #: went through one (``session``/``ask`` tiers only); ``None`` for
    #: ``allow``/``deny``/unmatched, which never consult a human.
    responder: str | None = None


def _canonical_cwd(cwd: str | None) -> str:
    """Normalize a cwd for grant scoping: resolve symlinks/relatives, fold case.

    Two different-looking paths that name the same directory (``.`` vs its
    absolute form, or Windows' case-insensitive filesystem) must scope to the
    same grant; two genuinely different directories must not. Falls back to
    ``abspath`` when the path doesn't exist yet (``realpath`` degrades to that
    anyway) so a not-yet-created directory still scopes consistently.
    """
    if not cwd:
        return ""
    try:
        return os.path.normcase(os.path.realpath(cwd))
    except OSError:  # pragma: no cover - defensive; realpath rarely raises
        return os.path.normcase(os.path.abspath(cwd))


def _hash_arguments(arguments: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(arguments), default=str, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


#: Best-effort masks for common secret shapes that show up in tool arguments
#: (an ``Authorization`` header, a bearer token, a ``key=value`` credential
#: pasted into a Bash command). This is a heuristic, not a guarantee — like
#: the compound-command splitter above, it closes the common, expensive case
#: (a token landing verbatim in a durable audit log) without claiming to
#: catch every way a secret can be shaped.
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer)\s+\S+"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
)


def _redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)} [redacted]", text)
    return text


def _preview_arguments(arguments: Mapping[str, Any], limit: int = 200) -> str:
    payload = _redact(json.dumps(dict(arguments), default=str, sort_keys=True))
    return payload if len(payload) <= limit else payload[: limit - 1] + "…"


def _channel_identity(channel: Channel) -> str:
    return getattr(channel, "name", None) or type(channel).__name__


@dataclass
class TieredGate:
    """ApprovalGate implementation driven by an ordered rule table.

    Implements :class:`~lazybridge.engines.coding.ApprovalGate` directly
    (structural typing — ``async def __call__(self, request) -> ApprovalDecision``
    is the whole contract): no bridge class re-declares the signature.
    """

    channel: Channel
    rules: tuple[Rule, ...]
    #: Callback invoked with every :class:`AuditRecord` as it is produced, in
    #: addition to appending it to :attr:`log`. Wire this to persist the
    #: audit trail somewhere durable (a file, a logger, a Store) — the
    #: in-memory list alone does not survive process restart.
    on_record: Any = None
    #: ``(provider, kind, name, canonical cwd, policy fingerprint)`` tuples a
    #: human already approved at the ``session`` tier. Kept here (not only in
    #: the engine's remembering wrapper) so the gate behaves the same with or
    #: without a LazyBridge ``Session`` attached.
    _session_grants: set[tuple[str, str, str, str, str]] = field(default_factory=set, repr=False)
    #: Structured audit trail of every decision this gate has made.
    log: list[AuditRecord] = field(default_factory=list, repr=False)

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        rule = self._match(request)
        decision, responder = await self._decide(request, rule)
        record = AuditRecord(
            timestamp=time.time(),
            provider=request.provider,
            kind=request.kind,
            tool_name=request.name,
            cwd=request.cwd,
            tier=rule.tier if rule else "unmatched",
            action=decision.action,
            message=decision.message,
            arguments_hash=_hash_arguments(request.arguments),
            arguments_preview=_preview_arguments(request.arguments),
            responder=responder,
        )
        self.log.append(record)
        if self.on_record is not None:
            self.on_record(record)
        return decision

    def _match(self, request: ApprovalRequest) -> Rule | None:
        """First matching rule — with compound shell commands handled honestly.

        ``git add x && git commit`` must NOT ride the tier of its first
        segment: the command is split on shell separators and every segment
        is matched on its own; the segment with the MOST SEVERE tier decides
        the whole call (deny > ask > session > allow). A segment matching no
        rule makes the whole call unmatched (default deny). Found live: the
        first Code Assistant run slipped a `git commit` past the per-call
        tier inside an `&&` chain.

        This is a text-pattern heuristic, not a shell parser — see the module
        docstring for what it does and does not guard against.
        """
        command = request.arguments.get("command")
        if not isinstance(command, str):
            return next((r for r in self.rules if r.matches(request)), None)
        segments = [s.strip() for s in re.split(r"&&|\|\||[;|\n]", command) if s.strip()]
        worst: Rule | None = None
        for segment in segments or [command]:
            probe = replace(request, arguments={**dict(request.arguments), "command": segment})
            rule = next((r for r in self.rules if r.matches(probe)), None)
            if rule is None:
                return None  # one unmatched segment sinks the whole call
            if worst is None or _SEVERITY[rule.tier] > _SEVERITY[worst.tier]:
                worst = rule
        return worst

    def _grant_key(self, request: ApprovalRequest, rule: Rule) -> tuple[str, str, str, str, str]:
        """Scope for a ``session`` grant: provider + kind + name + cwd + policy.

        Narrower than the prototype's ``(kind, name)`` on purpose: a grant
        earned for one tool in one working directory under one policy must
        not silently cover the same tool name in a different cwd (a
        different sandbox/repo) or under a different policy fingerprint (the
        rule that was actually shown to the human when they approved).
        """
        return (
            request.provider,
            request.kind,
            request.name,
            _canonical_cwd(request.cwd),
            rule.fingerprint(),
        )

    async def _decide(self, request: ApprovalRequest, rule: Rule | None) -> tuple[ApprovalDecision, str | None]:
        if rule is None:
            return (
                ApprovalDecision.deny(
                    f"'{request.name}' matches no rule in this agent's approval policy (default deny)"
                ),
                None,
            )
        if rule.tier == "allow":
            return ApprovalDecision.allow(), None
        if rule.tier == "deny":
            return ApprovalDecision.deny(f"'{request.name}' is barred for this agent by policy"), None
        key = self._grant_key(request, rule)
        if rule.tier == "session" and key in self._session_grants:
            return ApprovalDecision.allow(), None
        responder = _channel_identity(self.channel)
        approved = await self.channel.ask(_render(request, rule.tier))
        if not approved:
            return ApprovalDecision.deny(f"'{request.name}' denied by the human approver"), responder
        if rule.tier == "session":
            self._session_grants.add(key)
            return ApprovalDecision.allow_for_session(), responder
        return ApprovalDecision.allow(), responder


def _render(request: ApprovalRequest, tier: Tier) -> str:
    args = _redact(json.dumps(dict(request.arguments), default=str))
    if len(args) > 400:
        args = args[:400] + "…"
    once = " (approving grants it for this cwd/policy for the rest of the session)" if tier == "session" else ""
    return (
        f"[TieredGate] agent asks to run {request.kind} '{request.name}'{once}\n"
        f"  arguments: {args}\n"
        f"  cwd: {request.cwd or '-'}"
    )


def run_gate_sync(gate: TieredGate, request: ApprovalRequest) -> ApprovalDecision:
    """Convenience for tests/demos outside an event loop."""
    return asyncio.run(gate(request))


__all__ = [
    "AuditRecord",
    "Channel",
    "Rule",
    "TerminalChannel",
    "Tier",
    "TieredGate",
    "run_gate_sync",
]
