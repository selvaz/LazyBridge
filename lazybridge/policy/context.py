"""Invocation-local policy propagation for delegated work."""

from __future__ import annotations

import secrets
import uuid
from collections import Counter
from collections.abc import Collection, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock

from ._review_selection import ReviewIndependence, ReviewMode, ReviewSelection, select_reviewer
from .models import EffectivePolicy, PolicyError, PolicyWidening, TaskGrant
from .resolve import (
    _intersect_tool_policy,
    _read_only_review_policy,
    _with_policy_digest,
    _without_approvals,
    child_within_parent,
    validate_child_policy,
)


class PolicyBudgetExceeded(PolicyError):
    """Raised when admitting work would exceed a tree-wide bound."""

    def __init__(self, bound: str, limit: int) -> None:
        self.bound = bound
        self.limit = limit
        super().__init__(f"tree policy budget exceeded: {bound}={limit}")


@dataclass(frozen=True)
class _ReviewReservation:
    writer_engine: str
    selection: ReviewSelection
    agent_id: str | None
    session: object | None

    def shares_identity(self, agent_id: str | None, session: object | None) -> bool:
        return (agent_id is not None and agent_id == self.agent_id) or (session is not None and session is self.session)


@dataclass
class TreeBudget:
    """Shared, concurrency-safe admission budget for an invocation tree.

    Child identities are registered for the lifetime of the budget, so a
    later resume of the same child reuses its total/work-unit slot.
    Review reservations use the configured review mode and engine availability.
    """

    max_depth: int | None = None
    max_active_children: int | None = None
    max_total_children: int | None = None
    max_work_units: int | None = None
    review_mode: ReviewMode = "opposite"
    available_engines: Collection[str] = ("claude", "codex")
    _children: set[object] = field(default_factory=set, init=False, repr=False)
    _active: Counter[object] = field(default_factory=Counter, init=False, repr=False)
    _work_units: int = field(default=0, init=False, repr=False)
    _review_reservations: dict[object, _ReviewReservation] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("max_depth", "max_active_children", "max_total_children", "max_work_units"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        self.available_engines = frozenset(self.available_engines)

    def admit(
        self,
        child_id: object,
        *,
        depth: int,
        reserve_review: bool = False,
        work_units: int = 1,
        role: str | None = None,
        engine: str | None = None,
        fresh: bool = False,
        agent_id: str | None = None,
        session: object | None = None,
    ) -> ReviewIndependence | None:
        """Admit a child and return the independence of any claimed review slot.

        For a same-provider reviewer, ``fresh=True`` attests that the reviewer
        has a new engine and no shared or resumed writer state.
        """
        if work_units < 0:
            raise ValueError("work_units must be non-negative")
        if role not in (None, "reviewer"):
            raise ValueError("role must be reviewer or None")
        if reserve_review and role is not None:
            raise ValueError("a reviewer cannot reserve a review")
        if (reserve_review or role == "reviewer") and engine not in ("claude", "codex"):
            raise PolicyError("review reservations require a claude or codex engine")
        with self._lock:
            is_new = child_id not in self._children
            becomes_active = not self._active.get(child_id, 0)
            active_limit = self.max_active_children
            if self.max_depth is not None and depth > self.max_depth:
                raise PolicyBudgetExceeded("max_depth", self.max_depth)
            claim = None
            new_reservation = None
            if reserve_review and child_id not in self._review_reservations:
                assert engine is not None
                selection = select_reviewer(engine, self.available_engines, self.review_mode)
                new_reservation = _ReviewReservation(engine, selection, agent_id, session)
            if role == "reviewer" and becomes_active:
                claim = next(
                    (
                        (writer, pending.selection.independence)
                        for writer, pending in self._review_reservations.items()
                        if pending.selection.engine == engine
                        and writer != child_id
                        and not pending.shares_identity(agent_id, session)
                        and (pending.selection.independence == "opposite-engine" or (fresh is True and is_new))
                    ),
                    None,
                )
                if claim is None and any(
                    writer == child_id or pending.shares_identity(agent_id, session)
                    for writer, pending in self._review_reservations.items()
                ):
                    raise PolicyError("reviewer must use a different agent and session from the writer")
            if active_limit is not None and (becomes_active or new_reservation):
                occupied = len(self._active) + len(self._review_reservations) + int(new_reservation is not None)
                if claim is not None:
                    occupied -= 1
                if occupied + int(becomes_active) > active_limit:
                    raise PolicyBudgetExceeded("max_active_children", active_limit)
            if (
                role == "reviewer"
                and claim is None
                and self.review_mode == "opposite-or-fresh"
                and any(pending.writer_engine == engine for pending in self._review_reservations.values())
            ):
                raise PolicyError("same-engine review requires a fresh reviewer and a single available engine")
            if is_new and self.max_total_children is not None and len(self._children) >= self.max_total_children:
                raise PolicyBudgetExceeded("max_total_children", self.max_total_children)
            if is_new and self.max_work_units is not None and self._work_units + work_units > self.max_work_units:
                raise PolicyBudgetExceeded("max_work_units", self.max_work_units)
            if is_new:
                self._children.add(child_id)
                self._work_units += work_units
            if claim is not None:
                del self._review_reservations[claim[0]]
            if new_reservation is not None:
                self._review_reservations[child_id] = new_reservation
            self._active[child_id] += 1
            return claim[1] if claim is not None else None

    def release(self, child_id: object) -> None:
        """Release one active use while retaining its resumable slot."""
        with self._lock:
            self._active[child_id] -= 1
            if self._active[child_id] <= 0:
                del self._active[child_id]


class _InvocationCredential:
    # Keep the secret out of dataclass fields and public invocation snapshots.
    __slots__ = ("_approval_credential",)
    _approval_credential: str | None


@dataclass(frozen=True)
class PolicyContext(_InvocationCredential):
    """Authority carried by one invocation into all nested work."""

    effective_policy: EffectivePolicy | None
    delegable_grants: EffectivePolicy | None
    invocation_id: str
    parent_invocation_id: str | None = None
    tree_budget: TreeBudget | None = None
    depth: int = 0
    review_independence: ReviewIndependence | None = None
    parent: PolicyContext | None = field(default=None, repr=False, compare=False)
    approval_actor: str | None = None
    agent_id: str | None = None

    @property
    def actor(self) -> str:
        """Display identity for request metadata and audit records."""
        return self.approval_actor or self.invocation_id


_current: ContextVar[PolicyContext | None] = ContextVar("lazybridge_policy_context", default=None)
_in_invocation: ContextVar[bool] = ContextVar("lazybridge_in_invocation", default=False)
_delegated_agent: ContextVar[object | None] = ContextVar("lazybridge_delegated_agent", default=None)


def current_policy_context() -> PolicyContext | None:
    """Return the active invocation policy, if policy enforcement is enabled."""

    return _current.get()


def current_approver_credential() -> str | None:
    """Return only the active invocation's credential for a registry call."""
    context = _current.get()
    return getattr(context, "_approval_credential", None)


def _require_operator_context() -> None:
    if _in_invocation.get():
        raise PermissionError("operation requires the host outside an agent invocation")


@contextmanager
def _agent_tool_call(agent: object) -> Iterator[None]:
    token = _delegated_agent.set(agent)
    try:
        yield
    finally:
        _delegated_agent.reset(token)


def review_engine(engine: object) -> str | None:
    """Identify supported coding engines from the actual engine instance."""
    from lazybridge.engines.claude_code.engine import ClaudeCodeEngine
    from lazybridge.engines.codex.engine import CodexEngine

    if isinstance(engine, ClaudeCodeEngine):
        return "claude"
    if isinstance(engine, CodexEngine):
        return "codex"
    return None


def resolve_invocation_policy(
    configured: EffectivePolicy | TaskGrant | None,
    *,
    tool_call: bool = False,
) -> tuple[EffectivePolicy | None, PolicyContext | None]:
    """Resolve an agent's configured authority against the immediate caller."""

    parent = _current.get()
    if configured is None:
        inherited = parent.effective_policy if parent is not None else None
        return (_without_approvals(inherited) if inherited is not None else None), parent
    if (
        tool_call
        and parent is not None
        and parent.delegable_grants is not None
        and isinstance(configured, (EffectivePolicy, TaskGrant))
    ):
        return _intersect_tool_policy(parent.delegable_grants, configured), parent
    if isinstance(configured, TaskGrant):
        if parent is None or parent.delegable_grants is None:
            raise PolicyError("a TaskGrant requires a parent policy context")
        effective = child_within_parent(parent.delegable_grants, configured)
    elif isinstance(configured, EffectivePolicy):
        effective = configured
        if parent is not None and parent.delegable_grants is not None:
            validate_child_policy(parent.delegable_grants, effective)
        if effective.approval_authority or effective.approval_cards:
            if parent is not None and parent.delegable_grants is None:
                raise PolicyWidening("approval authority requires a parent grant")
            effective = _with_policy_digest(effective)
    else:
        raise TypeError("policy must be an EffectivePolicy, TaskGrant, or None")
    return effective, parent


@contextmanager
def policy_scope(
    configured: EffectivePolicy | TaskGrant | None,
    *,
    tree_budget: TreeBudget | None = None,
    child_id: object | None = None,
    reserve_review: bool = False,
    work_units: int = 1,
    bind: bool = True,
    role: str | None = None,
    engine: str | None = None,
    fresh: bool = False,
    read_only_review: bool = False,
    approval_actor: str | None = None,
) -> Iterator[PolicyContext | None]:
    """Bind policy authority for one invocation, with an optional display label."""

    with _policy_scope(
        configured,
        tree_budget=tree_budget,
        child_id=child_id,
        reserve_review=reserve_review,
        work_units=work_units,
        bind=bind,
        role=role,
        engine=engine,
        fresh=fresh,
        read_only_review=read_only_review,
        approval_actor=approval_actor,
    ) as context:
        yield context


@contextmanager
def _policy_scope(
    configured: EffectivePolicy | TaskGrant | None,
    *,
    tree_budget: TreeBudget | None = None,
    child_id: object | None = None,
    reserve_review: bool = False,
    work_units: int = 1,
    bind: bool = True,
    role: str | None = None,
    engine: str | None = None,
    fresh: bool = False,
    read_only_review: bool = False,
    approval_actor: str | None = None,
    agent_id: str | None = None,
    session: object | None = None,
) -> Iterator[PolicyContext | None]:
    if approval_actor is not None and not approval_actor.strip():
        raise PolicyError("approval actor must be nonblank")
    tool_call = child_id is not None and _delegated_agent.get() is child_id
    effective, parent = resolve_invocation_policy(configured, tool_call=tool_call)
    if read_only_review and effective is not None:
        effective = _read_only_review_policy(effective)
    budget = parent.tree_budget if parent is not None else tree_budget
    if effective is None and budget is None:
        with _bind_policy_context(None) if bind else nullcontext():
            yield None
        return
    depth = parent.depth + 1 if parent is not None else 0
    invocation_id = uuid.uuid4().hex
    # Pool/tool calls are new work. Direct calls retain their resumable slot.
    admitted_id = (invocation_id if tool_call else child_id) if parent and budget else None
    independence = None
    if admitted_id is not None and budget is not None:
        independence = budget.admit(
            admitted_id,
            depth=depth,
            reserve_review=reserve_review,
            work_units=work_units,
            role=role,
            engine=engine,
            fresh=fresh,
            agent_id=agent_id,
            session=session,
        )
    context = PolicyContext(
        effective_policy=effective,
        delegable_grants=effective,
        invocation_id=invocation_id,
        parent_invocation_id=parent.invocation_id if parent is not None else None,
        tree_budget=budget,
        depth=depth,
        review_independence=independence,
        parent=parent,
        approval_actor=approval_actor,
        agent_id=agent_id or (parent.agent_id if parent is not None else None) or invocation_id,
    )
    object.__setattr__(context, "_approval_credential", secrets.token_urlsafe(32))
    try:
        with _bind_policy_context(context) if bind else nullcontext():
            yield context
    finally:
        object.__setattr__(context, "_approval_credential", None)
        if admitted_id is not None and budget is not None:
            budget.release(admitted_id)


@contextmanager
def _bind_policy_context(context: PolicyContext | None) -> Iterator[None]:
    """Temporarily restore an existing invocation context around a resume."""
    token = _current.set(context)
    active = _in_invocation.set(True)
    delegated = _delegated_agent.set(None)
    try:
        yield
    finally:
        _delegated_agent.reset(delegated)
        _current.reset(token)
        _in_invocation.reset(active)
