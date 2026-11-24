"""Invocation-local policy propagation for delegated work."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock

from .models import EffectivePolicy, PolicyError, TaskGrant
from .resolve import child_within_parent, validate_child_policy


class PolicyBudgetExceeded(PolicyError):
    """Raised when admitting work would exceed a tree-wide bound."""

    def __init__(self, bound: str, limit: int) -> None:
        self.bound = bound
        self.limit = limit
        super().__init__(f"tree policy budget exceeded: {bound}={limit}")


@dataclass
class TreeBudget:
    """Shared, concurrency-safe admission budget for an invocation tree.

    Child identities are registered for the lifetime of the budget, so a
    later resume of the same child reuses its total/work-unit slot.
    """

    max_depth: int | None = None
    max_active_children: int | None = None
    max_total_children: int | None = None
    max_work_units: int | None = None
    _children: set[object] = field(default_factory=set, init=False, repr=False)
    _active: Counter[object] = field(default_factory=Counter, init=False, repr=False)
    _work_units: int = field(default=0, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("max_depth", "max_active_children", "max_total_children", "max_work_units"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")

    def admit(self, child_id: object, *, depth: int, reserve_review: bool = False, work_units: int = 1) -> None:
        """Atomically admit a child, or raise without changing counters."""
        if work_units < 0:
            raise ValueError("work_units must be non-negative")
        with self._lock:
            is_new = child_id not in self._children
            becomes_active = not self._active.get(child_id, 0)
            active_limit = self.max_active_children
            if self.max_depth is not None and depth > self.max_depth:
                raise PolicyBudgetExceeded("max_depth", self.max_depth)
            if becomes_active and active_limit is not None:
                usable = max(0, active_limit - int(reserve_review))
                if len(self._active) >= usable:
                    raise PolicyBudgetExceeded("max_active_children", active_limit)
            if is_new and self.max_total_children is not None and len(self._children) >= self.max_total_children:
                raise PolicyBudgetExceeded("max_total_children", self.max_total_children)
            if is_new and self.max_work_units is not None and self._work_units + work_units > self.max_work_units:
                raise PolicyBudgetExceeded("max_work_units", self.max_work_units)
            if is_new:
                self._children.add(child_id)
                self._work_units += work_units
            self._active[child_id] += 1

    def release(self, child_id: object) -> None:
        """Release one active use while retaining its resumable slot."""
        with self._lock:
            self._active[child_id] -= 1
            if self._active[child_id] <= 0:
                del self._active[child_id]


@dataclass(frozen=True)
class PolicyContext:
    """Authority carried by one invocation into all nested work."""

    effective_policy: EffectivePolicy | None
    delegable_grants: EffectivePolicy | None
    invocation_id: str
    parent_invocation_id: str | None = None
    tree_budget: TreeBudget | None = None
    depth: int = 0


_current: ContextVar[PolicyContext | None] = ContextVar("lazybridge_policy_context", default=None)


def current_policy_context() -> PolicyContext | None:
    """Return the active invocation policy, if policy enforcement is enabled."""

    return _current.get()


def resolve_invocation_policy(
    configured: EffectivePolicy | TaskGrant | None,
) -> tuple[EffectivePolicy | None, PolicyContext | None]:
    """Resolve an agent's configured authority against the immediate caller."""

    parent = _current.get()
    if configured is None:
        return (parent.effective_policy, parent) if parent is not None else (None, None)
    if isinstance(configured, TaskGrant):
        if parent is None or parent.delegable_grants is None:
            raise PolicyError("a TaskGrant requires a parent policy context")
        effective = child_within_parent(parent.delegable_grants, configured)
    elif isinstance(configured, EffectivePolicy):
        effective = configured
        if parent is not None and parent.delegable_grants is not None:
            validate_child_policy(parent.delegable_grants, effective)
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
) -> Iterator[PolicyContext | None]:
    """Bind policy authority for exactly one agent invocation."""

    effective, parent = resolve_invocation_policy(configured)
    budget = parent.tree_budget if parent is not None else tree_budget
    if effective is None and budget is None:
        yield None
        return
    depth = parent.depth + 1 if parent is not None else 0
    admitted_id = child_id if parent is not None and budget is not None else None
    if admitted_id is not None:
        budget.admit(admitted_id, depth=depth, reserve_review=reserve_review, work_units=work_units)
    context = PolicyContext(
        effective_policy=effective,
        delegable_grants=effective,
        invocation_id=uuid.uuid4().hex,
        parent_invocation_id=parent.invocation_id if parent is not None else None,
        tree_budget=budget,
        depth=depth,
    )
    token = _current.set(context) if bind else None
    try:
        yield context
    finally:
        if token is not None:
            _current.reset(token)
        if admitted_id is not None:
            budget.release(admitted_id)


@contextmanager
def bind_policy_context(context: PolicyContext | None) -> Iterator[None]:
    """Temporarily restore an existing invocation context around a resume."""
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)
