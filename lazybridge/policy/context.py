"""Invocation-local policy propagation for delegated work."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .models import EffectivePolicy, PolicyError, TaskGrant
from .resolve import child_within_parent, validate_child_policy


@dataclass(frozen=True)
class PolicyContext:
    """Authority carried by one invocation into all nested work."""

    effective_policy: EffectivePolicy
    delegable_grants: EffectivePolicy
    invocation_id: str
    parent_invocation_id: str | None = None


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
        if parent is None:
            raise PolicyError("a TaskGrant requires a parent policy context")
        effective = child_within_parent(parent.delegable_grants, configured)
    elif isinstance(configured, EffectivePolicy):
        effective = configured
        if parent is not None:
            validate_child_policy(parent.delegable_grants, effective)
    else:
        raise TypeError("policy must be an EffectivePolicy, TaskGrant, or None")
    return effective, parent


@contextmanager
def policy_scope(configured: EffectivePolicy | TaskGrant | None) -> Iterator[PolicyContext | None]:
    """Bind policy authority for exactly one agent invocation."""

    effective, parent = resolve_invocation_policy(configured)
    if effective is None:
        yield None
        return
    context = PolicyContext(
        effective_policy=effective,
        delegable_grants=effective,
        invocation_id=uuid.uuid4().hex,
        parent_invocation_id=parent.invocation_id if parent is not None else None,
    )
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)
