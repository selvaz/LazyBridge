"""Typed approval routing over trusted invocation ancestry."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

if TYPE_CHECKING:
    from lazybridge.policy import ApprovalTier, PolicyContext

    from .queue import ApprovalTicket

DEFAULT_ACTION_TIERS: dict[str, ApprovalTier] = {
    "command": "routine",
    "run_tests": "routine",
    "local_commit": "routine",
    "scratch_env": "routine",
    "cleanup": "routine",
    "network": "sensitive",
    "package_install": "sensitive",
    "push_branch": "sensitive",
    "publish_pr": "external",
    "production": "external",
    "money": "external",
    "deletion": "external",
    "message": "external",
}

_SummaryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EscalationSummary(BaseModel):
    """Context supplied by the forwarder or recorded by the timeout handler."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    what: _SummaryText
    why: _SummaryText
    risk: _SummaryText
    alternatives: tuple[_SummaryText, ...]


class ApprovalGrantUse(BaseModel):
    """Identity and the authority that made an assignee eligible."""

    model_config = ConfigDict(frozen=True)

    actor: str
    action_class: str
    source: Literal["policy", "owner", "independent"]
    invocation_id: str | None = None
    policy_digest: str | None = None
    agent_id: str | None = None


class ApproverContext(BaseModel):
    """Identity and policy reference for routing views and feedback."""

    model_config = ConfigDict(revalidate_instances="always")

    actor: str
    role: str | None = None
    invocation_id: str | None = None
    policy_digest: str | None = None
    agent_id: str | None = None
    roles: frozenset[str] = frozenset()

    @classmethod
    def from_invocation(cls, context: PolicyContext) -> ApproverContext:
        policy = context.effective_policy
        return cls(
            actor=context.invocation_id,
            invocation_id=context.invocation_id,
            policy_digest=policy.digest if policy else None,
            agent_id=context.agent_id,
        )


class ApprovalHop(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_actor: str
    to: ApprovalGrantUse
    at: datetime
    reason: Literal["initial", "forwarded", "timeout"]
    summary: EscalationSummary | None = None


class ApprovalRoute(BaseModel):
    """Durable route, including the eligible escalation chain and its history."""

    model_config = ConfigDict(frozen=True)

    assignee: ApprovalGrantUse | None = None
    remaining: tuple[ApprovalGrantUse, ...] = ()
    deadline: datetime | None = None
    hops: tuple[ApprovalHop, ...] = ()

    def advance(self, timeout: timedelta, *, now: datetime, hop: ApprovalHop) -> ApprovalRoute:
        return ApprovalRoute(
            assignee=hop.to,
            remaining=self.remaining[1:],
            deadline=now + timeout if len(self.remaining) > 1 else None,
            hops=(*self.hops, hop),
        )


def invocation_chain(requester: PolicyContext) -> tuple[PolicyContext, ...]:
    """Return the invocation chain from root through requester."""
    chain = [requester]
    parent = requester.parent
    while parent is not None:
        chain.append(parent)
        parent = parent.parent
    return tuple(reversed(chain))


def build_route(
    chain: tuple[PolicyContext, ...],
    *,
    tier: ApprovalTier,
    action_class: str,
    owner: str,
    operator_only: bool,
    now: datetime,
    timeout: timedelta,
) -> ApprovalRoute:
    requester = chain[-1] if chain else None
    candidates: list[ApprovalGrantUse] = []
    if requester is not None and tier != "external" and not operator_only:
        ancestors = chain[:-2] if tier == "sensitive" else chain[:-1]
        for ancestor in reversed(ancestors):
            policy = ancestor.effective_policy
            if (
                ancestor.agent_id != requester.agent_id
                and policy is not None
                and action_class in policy.approval_authority
            ):
                candidates.append(
                    ApprovalGrantUse(
                        actor=ancestor.invocation_id,
                        action_class=action_class,
                        source="policy",
                        invocation_id=ancestor.invocation_id,
                        policy_digest=policy.digest,
                        agent_id=ancestor.agent_id,
                    )
                )
    candidates.append(ApprovalGrantUse(actor=owner, action_class=action_class, source="owner"))
    first, *remaining = candidates
    from_actor = requester.invocation_id if requester is not None else "unscoped"
    return ApprovalRoute(
        assignee=first,
        remaining=tuple(remaining),
        deadline=now + timeout if remaining else None,
        hops=(ApprovalHop(from_actor=from_actor, to=first, at=now, reason="initial"),),
    )


def route_ticket(ticket: ApprovalTicket) -> ApprovalGrantUse | None:
    """Return the ticket's current assignee and its qualifying grant."""
    return ticket.route.assignee if ticket.route is not None else None
