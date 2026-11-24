from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from lazybridge import Store
from lazybridge.ext.approval import (
    ApprovalQueue,
    ApprovalTarget,
    ApprovalTicket,
    ApproverContext,
    EscalationSummary,
    ExecutorContext,
    TypedApprovalRequest,
    route_ticket,
)
from lazybridge.policy import (
    ApprovalTier,
    PolicyContext,
    PolicyWidening,
    TaskGrant,
    TreeBudget,
    child_within_parent,
    compile_claude,
    compile_codex,
    policy_scope,
    resolve,
)

AUTHORITY = frozenset({"run_tests", "package_install", "publish_pr"})
OWNER = ApproverContext(actor="owner", role="owner")


@pytest.fixture
def authenticate() -> Mock:
    return Mock(return_value=None)


class Clock:
    now = datetime(2026, 11, 24, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def grant(folder: Path, authority: frozenset[str] = frozenset()) -> TaskGrant:
    return TaskGrant(folder, {"tool_classes": ["read"]}, "off", "local", approval_authority=authority)


@contextmanager
def tree(
    folder: Path, *, worker_authority: frozenset[str] = AUTHORITY, worker_profile: str = "edit-and-run"
) -> Iterator[tuple[PolicyContext, PolicyContext, PolicyContext, PolicyContext]]:
    owner_policy = resolve("edit-and-run", grant(folder, AUTHORITY))
    with (
        policy_scope(owner_policy, approval_actor="owner") as owner,
        policy_scope(grant(folder, AUTHORITY), approval_actor="orchestrator") as orchestrator,
        policy_scope(
            child_within_parent(owner_policy, grant(folder, worker_authority), profile=worker_profile),
            approval_actor="worker",
        ) as worker,
        policy_scope(grant(folder, worker_authority), approval_actor="sub-worker") as requester,
    ):
        assert owner and orchestrator and worker and requester
        yield owner, orchestrator, worker, requester


def request(
    requester: PolicyContext, action_class: str = "run_tests", tier: ApprovalTier = "routine"
) -> TypedApprovalRequest:
    return TypedApprovalRequest.create(
        task_id="task",
        attempt=1,
        action=action_class,
        target=ApprovalTarget(),
        requested_by=requester.actor,
        policy_revision="revision",
        tier=tier,
        action_class=action_class,
    )


def assignee(ticket: ApprovalTicket) -> str:
    grant_used = route_ticket(ticket)
    assert grant_used is not None
    return grant_used.actor


def test_a1_nearest_authorized_parent_decides(tmp_path: Path, authenticate: Mock) -> None:
    queue = ApprovalQueue(Store(), authenticate_approver=authenticate)
    with tree(tmp_path) as chain:
        _, orchestrator, worker, requester = chain
        req = request(requester)
        ticket = queue.create_typed_ticket(req, requester=requester)
        assert (ticket.tier, ticket.action_class) == ("routine", "run_tests")
        assert ticket.requester_path == tuple(context.invocation_id for context in chain)
        assert assignee(ticket) == "worker"
        approver = ApproverContext.from_invocation(worker)
        assert queue.list_pending_tickets(assignee=approver) == [ticket]
        assert queue.list_pending_tickets(assignee=OWNER) == []
        authenticate.return_value = OWNER
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")
        other_ancestor = ApproverContext.from_invocation(orchestrator)
        authenticate.return_value = other_ancestor
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=other_ancestor, verdict="approved")
        for change in ({"policy_digest": None}, {"policy_digest": "stale"}, {"invocation_id": "sibling"}):
            stale_approver = approver.model_copy(update=change)
            authenticate.return_value = stale_approver
            assert not queue.decide_typed_ticket(ticket.approval_id, approver=stale_approver, verdict="approved")
        authenticate.return_value = approver
        assert queue.decide_typed_ticket(ticket.approval_id, approver=approver, verdict="approved")
        decided = queue.get_ticket(ticket.approval_id)
        assert decided and decided.decision
        assert decided.decision.grant == route_ticket(ticket)
        assert decided.decision.decider == "worker"
        executor = ExecutorContext(
            actor=requester.actor,
            task_id=req.task_id,
            attempt=req.attempt,
            digest=req.digest,
            policy_revision=req.policy_revision,
            target=req.target,
        )
        assert queue.consume_typed_ticket(ticket.approval_id, executor=executor)
        assert not queue.consume_typed_ticket(ticket.approval_id, executor=executor)


def test_a2_read_parent_without_authority_is_skipped(tmp_path: Path, authenticate: Mock) -> None:
    queue = ApprovalQueue(Store(), authenticate_approver=authenticate)
    with tree(tmp_path, worker_authority=frozenset(), worker_profile="read") as (_, orchestrator, worker, requester):
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        assert assignee(ticket) == "orchestrator"
        assert queue.list_pending_tickets(assignee=ApproverContext.from_invocation(worker)) == []
        approver = ApproverContext.from_invocation(orchestrator)
        authenticate.return_value = approver
        assert queue.decide_typed_ticket(ticket.approval_id, approver=approver, verdict="approved")


def test_a3_sensitive_actions_skip_direct_parent(tmp_path: Path, authenticate: Mock) -> None:
    queue = ApprovalQueue(Store(), authenticate_approver=authenticate)
    with tree(tmp_path) as (_, orchestrator, worker, requester):
        ticket = queue.create_typed_ticket(request(requester, "package_install", "sensitive"), requester=requester)
        assert assignee(ticket) == "orchestrator"
        parent = ApproverContext.from_invocation(worker)
        authenticate.return_value = parent
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=parent, verdict="approved")
        approver = ApproverContext.from_invocation(orchestrator)
        authenticate.return_value = approver
        assert queue.decide_typed_ticket(ticket.approval_id, approver=approver, verdict="approved")


def test_a4_publication_requires_owner(tmp_path: Path, authenticate: Mock) -> None:
    queue = ApprovalQueue(Store(), authenticate_approver=authenticate)
    with tree(tmp_path) as (_, orchestrator, _, requester):
        req = request(requester, "publish_pr", "external")
        ticket = queue.create_typed_ticket(req, requester=requester)
        assert assignee(ticket) == "owner"
        ancestor = ApproverContext.from_invocation(orchestrator)
        authenticate.return_value = ancestor
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=ancestor, verdict="approved")
        without_role = ApproverContext(actor="owner")
        authenticate.return_value = without_role
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=without_role, verdict="approved")
        authenticate.return_value = OWNER
        assert queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")
        decided = queue.get_ticket(ticket.approval_id)
        assert decided and decided.decision and decided.decision.grant
        assert decided.decision.grant.source == "owner"
        with pytest.raises(ValueError, match="tier"):
            queue.create_typed_ticket(request(requester, "publish_pr", "routine"), requester=requester)
        with pytest.raises(ValueError, match="trusted requester"):
            queue.create_typed_ticket(req)


def test_external_can_route_to_configured_independent_approver(tmp_path: Path, authenticate: Mock) -> None:
    queue = ApprovalQueue(
        Store(), authenticate_approver=authenticate, independent_approvers={"publish_pr": "release-reviewer"}
    )
    with tree(tmp_path) as (*_, requester):
        ticket = queue.create_typed_ticket(request(requester, "publish_pr", "external"), requester=requester)
        assert assignee(ticket) == "release-reviewer"
        authenticate.return_value = OWNER
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")
        approver = ApproverContext(actor="release-reviewer")
        authenticate.return_value = ApproverContext.from_invocation(requester)
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=approver, verdict="approved")
        authenticate.return_value = approver
        assert queue.decide_typed_ticket(ticket.approval_id, approver=approver, verdict="approved")


def test_a5_requester_cannot_approve_even_with_the_grant(tmp_path: Path, authenticate: Mock) -> None:
    queue = ApprovalQueue(Store(), authenticate_approver=authenticate)
    with tree(tmp_path) as (*_, requester):
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        approver = ApproverContext.from_invocation(requester)
        authenticate.return_value = approver
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=approver, verdict="approved")
    with policy_scope(resolve("read", grant(tmp_path)), approval_actor="owner") as owner:
        assert owner
        ticket = queue.create_typed_ticket(request(owner, "publish_pr", "external"), requester=owner)
        assert route_ticket(ticket) is None
        authenticate.return_value = OWNER
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")


def test_a5_public_ticket_values_cannot_authenticate_an_approver(tmp_path: Path, authenticate: Mock) -> None:
    store = Store()
    queue = ApprovalQueue(store, authenticate_approver=authenticate)
    unauthenticated_queue = ApprovalQueue(store)
    with tree(tmp_path) as (_, _, worker, requester):
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        public = queue.get_ticket(ticket.approval_id)
        assert public and public.route and public.route.assignee
        authenticate.return_value = ApproverContext.from_invocation(requester)
        for grant_used in (public.route.assignee, *public.route.remaining):
            replay = ApproverContext(
                actor=grant_used.actor,
                invocation_id=grant_used.invocation_id,
                policy_digest=grant_used.policy_digest,
                role="owner" if grant_used.source == "owner" else None,
            )
            assert not unauthenticated_queue.decide_typed_ticket(public.approval_id, approver=replay, verdict="approved")
            assert not queue.decide_typed_ticket(public.approval_id, approver=replay, verdict="approved")
        assert queue.get_ticket(public.approval_id) == public

        # The assigned ancestor can still approve through a separately authenticated call.
        approver = ApproverContext.from_invocation(worker)
        authenticate.return_value = approver
        assert queue.decide_typed_ticket(public.approval_id, approver=approver, verdict="approved")


def test_a6_manual_forward_carries_summaries_to_the_owner(tmp_path: Path, authenticate: Mock) -> None:
    queue = ApprovalQueue(Store(), authenticate_approver=authenticate)
    worker_summary = EscalationSummary(
        what="Run integration tests",
        why="Confirm a regression fix",
        risk="The test touches a shared fixture",
        alternatives=("Run the unit tests only",),
    )
    orchestrator_summary = EscalationSummary(
        what="Run integration tests", why="Owner controls the fixture", risk="Fixture contention", alternatives=()
    )
    with tree(tmp_path) as (_, orchestrator, worker, requester):
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        worker_context = ApproverContext.from_invocation(worker)
        authenticate.return_value = OWNER
        assert not queue.escalate(ticket.approval_id, worker_summary, approver=OWNER)
        authenticate.return_value = ApproverContext.from_invocation(requester)
        assert not queue.escalate(ticket.approval_id, worker_summary, approver=worker_context)
        authenticate.return_value = worker_context
        assert queue.escalate(ticket.approval_id, worker_summary, approver=worker_context)
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=worker_context, verdict="approved")
        orchestrator_context = ApproverContext.from_invocation(orchestrator)
        authenticate.return_value = orchestrator_context
        assert queue.escalate(ticket.approval_id, orchestrator_summary, approver=orchestrator_context)
        [at_owner] = queue.list_pending_tickets(assignee=OWNER)
        assert at_owner.route is not None
        assert [hop.to.actor for hop in at_owner.route.hops] == ["worker", "orchestrator", "owner"]
        assert [hop.summary for hop in at_owner.route.hops] == [None, worker_summary, orchestrator_summary]


def test_a7_timeout_advances_one_eligible_hop_and_preserves_ttl(tmp_path: Path, authenticate: Mock) -> None:
    clock = Clock()
    timeout = timedelta(seconds=5)
    store = Store(db=str(tmp_path / "approvals.db"))
    queue = ApprovalQueue(store, clock=clock, escalation_timeout=timeout, authenticate_approver=authenticate)
    with tree(tmp_path) as (_, _, worker, requester):
        ticket = queue.create_typed_ticket(request(requester), requester=requester)
        clock.now += 3 * timeout
        with ThreadPoolExecutor(max_workers=4) as pool:
            observations = list(pool.map(queue.get_ticket, [ticket.approval_id] * 8))
        assert all(observed and assignee(observed) == "orchestrator" for observed in observations)
        approver = ApproverContext.from_invocation(worker)
        authenticate.return_value = approver
        assert not queue.decide_typed_ticket(ticket.approval_id, approver=approver, verdict="approved")
        reread = ApprovalQueue(store, clock=clock, escalation_timeout=timeout)
        first = reread.get_ticket(ticket.approval_id)
        assert first and first.route
        assert assignee(first) == "orchestrator"
        assert first.route.hops[-1].reason == "timeout"
        assert first.route.hops[-1].summary is not None
        assert first.expires_at == ticket.expires_at
        clock.now += timeout
        [second] = reread.list_pending_tickets(assignee=OWNER)
        assert second.route is not None
        assert [hop.reason for hop in second.route.hops] == ["initial", "timeout", "timeout"]
        assert second.route.deadline is None


def test_a8_authority_is_delegable_and_changes_only_the_policy_digest(tmp_path: Path) -> None:
    base = resolve("read", grant(tmp_path))
    parent = resolve("read", grant(tmp_path, frozenset({"run_tests"})))
    assert base.digest != parent.digest
    for compile_policy in (compile_claude, compile_codex):
        base_settings, parent_settings = compile_policy(base), compile_policy(parent)
        assert base_settings.pop("policy_digest") == base.digest
        assert parent_settings.pop("policy_digest") == parent.digest
        assert base_settings == parent_settings
    child = child_within_parent(parent, grant(tmp_path, frozenset({"run_tests"})))
    assert child.approval_authority == frozenset({"run_tests"})
    with pytest.raises(PolicyWidening, match="approval authority"):
        child_within_parent(parent, grant(tmp_path, AUTHORITY))
    with policy_scope(parent), policy_scope(None) as default_child:
        assert default_child and default_child.effective_policy
        assert default_child.effective_policy.approval_authority == frozenset()
        assert default_child.effective_policy.digest == base.digest
    with (
        policy_scope(None, tree_budget=TreeBudget()),
        pytest.raises(PolicyWidening, match="parent grant"),
        policy_scope(parent),
    ):
        pytest.fail("a budget-only parent delegated approval authority it did not hold")
    with policy_scope(replace(parent, digest=base.digest)) as context:
        assert context and context.effective_policy
        assert context.effective_policy.digest == parent.digest


def test_historical_typed_ticket_keeps_its_original_request_digest() -> None:
    old_body = {
        "task_id": "old-task",
        "attempt": 1,
        "action": "git.push",
        "target": ApprovalTarget().model_dump(mode="json"),
        "requested_by": "old-worker",
        "policy_revision": "old-revision",
        "payload": {},
    }
    old_digest = hashlib.sha256(json.dumps(old_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    store = Store()
    queue = ApprovalQueue(store, authenticate_approver=lambda: OWNER)
    request = TypedApprovalRequest.model_validate({**old_body, "digest": old_digest})
    assert request.canonical_digest() == old_digest
    ticket = queue.create_typed_ticket(request)
    old_raw = ticket.model_dump(
        mode="json",
        exclude={
            "tier",
            "action_class",
            "requester_path",
            "route",
            "card_proposal",
            "proposal_fingerprint",
            "proposal_parent",
            "proposal_parent_invocation_id",
        },
    )
    old_raw["request"] = {**old_body, "digest": old_digest}
    store.write(f"approval:{ticket.approval_id}", old_raw)
    assert queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")
    assert queue.consume_typed_ticket(
        ticket.approval_id,
        executor=ExecutorContext(
            actor=request.requested_by,
            task_id=request.task_id,
            attempt=request.attempt,
            digest=old_digest,
            policy_revision=request.policy_revision,
            target=request.target,
        ),
    )
