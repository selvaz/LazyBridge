from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest

from lazybridge import Store
from lazybridge.ext.approval import (
    ApprovalQueue,
    ApprovalTarget,
    ApproverContext,
    ExecutorContext,
    TypedApprovalRequest,
)

OWNER = ApproverContext(actor="owner", role="owner")


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 23, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def request(**changes: object) -> TypedApprovalRequest:
    values = {
        "task_id": "task-1",
        "attempt": 2,
        "action": "git.push",
        "target": ApprovalTarget(repo="org/repo", ref="refs/heads/topic", sha="abc123"),
        "requested_by": "agent-7",
        "policy_revision": "policy-4",
        "payload": {"message": "ship", "api_token": "do-not-store"},
    }
    values.update(changes)
    return TypedApprovalRequest.create(**values)


def context(req: TypedApprovalRequest, **changes: object) -> ExecutorContext:
    values = {
        "actor": req.requested_by,
        "task_id": req.task_id,
        "attempt": req.attempt,
        "digest": req.digest,
        "policy_revision": req.policy_revision,
        "target": req.target,
    }
    values.update(changes)
    return ExecutorContext(**values)


@pytest.fixture
def registry() -> tuple[ApprovalQueue, Clock]:
    clock = Clock()
    return ApprovalQueue(Store(), clock=clock, authenticate_approver=lambda: OWNER), clock


def test_digest_redaction_decision_and_effect_outcome(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    req = request()
    ticket = queue.create_typed_ticket(req, prompt="Push topic")
    assert ticket.request is not None
    assert ticket.request.payload == {"message": "ship", "api_token": "[redacted]"}
    assert queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")
    assert queue.consume_typed_ticket(ticket.approval_id, executor=context(req))
    assert queue.record_effect_outcome(ticket.approval_id, executor=context(req), succeeded=True, detail="pushed")
    stored = queue.get_ticket(ticket.approval_id)
    assert stored and stored.decision and stored.decision.decider == "owner"
    assert stored.effect_outcome and stored.effect_outcome.succeeded


def test_rejects_wrong_digest_at_creation(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    bad = request().model_copy(update={"digest": "wrong"})
    with pytest.raises(ValueError, match="digest"):
        queue.create_typed_ticket(bad)


def test_create_digests_the_validated_canonical_request() -> None:
    req = request(attempt="2", target={"repo": "org/repo"})

    assert req.attempt == 2
    assert req.target == ApprovalTarget(repo="org/repo")
    assert req.digest == req.canonical_digest()


@pytest.mark.parametrize(
    "change",
    [
        {"actor": "other"},
        {"task_id": "other"},
        {"attempt": 3},
        {"digest": "bad"},
        {"policy_revision": "old"},
        {"target": ApprovalTarget(repo="org/repo", ref="refs/heads/other", sha="abc123")},
        {"target": ApprovalTarget(repo="org/repo", ref="refs/heads/topic", sha="def456")},
    ],
)
def test_consumption_rejects_every_binding_mismatch(
    registry: tuple[ApprovalQueue, Clock], change: dict[str, object]
) -> None:
    queue, _ = registry
    req = request()
    ticket = queue.create_typed_ticket(req)
    assert queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")
    assert not queue.consume_typed_ticket(ticket.approval_id, executor=context(req, **change))


def test_authenticated_approver_cannot_claim_another_actor(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    ticket = queue.create_typed_ticket(request())
    assert not queue.decide_typed_ticket(
        ticket.approval_id, approver=OWNER, actor="bob", verdict="approved"
    )


@pytest.mark.parametrize(
    "non_owner",
    [ApproverContext(actor="coordinator", role="orchestrator"), ApproverContext(actor="owner")],
)
def test_typed_ticket_without_requester_path_is_owner_only(non_owner: ApproverContext) -> None:
    store = Store()
    queue = ApprovalQueue(store, authenticate_approver=lambda: non_owner)
    ticket = queue.create_typed_ticket(request())
    assert not ticket.requester_path
    assert queue.list_pending_tickets(assignee=non_owner) == []
    assert not queue.decide_typed_ticket(ticket.approval_id, approver=non_owner, verdict="approved")
    assert not queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")
    assert queue.get_ticket(ticket.approval_id) == ticket

    owner_queue = ApprovalQueue(store, authenticate_approver=lambda: OWNER)
    assert owner_queue.list_pending_tickets(assignee=OWNER) == [ticket]
    assert owner_queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")


def test_decision_and_consumption_are_single_winner_under_race(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    req = request()
    ticket = queue.create_typed_ticket(req)
    with ThreadPoolExecutor(max_workers=8) as pool:
        decisions = list(
            pool.map(
                lambda _: queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved"),
                range(8),
            )
        )
    assert decisions.count(True) == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        consumptions = list(
            pool.map(lambda _: queue.consume_typed_ticket(ticket.approval_id, executor=context(req)), range(8))
        )
    assert consumptions.count(True) == 1


def test_expiry_blocks_decision_and_effect_consumption(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, clock = registry
    ticket = queue.create_typed_ticket(request(), ttl=timedelta(seconds=1))
    clock.now += timedelta(seconds=1)
    assert not queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")

    req = request(task_id="task-2")
    ticket = queue.create_typed_ticket(req, ttl=timedelta(seconds=1))
    assert queue.decide_typed_ticket(ticket.approval_id, approver=OWNER, verdict="approved")
    clock.now += timedelta(seconds=1)
    assert not queue.consume_typed_ticket(ticket.approval_id, executor=context(req))


def test_pending_listing_uses_typed_clock_and_keeps_legacy_clock(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, clock = registry
    clock.now = datetime.now(UTC) + timedelta(days=1)
    legacy = queue.create_ticket(task_id="old", prompt="old prompt", ttl=timedelta(hours=1))
    expiring = queue.create_typed_ticket(request(), ttl=timedelta(seconds=1))
    clock.now += timedelta(milliseconds=1)
    waiting = queue.create_typed_ticket(request(), ttl=timedelta(seconds=2))

    assert queue.list_pending_tickets() == [legacy, expiring, waiting]

    clock.now = expiring.expires_at
    assert queue.list_pending_tickets() == [legacy, waiting]
    assert queue.get_ticket(legacy.approval_id) == legacy


def test_get_ticket_reports_effective_expiry_and_can_persist_it(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, clock = registry
    ticket = queue.create_typed_ticket(request(), ttl=timedelta(seconds=1))
    assert queue.get_ticket(ticket.approval_id) == ticket

    clock.now = ticket.expires_at
    expired = ticket.model_copy(update={"status": "expired"})
    assert queue.get_ticket(ticket.approval_id) == expired

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: queue.expire_ticket(ticket.approval_id), range(8)))
    assert results.count(True) == 1
    clock.now = ticket.created_at
    assert queue.get_ticket(ticket.approval_id) == expired


@pytest.mark.parametrize(("verdict", "consume"), [("approved", False), ("rejected", False), ("approved", True)])
def test_expiry_preserves_decisions_and_consumption(
    registry: tuple[ApprovalQueue, Clock], verdict: Literal["approved", "rejected"], consume: bool
) -> None:
    queue, clock = registry
    req = request()
    ticket = queue.create_typed_ticket(req, ttl=timedelta(seconds=1))
    approval_id = ticket.approval_id
    assert queue.decide_typed_ticket(approval_id, approver=OWNER, verdict=verdict)
    if consume:
        assert queue.consume_typed_ticket(approval_id, executor=context(req))
        assert queue.record_effect_outcome(approval_id, executor=context(req), succeeded=True)
    settled = queue.get_ticket(approval_id)
    assert settled is not None and settled.status == verdict

    clock.now = ticket.expires_at
    assert queue.get_ticket(approval_id) == settled
    assert not queue.expire_ticket(approval_id)
    assert queue.get_ticket(approval_id) == settled


def test_legacy_history_deserializes_but_cannot_govern(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    legacy = queue.create_ticket(task_id="old", prompt="old prompt")
    loaded = queue.get_ticket(legacy.approval_id)
    assert loaded and loaded.legacy
    assert queue.approve_ticket(legacy.approval_id, actor="owner", channel="cli")
    req = request(task_id="old")
    assert not queue.consume_typed_ticket(legacy.approval_id, executor=context(req))


def test_legacy_decision_and_consumption_paths_refuse_typed_ticket(
    registry: tuple[ApprovalQueue, Clock],
) -> None:
    queue, _ = registry
    req = request()

    approved = queue.create_typed_ticket(req, prompt="Push topic")
    assert not queue.approve_ticket(approved.approval_id, actor="anyone", channel="legacy")
    assert not queue.consume_typed_ticket(approved.approval_id, executor=context(req))

    rejected = queue.create_typed_ticket(request(task_id="task-2"), prompt="Reject topic")
    assert not queue.reject_ticket(rejected.approval_id, actor="anyone", channel="legacy", reason="no")

    decided = queue.create_typed_ticket(request(task_id="task-3"), prompt="Reusable")
    assert queue.decide_typed_ticket(decided.approval_id, approver=OWNER, verdict="approved")
    assert not queue.claim_earlier_approval(task_id="task-3", prompt="Reusable")
    assert not queue.mark_approval_consumed(decided.approval_id)


def test_typed_consumption_requires_approved_decision_record(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    req = request()
    ticket = queue.create_typed_ticket(req)
    key = queue._key(ticket.approval_id)
    with queue._anchored() as anchored:
        raw = anchored.read(key)
        assert isinstance(raw, dict)
        raw["status"] = "approved"
        anchored.write(key, raw)

    assert not queue.consume_typed_ticket(ticket.approval_id, executor=context(req))


@pytest.mark.parametrize("verdict", ["approved", "rejected"])
def test_legacy_decisions_use_injected_clock(registry: tuple[ApprovalQueue, Clock], verdict: str) -> None:
    queue, clock = registry
    ticket = queue.create_ticket(task_id="old", prompt="old prompt")
    clock.now = ticket.expires_at

    if verdict == "approved":
        result = queue.approve_ticket(ticket.approval_id, actor="owner", channel="cli")
    else:
        result = queue.reject_ticket(ticket.approval_id, actor="owner", channel="cli", reason="no")

    assert not result
