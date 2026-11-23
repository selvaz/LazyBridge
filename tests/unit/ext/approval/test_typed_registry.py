from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from lazybridge import Store
from lazybridge.ext.approval import (
    ApprovalQueue,
    ApprovalTarget,
    ApproverContext,
    ExecutorContext,
    TypedApprovalRequest,
)


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
    return ApprovalQueue(Store(), clock=clock), clock


def test_digest_redaction_decision_and_effect_outcome(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    req = request()
    ticket = queue.create_typed_ticket(req, prompt="Push topic")
    assert ticket.request is not None
    assert ticket.request.payload == {"message": "ship", "api_token": "[redacted]"}
    assert queue.decide_typed_ticket(ticket.approval_id, approver=ApproverContext(actor="owner"), verdict="approved")
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
    assert queue.decide_typed_ticket(ticket.approval_id, approver=ApproverContext(actor="owner"), verdict="approved")
    assert not queue.consume_typed_ticket(ticket.approval_id, executor=context(req, **change))


def test_authenticated_approver_cannot_claim_another_actor(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    ticket = queue.create_typed_ticket(request())
    assert not queue.decide_typed_ticket(
        ticket.approval_id, approver=ApproverContext(actor="alice"), actor="bob", verdict="approved"
    )


def test_decision_and_consumption_are_single_winner_under_race(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    req = request()
    ticket = queue.create_typed_ticket(req)
    with ThreadPoolExecutor(max_workers=8) as pool:
        decisions = list(
            pool.map(
                lambda i: queue.decide_typed_ticket(
                    ticket.approval_id, approver=ApproverContext(actor=f"owner-{i}"), verdict="approved"
                ),
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
    assert not queue.decide_typed_ticket(
        ticket.approval_id, approver=ApproverContext(actor="owner"), verdict="approved"
    )

    req = request(task_id="task-2")
    ticket = queue.create_typed_ticket(req, ttl=timedelta(seconds=1))
    assert queue.decide_typed_ticket(ticket.approval_id, approver=ApproverContext(actor="owner"), verdict="approved")
    clock.now += timedelta(seconds=1)
    assert not queue.consume_typed_ticket(ticket.approval_id, executor=context(req))


def test_legacy_history_deserializes_but_cannot_govern(registry: tuple[ApprovalQueue, Clock]) -> None:
    queue, _ = registry
    legacy = queue.create_ticket(task_id="old", prompt="old prompt")
    loaded = queue.get_ticket(legacy.approval_id)
    assert loaded and loaded.legacy
    assert queue.approve_ticket(legacy.approval_id, actor="owner", channel="cli")
    req = request(task_id="old")
    assert not queue.consume_typed_ticket(legacy.approval_id, executor=context(req))
