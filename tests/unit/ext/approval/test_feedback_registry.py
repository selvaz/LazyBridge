from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from lazybridge import Store
from lazybridge.ext.approval import (
    ApprovalQueue,
    ApprovalTarget,
    ApproverContext,
    ExecutorContext,
    FeedbackAnswer,
    FeedbackRequest,
    RequestExecutorContext,
)

from .test_typed_registry import OWNER, Clock
from .test_typed_registry import request as approval_request


def feedback_request(**changes: object) -> FeedbackRequest:
    values = {
        "task_id": "task-1",
        "attempt": 2,
        "requested_by": "agent-7",
        "question": "Which rollout should I use?",
        "options": ["Canary", "Staged"],
        "required_responder": "orchestrator",
        "context_refs": ["artifact:rollout-plan"],
        "policy_revision": "policy-4",
    }
    return FeedbackRequest.create(**(values | changes))


def executor_for(request: FeedbackRequest, **changes: object) -> RequestExecutorContext:
    bindings = {
        "actor": request.requested_by,
        "task_id": request.task_id,
        "attempt": request.attempt,
        "digest": request.digest,
        "policy_revision": request.policy_revision,
    }
    return RequestExecutorContext.model_validate(bindings | changes)


ORCHESTRATOR = ApproverContext(actor="coordinator-2", role="orchestrator")


def test_feedback_creation_persists_redacted_request_in_shared_registry() -> None:
    store = Store()
    queue = ApprovalQueue(store, clock=Clock())
    approval = queue.create_typed_ticket(approval_request())
    request = feedback_request(payload={"nested": [{"api_key": "do-not-store", "region": "west"}]})
    ticket = queue.create_feedback_ticket(request)

    assert ticket.kind == "escalation"
    assert not ticket.legacy
    assert ticket.status == "pending"
    assert ticket.prompt == request.question
    assert ticket.feedback_request is not None
    assert ticket.feedback_request.context_refs == ["artifact:rollout-plan"]
    assert ticket.feedback_request.digest == request.canonical_digest()
    assert ticket.feedback_request.payload == {"nested": [{"api_key": "[redacted]", "region": "west"}]}
    assert request.payload == {"nested": [{"api_key": "do-not-store", "region": "west"}]}
    assert "do-not-store" not in repr(store.read_all())
    assert set(store.keys()) == {f"approval:{approval.approval_id}", f"approval:{ticket.approval_id}"}


def test_pending_approvals_exclude_feedback_before_applying_limit() -> None:
    clock = Clock()
    queue = ApprovalQueue(Store(), clock=clock)
    feedback = queue.create_feedback_ticket(feedback_request())
    clock.now += timedelta(seconds=1)
    approval = queue.create_typed_ticket(approval_request(), kind="escalation")

    assert queue.list_pending_tickets(limit=1) == [approval]

    legacy = queue.create_ticket(task_id="old", prompt="Review this", kind="escalation")
    assert queue.list_pending_tickets() == sorted([approval, legacy], key=lambda ticket: ticket.created_at)
    assert queue.list_pending_feedback() == [feedback]


def test_feedback_digest_covers_responder_role_and_rejects_tampering() -> None:
    queue = ApprovalQueue(Store(), clock=Clock())
    original = feedback_request()
    tampered = original.model_copy(update={"required_responder": "owner"})
    assert tampered.canonical_digest() != original.digest
    with pytest.raises(ValueError, match="digest"):
        queue.create_feedback_ticket(tampered)
    with pytest.raises(ValueError, match="ttl"):
        queue.create_feedback_ticket(original, ttl=timedelta(0))


@pytest.mark.parametrize("changes", [{"attempt": 0}, {"question": " "}, {"options": ["x" * 121]}])
def test_feedback_creation_revalidates_request_models(changes: dict[str, object]) -> None:
    queue = ApprovalQueue(Store(), clock=Clock())
    invalid = feedback_request().model_copy(update=changes)
    invalid.digest = invalid.canonical_digest()

    with pytest.raises(ValidationError):
        queue.create_feedback_ticket(invalid)


def test_feedback_choice_persists_once_and_is_private_after_reopen(tmp_path: Path) -> None:
    db = str(tmp_path / "registry.sqlite")
    clock = Clock()
    request = feedback_request()
    executor = executor_for(request)
    with Store(db=db) as store:
        queue = ApprovalQueue(store, clock=clock)
        ticket = queue.create_feedback_ticket(request)
        assert queue.read_feedback_answer(ticket.approval_id, executor=executor) is None
        clock.now += timedelta(seconds=1)
        assert queue.answer_feedback_ticket(ticket.approval_id, responder=ORCHESTRATOR, choice="Canary")
        assert not queue.answer_feedback_ticket(ticket.approval_id, responder=ORCHESTRATOR, choice="Staged")

    with Store(db=db) as store:
        queue = ApprovalQueue(store, clock=clock)
        expected = FeedbackAnswer(responder=ORCHESTRATOR.actor, answered_at=clock.now, choice="Canary")
        assert queue.read_feedback_answer(ticket.approval_id, executor=executor) == expected
        assert queue.read_feedback_answer(ticket.approval_id, executor=executor) == expected
        effect_executor = ExecutorContext(**executor.model_dump(), target=ApprovalTarget())
        assert queue.read_feedback_answer(ticket.approval_id, executor=effect_executor) == expected
        public_ticket = queue.get_ticket(ticket.approval_id)
        assert public_ticket and public_ticket.status == "answered"
        assert "feedback_answer" not in public_ticket.model_dump()
        assert queue.list_pending_feedback() == queue.list_pending_tickets() == []
        assert not queue.expire_ticket(ticket.approval_id)


@pytest.mark.parametrize("role", [None, "agent", "orchestrator"])
def test_owner_feedback_uses_authenticated_role_not_actor_name(role: str | None) -> None:
    queue = ApprovalQueue(Store(), clock=Clock())
    request = feedback_request(required_responder="owner", options=None)
    ticket = queue.create_feedback_ticket(request)
    assert not queue.answer_feedback_ticket(
        ticket.approval_id, responder=ApproverContext(actor="owner", role=role), text="Proceed"
    )
    owner = ApproverContext(actor="owner-account-1", role="owner")
    assert queue.answer_feedback_ticket(ticket.approval_id, responder=owner, text="Use the regional rollout")
    answer = queue.read_feedback_answer(ticket.approval_id, executor=executor_for(request))
    assert answer and answer.text == "Use the regional rollout" and answer.choice is None


@pytest.mark.parametrize(
    "bindings",
    [
        {"actor": "coordinator-2"},
        {"task_id": "task-2"},
        {"attempt": 3},
        {"digest": "other-request"},
        {"policy_revision": "old-policy"},
    ],
)
def test_feedback_read_rejects_each_binding_mismatch(bindings: dict[str, object]) -> None:
    queue = ApprovalQueue(Store(), clock=Clock())
    request = feedback_request()
    ticket = queue.create_feedback_ticket(request)
    assert queue.answer_feedback_ticket(ticket.approval_id, responder=ORCHESTRATOR, choice="Staged")
    assert queue.read_feedback_answer(ticket.approval_id, executor=executor_for(request, **bindings)) is None


@pytest.mark.parametrize("answer", [{}, {"choice": "Other"}, {"text": "  "}, {"choice": "Canary", "text": "Both"}])
def test_invalid_answer_leaves_feedback_pending(answer: dict[str, str]) -> None:
    queue = ApprovalQueue(Store(), clock=Clock())
    ticket = queue.create_feedback_ticket(feedback_request())

    assert not queue.answer_feedback_ticket(ticket.approval_id, responder=ORCHESTRATOR, **answer)
    assert queue.list_pending_feedback() == [ticket]


def test_feedback_cannot_be_used_as_an_approval_and_legacy_escalation_keeps_its_verbs() -> None:
    queue = ApprovalQueue(Store(), clock=Clock(), allow_credential_issuance=True)
    credential = queue.issue_approver_credential(OWNER.actor, {"owner"})
    request = feedback_request()
    feedback = queue.create_feedback_ticket(request)
    legacy = queue.create_ticket(task_id=request.task_id, prompt=request.question, kind="escalation")
    approval = queue.create_typed_ticket(approval_request(), kind="escalation")
    executor = ExecutorContext(**executor_for(request).model_dump(), target=ApprovalTarget())

    assert not queue.approve_ticket(feedback.approval_id, actor="owner", channel="cli")
    assert not queue.reject_ticket(feedback.approval_id, actor="owner", channel="cli", reason="no")
    assert not queue.decide_typed_ticket(feedback.approval_id, credential=credential, verdict="approved")
    assert not queue.answer_feedback_ticket(approval.approval_id, responder=ORCHESTRATOR, text="yes")
    assert not queue.answer_feedback_ticket(legacy.approval_id, responder=ORCHESTRATOR, text="yes")
    assert queue.answer_feedback_ticket(feedback.approval_id, responder=ORCHESTRATOR, text="yes")
    assert not queue.consume_typed_ticket(feedback.approval_id, executor=executor)
    assert not queue.mark_approval_consumed(feedback.approval_id)
    assert queue.approve_ticket(legacy.approval_id, actor="owner", channel="cli")
    assert queue.decide_typed_ticket(approval.approval_id, credential=credential, verdict="approved")


def test_concurrent_feedback_answers_have_one_winner() -> None:
    queue = ApprovalQueue(Store(), clock=Clock())
    request = feedback_request()
    ticket = queue.create_feedback_ticket(request)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda i: queue.answer_feedback_ticket(ticket.approval_id, responder=ORCHESTRATOR, text=str(i)),
                range(8),
            )
        )
    assert results.count(True) == 1
    answer = queue.read_feedback_answer(ticket.approval_id, executor=executor_for(request))
    assert answer and answer.text == str(results.index(True))


def test_pending_expiry_and_answer_read_use_typed_clock() -> None:
    clock = Clock()
    queue = ApprovalQueue(Store(), clock=clock)
    request = feedback_request()
    expired = queue.create_feedback_ticket(request, ttl=timedelta(seconds=1))
    answered = queue.create_feedback_ticket(request, ttl=timedelta(seconds=1))
    approval = queue.create_typed_ticket(approval_request(), ttl=timedelta(seconds=1))
    assert queue.answer_feedback_ticket(answered.approval_id, responder=ORCHESTRATOR, choice="Canary")

    clock.now += timedelta(seconds=1)
    for ticket in (expired, approval):
        loaded = queue.get_ticket(ticket.approval_id)
        assert loaded and loaded.status == "expired"
    assert queue.list_pending_feedback() == queue.list_pending_tickets() == []
    assert not queue.answer_feedback_ticket(expired.approval_id, responder=ORCHESTRATOR, choice="Canary")
    assert queue.read_feedback_answer(answered.approval_id, executor=executor_for(request)) is None
    assert queue.expire_ticket(expired.approval_id)


def test_pending_feedback_orders_by_priority_then_age_before_applying_limit() -> None:
    clock = Clock()
    queue = ApprovalQueue(Store(), clock=clock)
    queue.create_typed_ticket(approval_request())
    low = queue.create_feedback_ticket(feedback_request(priority="low"))
    normal = queue.create_feedback_ticket(feedback_request())
    high = queue.create_feedback_ticket(feedback_request(priority="high"))
    urgent_old = queue.create_feedback_ticket(feedback_request(priority="urgent"))
    clock.now += timedelta(seconds=1)
    urgent_new = queue.create_feedback_ticket(feedback_request(priority="urgent"))

    assert queue.list_pending_feedback() == [urgent_old, urgent_new, high, normal, low]
    assert queue.list_pending_feedback(limit=3) == [urgent_old, urgent_new, high]
