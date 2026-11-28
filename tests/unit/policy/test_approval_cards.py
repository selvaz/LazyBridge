from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from lazybridge import Store
from lazybridge.ext.approval import (
    ApprovalQueue,
    ApprovalTarget,
    ApprovalTicket,
    ApproverContext,
    CardApproval,
    TypedApprovalRequest,
    route_ticket,
)
from lazybridge.ext.approval.cards import _CardRegistry, _CardState
from lazybridge.policy import (
    ApprovalCard,
    ApprovalTier,
    CardAction,
    CardMatcher,
    PolicyContext,
    PolicyWidening,
    TaskGrant,
    policy_scope,
    resolve,
)


class Clock:
    now = datetime(2026, 11, 24, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def grant(folder: Path, *cards: ApprovalCard, shell: bool = True, network: str = "off") -> TaskGrant:
    return TaskGrant(
        folder,
        {"tool_classes": ["read", "shell"] if shell else ["read"]},
        network,
        "local",
        approval_cards=cards,
    )


def request(
    requester: PolicyContext,
    action: CardAction,
    *,
    action_class: str = "cleanup",
    tier: ApprovalTier = "routine",
    task_id: str = "task",
    attempt: int = 1,
    target: ApprovalTarget | None = None,
) -> TypedApprovalRequest:
    return TypedApprovalRequest.create(
        task_id=task_id,
        attempt=attempt,
        action=action_class,
        target=target or ApprovalTarget(),
        requested_by=requester.actor,
        policy_revision="revision",
        tier=tier,
        action_class=action_class,
        card_action=action,
    )


def cleanup(*argv: str, paths: tuple[Path, ...] = (Path("scratch/item"),)) -> CardAction:
    return CardAction(argv=argv or ("rm", "scratch/item"), cwd=Path("."), paths=paths, network="off")


def test_a9_standing_cleanup_matches_only_a_single_in_folder_command(tmp_path: Path) -> None:
    folder = tmp_path / "worktree"
    folder.mkdir()
    store = Store()
    queue = ApprovalQueue(store)
    owner_credential = ApprovalQueue(store, allow_credential_issuance=True)
    credential = owner_credential.issue_approver_credential("owner", {"owner"})
    original = queue.issue_card(
        credential=credential,
        action_class="cleanup",
        tier="routine",
        matcher=CardMatcher(command=("rm",), path_prefix=folder / "scratch"),
        transferable=True,
    )
    policy = resolve("edit-and-run", grant(folder, original))
    with policy_scope(policy, approval_actor="owner") as owner:
        assert owner
        copy = queue.copy_card(original.card_id, holder=owner)
        with policy_scope(grant(folder, copy), approval_actor="worker") as worker:
            assert worker and worker.effective_policy
            req = request(worker, cleanup())
            approved = queue.submit_typed_request(req, requester=worker)
            assert isinstance(approved, CardApproval)
            assert approved.chain == (original.card_id, copy.card_id)
            assert approved.card_id == copy.card_id
            assert approved.task_id == req.task_id and approved.action == req.action
            assert approved.policy_digest == worker.effective_policy.digest
            assert approved.request_digest == req.digest
            assert queue.list_card_approvals() == [approved]
            assert queue.list_pending_tickets() == []

            for action in (
                cleanup("rm", "scratch/item", "&&", "whoami"),
                cleanup("rm", "scratch/item; whoami"),
                cleanup("rm", "scratch/item\nwhoami"),
                cleanup("rm", "scratch/item", "|", "whoami"),
                cleanup("rm", "scratch/item", "$(whoami)"),
                cleanup("rm", "../outside", paths=(Path("../outside"),)),
                cleanup("rm", "scratch/item", paths=(Path("scratch/item"), tmp_path / "outside")),
                cleanup().model_copy(update={"cwd": tmp_path}),
                cleanup("rm", "scratch/..", paths=(Path("scratch/.."),)),
            ):
                ticket = queue.submit_typed_request(request(worker, action), requester=worker)
                assert isinstance(ticket, ApprovalTicket)
                assignee = route_ticket(ticket)
                assert assignee and assignee.actor == "owner"

            assert isinstance(queue.submit_typed_request(req, requester=worker, operator_only=True), ApprovalTicket)
            assert queue.list_card_approvals() == [approved]

        with policy_scope(grant(folder), approval_actor="empty-parent") as empty:
            assert empty
            with pytest.raises(PolicyWidening, match="does not hold"):
                queue.copy_card(original.card_id, holder=empty)
            with pytest.raises(PolicyWidening, match="does not hold"), policy_scope(grant(folder, copy)):
                pytest.fail("a child received a card absent from the parent's hand")

    with pytest.raises(ValueError, match="tier"):
        queue.issue_card(
            credential=credential, action_class="publish_pr", tier="routine", matcher=CardMatcher(command=("gh", "pr"))
        )
    with pytest.raises(ValidationError, match=r"routine.*sensitive"):
        ApprovalCard.model_validate(
            {"card_id": "publication", "action_class": "publish_pr", "tier": "external", "matcher": original.matcher}
        )


def test_a10_three_identical_misses_propose_one_card_to_the_parent(tmp_path: Path) -> None:
    store = Store(db=str(tmp_path / "approvals.db"))
    queue = ApprovalQueue(store)
    with policy_scope(resolve("edit-and-run", grant(tmp_path)), approval_actor="owner") as owner:
        assert owner
        parent = ApproverContext.from_invocation(owner).model_copy(update={"actor": "unrelated-label"})
        with policy_scope(grant(tmp_path), approval_actor="worker") as worker:
            assert worker
            for attempt in (1, 2):
                assert isinstance(
                    queue.submit_typed_request(request(worker, cleanup(), attempt=attempt), requester=worker),
                    ApprovalTicket,
                )
            different = cleanup("rm", "scratch/different", paths=(Path("scratch/different"),))
            queue.submit_typed_request(request(worker, different), requester=worker)
        assert queue.list_card_proposals(parent=parent) == []
        with policy_scope(grant(tmp_path), approval_actor="owner") as renamed:
            assert renamed and renamed.agent_id == worker.agent_id
            queue.submit_typed_request(request(renamed, cleanup(), attempt=3), requester=renamed)
        reopened = ApprovalQueue(store)
        [proposal] = reopened.list_card_proposals(parent=parent)
        assert proposal.occurrences == 3
        assert proposal.requester in {"worker", "owner"}
        assert proposal.task_id == "task"
        assert proposal.matcher.command == ("rm", "scratch/item")
        assert proposal.matcher.arguments == ()
        assert proposal.matcher.path_prefix == tmp_path / "scratch" / "item"
        assert reopened.list_card_proposals(parent=ApproverContext.from_invocation(worker)) == []
        assert reopened.list_card_proposals(parent=parent.model_copy(update={"invocation_id": "different"})) == []


def test_a11_revocation_cascades_two_levels_and_nontransferable_stops_delegation(tmp_path: Path) -> None:
    clock = Clock()
    store = Store(db=str(tmp_path / "approvals.db"))
    queue = ApprovalQueue(store, clock=clock)
    owner_credential = ApprovalQueue(store, allow_credential_issuance=True)
    credential = owner_credential.issue_approver_credential("owner", {"owner"})
    original = queue.issue_card(
        credential=credential,
        action_class="cleanup",
        tier="routine",
        matcher=CardMatcher(command=("rm",), path_prefix=tmp_path / "scratch"),
        expires_at=clock.now + timedelta(hours=3),
        transferable=True,
    )
    with policy_scope(resolve("edit-and-run", grant(tmp_path, original)), approval_actor="owner") as owner:
        assert owner and original.expires_at
        with pytest.raises(PolicyWidening, match="limits"):
            queue.copy_card(original.card_id, holder=owner, expires_at=original.expires_at + timedelta(seconds=1))
        with pytest.raises(PolicyWidening, match="matcher"):
            queue.copy_card(original.card_id, holder=owner, matcher=CardMatcher(command=("rm",)))
        copy = queue.copy_card(original.card_id, holder=owner, transferable=True)
        with policy_scope(grant(tmp_path, copy), approval_actor="worker") as worker:
            assert worker
            leaf = queue.copy_card(copy.card_id, holder=worker)
            assert leaf.provenance == (original.card_id, copy.card_id)
            with policy_scope(grant(tmp_path, leaf), approval_actor="sub-worker") as sub_worker:
                assert sub_worker
                with pytest.raises(PolicyWidening, match="transferable"):
                    queue.copy_card(leaf.card_id, holder=sub_worker)
                req = request(sub_worker, cleanup())
                assert isinstance(queue.submit_typed_request(req, requester=sub_worker), CardApproval)
                assert ApprovalQueue(store, clock=clock).revoke_card(original.card_id, credential=credential)
                assert isinstance(queue.submit_typed_request(req, requester=sub_worker), ApprovalTicket)
            assert isinstance(queue.submit_typed_request(request(worker, cleanup()), requester=worker), ApprovalTicket)


def test_a12_class_card_specialisation_checks_branch_path_uses_and_expiry(tmp_path: Path) -> None:
    clock = Clock()
    store = Store()
    queue = ApprovalQueue(store, clock=clock)
    owner_credential = ApprovalQueue(store, allow_credential_issuance=True)
    credential = owner_credential.issue_approver_credential("owner", {"owner"})
    original = queue.issue_card(
        credential=credential,
        action_class="local_commit",
        tier="routine",
        matcher=CardMatcher(command=("git", "commit"), path_prefix=tmp_path),
        max_uses=6,
        transferable=True,
    )

    def commit(*, branch: str = "topic", path: str = "src/code.py") -> CardAction:
        return CardAction(
            argv=("git", "commit", "--only", path), cwd=Path("."), paths=(Path(path),), network="off", branch=branch
        )

    with policy_scope(resolve("edit-and-run", grant(tmp_path, original)), approval_actor="owner") as owner:
        assert owner
        with pytest.raises(PolicyWidening, match="matcher"):
            queue.copy_card(original.card_id, holder=owner, matcher=CardMatcher(command=("git",)))
        with pytest.raises(PolicyWidening, match="limits"):
            queue.copy_card(original.card_id, holder=owner, max_uses=7)
        restrictions = CardMatcher(command=("git", "commit"), branch="topic", path_prefix=tmp_path / "src")
        copy = queue.copy_card(
            original.card_id,
            holder=owner,
            matcher=restrictions,
            task_id="task",
            max_uses=3,
            expires_at=clock.now + timedelta(hours=2),
        )
        assert copy.provenance == (original.card_id,)
        assert copy.max_uses == 3
        assert copy.expires_at == clock.now + timedelta(hours=2)
        expiry_copy = queue.copy_card(
            original.card_id, holder=owner, matcher=restrictions, expires_at=clock.now + timedelta(hours=2)
        )
        with policy_scope(grant(tmp_path, copy), approval_actor="worker") as worker:
            assert worker
            for action in (commit(branch="other"), commit(path="docs/index.md"), commit(path="src-other/item")):
                assert isinstance(
                    queue.submit_typed_request(request(worker, action, action_class="local_commit"), requester=worker),
                    ApprovalTicket,
                )
            assert isinstance(
                queue.submit_typed_request(
                    request(worker, commit(), action_class="local_commit", task_id="other"), requester=worker
                ),
                ApprovalTicket,
            )
            for attempt in (1, 2, 3, 4):
                result = queue.submit_typed_request(
                    request(worker, commit(), action_class="local_commit", attempt=attempt), requester=worker
                )
                assert isinstance(result, CardApproval if attempt <= 3 else ApprovalTicket)
        with policy_scope(grant(tmp_path, expiry_copy), approval_actor="second-worker") as worker:
            assert worker
            clock.now += timedelta(hours=2)
            assert isinstance(
                queue.submit_typed_request(request(worker, commit(), action_class="local_commit"), requester=worker),
                ApprovalTicket,
            )


def test_cards_respect_the_execution_ceiling_and_symlinks(tmp_path: Path) -> None:
    folder = tmp_path / "worktree"
    folder.mkdir()
    store = Store()
    queue = ApprovalQueue(store)
    owner_credential = ApprovalQueue(store, allow_credential_issuance=True)
    credential = owner_credential.issue_approver_credential("owner", {"owner"})
    off = queue.issue_card(
        credential=credential, action_class="cleanup", tier="routine", matcher=CardMatcher(command=("rm",))
    )
    on = queue.issue_card(
        credential=credential,
        action_class="package_install",
        tier="sensitive",
        matcher=CardMatcher(command=("pip", "install"), network="on"),
    )
    with policy_scope(resolve("read", grant(folder, off, shell=False)), approval_actor="read-only") as requester:
        assert requester
        assert isinstance(
            queue.submit_typed_request(request(requester, cleanup()), requester=requester), ApprovalTicket
        )
    with policy_scope(resolve("edit-and-run", grant(folder, on)), approval_actor="offline") as requester:
        assert requester
        install = CardAction(argv=("pip", "install", "test-package"), cwd=Path("."), paths=(Path("."),), network="on")
        req = request(requester, install, action_class="package_install", tier="sensitive")
        assert isinstance(queue.submit_typed_request(req, requester=requester), ApprovalTicket)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        link = folder / "scratch"
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(outside), str(link))
        else:
            link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"creating a directory symlink is unavailable: {exc}")
    with policy_scope(resolve("edit-and-run", grant(folder, off)), approval_actor="worker") as requester:
        assert requester
        assert isinstance(
            queue.submit_typed_request(request(requester, cleanup()), requester=requester), ApprovalTicket
        )


def test_owner_templates_copy_held_cards_and_child_without_grant_inherits_no_cards(tmp_path: Path) -> None:
    store = Store()
    queue = ApprovalQueue(store)
    owner_credential = ApprovalQueue(store, allow_credential_issuance=True)
    credential = owner_credential.issue_approver_credential("owner", {"owner"})
    matcher = CardMatcher(command=("rm",))
    with pytest.raises(PermissionError, match="owner"):
        queue.issue_card(credential="worker", action_class="cleanup", tier="routine", matcher=matcher)
    original = queue.issue_card(
        credential=credential, action_class="cleanup", tier="routine", matcher=matcher, transferable=True
    )
    queue.set_card_template("coder", (original.card_id,), credential=credential)
    parent = resolve("edit-and-run", grant(tmp_path, original))
    empty = resolve("edit-and-run", grant(tmp_path))
    assert parent.digest != empty.digest
    with policy_scope(parent, approval_actor="owner") as owner:
        assert owner
        [copy] = queue.copy_card_template("coder", holder=owner)
        assert copy.provenance == (original.card_id,)
        assert not copy.transferable
        assert queue.copy_card_template("reviewer", holder=owner) == ()
        with policy_scope(None, approval_actor="worker") as child:
            assert child and child.effective_policy
            assert child.effective_policy.digest == empty.digest
            assert queue.copy_card_template("coder", holder=child) == ()
            assert isinstance(queue.submit_typed_request(request(child, cleanup()), requester=child), ApprovalTicket)


def test_sibling_copies_share_upstream_use_ceiling_under_concurrency(tmp_path: Path) -> None:
    store = Store()
    queue = ApprovalQueue(store)
    owner_credential = ApprovalQueue(store, allow_credential_issuance=True)
    credential = owner_credential.issue_approver_credential("owner", {"owner"})
    original = queue.issue_card(
        credential=credential,
        action_class="cleanup",
        tier="routine",
        matcher=CardMatcher(command=("rm",)),
        max_uses=2,
        transferable=True,
    )
    owner_policy = resolve("edit-and-run", grant(tmp_path, original))
    with policy_scope(owner_policy, approval_actor="owner") as owner:
        assert owner
        copies = [queue.copy_card(original.card_id, holder=owner) for _ in range(2)]

    def submit(i: int) -> CardApproval | ApprovalTicket:
        sibling = i % len(copies)
        with (
            policy_scope(owner_policy, approval_actor="owner"),
            policy_scope(grant(tmp_path, copies[sibling]), approval_actor=f"sibling-{sibling}") as child,
        ):
            assert child
            return queue.submit_typed_request(request(child, cleanup(), attempt=i + 1), requester=child)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(10)))
    assert sum(isinstance(result, CardApproval) for result in results) == 2
    assert len(queue.list_card_approvals()) == 2
    assert len(queue.list_pending_tickets()) == 8


def test_compaction_preserves_audit_and_removes_unusable_cards(tmp_path: Path) -> None:
    clock = Clock()
    store = Store()
    queue = ApprovalQueue(store, clock=clock, allow_credential_issuance=True)
    credential = queue.issue_approver_credential("owner", {"owner"})
    original = queue.issue_card(
        credential=credential,
        action_class="cleanup",
        tier="routine",
        matcher=CardMatcher(command=("rm",)),
        transferable=True,
    )
    queue.set_card_template("coder", (original.card_id,), credential=credential)
    with policy_scope(resolve("edit-and-run", grant(tmp_path, original)), approval_actor="owner") as owner:
        assert owner
        child = queue.copy_card(original.card_id, holder=owner, transferable=True)
        with policy_scope(grant(tmp_path, child), approval_actor="worker") as worker:
            assert worker
            receipt = queue.submit_typed_request(request(worker, cleanup()), requester=worker)
            assert isinstance(receipt, CardApproval)
            assert queue.revoke_card(original.card_id, credential=credential)
            assert not queue.revoke_card(original.card_id, credential=credential)
            assert not queue.revoke_card(child.card_id, credential=credential)
            with pytest.raises(PolicyWidening):
                queue.copy_card(child.card_id, holder=worker)
            assert isinstance(queue.submit_typed_request(request(worker, cleanup()), requester=worker), ApprovalTicket)
        with pytest.raises(PolicyWidening):
            queue.copy_card(original.card_id, holder=owner)
        assert queue.copy_card_template("coder", holder=owner) == ()

    with queue._anchored() as anchored:
        raw = anchored.read(queue._cards._key)
    state = _CardState.model_validate(raw)
    assert set(raw) == {"cards", "revoked", "used", "approvals", "templates"}
    assert not state.live(original, clock.now)
    assert not state.live(child, clock.now)
    assert state.cards == {} and state.revoked == set() and state.used == {}
    assert state.templates["coder"] == ()
    assert state.approvals == [receipt] == queue.list_card_approvals()


def test_next_update_compacts_large_registry_in_one_cas() -> None:
    from contextlib import nullcontext
    from unittest.mock import MagicMock

    clock = Clock()
    live = ApprovalCard(card_id="live", action_class="cleanup", tier="routine", matcher=CardMatcher(command=("rm",)))
    revoked = {f"revoked-{i}" for i in range(2000)}
    expired = {f"expired-{i}" for i in range(2000)}
    cards = {"live": live}
    cards.update((key, live.model_copy(update={"card_id": key})) for key in revoked)
    cards.update((key, live.model_copy(update={"card_id": key, "expires_at": clock.now})) for key in expired)
    used_up = live.model_copy(update={"card_id": "used", "max_uses": 1})
    orphan = live.model_copy(update={"card_id": "orphan", "provenance": ("missing",)})
    descendant = live.model_copy(update={"card_id": "descendant", "provenance": ("orphan",)})
    cards.update((card.card_id, card) for card in (used_up, orphan, descendant))
    state = _CardState(
        cards=cards,
        revoked=revoked,
        used={"used": 1, "live": 2},
        templates={"coder": ("live", "used", next(iter(revoked)))},
    )
    raw = state.model_dump(mode="json")
    store = MagicMock()
    store.read.return_value = raw
    store.compare_and_swap.return_value = True
    registry = _CardRegistry(lambda: nullcontext(store), "cards", clock)
    new = live.model_copy(update={"card_id": "new"})
    assert registry.issue(new) == new
    store.read.assert_called_once_with("cards")
    store.compare_and_swap.assert_called_once()
    key, expected, saved = store.compare_and_swap.call_args.args
    assert key == "cards" and expected == raw
    compacted = _CardState.model_validate(saved)
    assert set(compacted.cards) == {"live", "new"}
    assert compacted.revoked == set() and compacted.used == {"live": 2}
    assert compacted.templates == {"coder": ("live",)}
    assert not compacted.live(descendant, clock.now)


def test_compaction_retries_against_the_winning_cas_state() -> None:
    from contextlib import nullcontext
    from unittest.mock import MagicMock

    clock = Clock()
    card = ApprovalCard(card_id="new", action_class="cleanup", tier="routine", matcher=CardMatcher(command=("rm",)))
    dead = card.model_copy(update={"card_id": "dead"})
    concurrent = card.model_copy(update={"card_id": "concurrent"})
    first = _CardState(cards={"dead": dead}, revoked={"dead"}).model_dump(mode="json")
    second = _CardState(cards={"dead": dead, "concurrent": concurrent}, revoked={"dead"}).model_dump(mode="json")
    store = MagicMock()
    store.read.side_effect = [first, second]
    store.compare_and_swap.side_effect = [False, True]
    registry = _CardRegistry(lambda: nullcontext(store), "cards", clock)
    registry.issue(card)
    assert store.compare_and_swap.call_count == 2
    _, expected, saved = store.compare_and_swap.call_args.args
    assert expected == second
    assert set(saved["cards"]) == {"new", "concurrent"}
    assert saved["revoked"] == []
