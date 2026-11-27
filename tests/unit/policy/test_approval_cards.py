from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Lock

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


def test_approvals_on_disjoint_card_chains_each_use_one_cas(tmp_path: Path) -> None:
    class RacingStore(Store):
        def __init__(self):
            super().__init__()
            self.barrier = Barrier(2)
            self.attempt_lock = Lock()
            self.attempts = []
            self.armed = False

        def compare_and_swap_many(self, expected, updates, **kwargs):
            if self.armed:
                with self.attempt_lock:
                    self.attempts.append(set(expected))
                    first_attempt = len(self.attempts) <= 2
                if first_attempt:
                    self.barrier.wait(timeout=5)
            return super().compare_and_swap_many(expected, updates, **kwargs)

    store = RacingStore()
    queue = ApprovalQueue(store, prefix="isolated", allow_credential_issuance=True)
    credential = queue.issue_approver_credential("owner", {"owner"})
    originals = [
        queue.issue_card(
            credential=credential,
            action_class="cleanup",
            tier="routine",
            matcher=CardMatcher(command=("rm",)),
            transferable=True,
        )
        for _ in range(2)
    ]
    parent = resolve("edit-and-run", grant(tmp_path, *originals))
    with policy_scope(parent, approval_actor="owner") as owner:
        assert owner
        copies = [queue.copy_card(original.card_id, holder=owner) for original in originals]

    def submit(copy: ApprovalCard) -> CardApproval | ApprovalTicket:
        with policy_scope(parent), policy_scope(grant(tmp_path, copy), approval_actor="worker") as worker:
            assert worker
            return queue.submit_typed_request(request(worker, cleanup()), requester=worker)

    store.armed = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, copies))
    assert all(isinstance(result, CardApproval) for result in results)
    first, second = store.attempts
    assert first.isdisjoint(second)
    prefix = "approval-cards:isolated:card:"
    assert {frozenset(key for key in attempt if key.startswith(prefix)) for attempt in store.attempts} == {
        frozenset(prefix + card_id for card_id in copy.chain) for copy in copies
    }


def test_legacy_card_document_migrates_lazily_and_only_once(tmp_path: Path) -> None:
    clock = Clock()
    with Store(db=str(tmp_path / "legacy.db")) as store:
        queue = ApprovalQueue(store, prefix="legacy", clock=clock, allow_credential_issuance=True)
        credential = queue.issue_approver_credential("owner", {"owner"})
        original = ApprovalCard(
            card_id="original",
            action_class="cleanup",
            tier="routine",
            matcher=CardMatcher(command=("rm",)),
            transferable=True,
            max_uses=2,
        )
        old_copy = original.model_copy(update={"card_id": "old-copy", "provenance": original.chain})
        revoked = old_copy.model_copy(update={"card_id": "revoked"})
        receipt = CardApproval(
            card_id=old_copy.card_id,
            chain=old_copy.chain,
            task_id="previous-job",
            action="cleanup",
            action_class="cleanup",
            actor="previous-worker",
            invocation_id="previous-invocation",
            policy_digest="previous-policy",
            request_digest="previous-request",
            approved_at=clock.now,
        )
        key = "approval-cards:legacy"
        store.write(
            key,
            {
                "cards": {card.card_id: card.model_dump(mode="json") for card in (original, old_copy, revoked)},
                "revoked": [revoked.card_id],
                "used": {card_id: 1 for card_id in old_copy.chain},
                "templates": {"coder": [original.card_id]},
                "approvals": [receipt.model_dump(mode="json")],
            },
        )
        assert store.items(prefix=f"{key}:card:") == []

        assert queue.list_card_approvals() == [receipt]
        assert store.read(f"{key}:card:old-copy")["used"] == 1
        assert store.read(f"{key}:card:revoked")["revoked"] is True
        assert store.read(f"{key}:template:coder") == [original.card_id]

        with policy_scope(resolve("edit-and-run", grant(tmp_path, original)), approval_actor="owner") as owner:
            assert owner
            [copy] = queue.copy_card_template("coder", holder=owner)
            with policy_scope(grant(tmp_path, copy), approval_actor="worker") as worker:
                assert worker
                approved = queue.submit_typed_request(request(worker, cleanup()), requester=worker)
                assert isinstance(approved, CardApproval)
                assert isinstance(
                    queue.submit_typed_request(request(worker, cleanup(), attempt=2), requester=worker), ApprovalTicket
                )

        assert queue.prune_card_copies() == 1
        queue.set_card_template("coder", (), credential=credential)
        reopened = ApprovalQueue(store, prefix="legacy", clock=clock)
        assert reopened.list_card_approvals() == [receipt, approved]
        assert store.read(f"{key}:card:original")["used"] == 2
        assert store.read(f"{key}:card:revoked") is None
        assert store.read(f"{key}:template:coder") == []


def test_prune_removes_only_retired_copies_and_keeps_receipts(tmp_path: Path) -> None:
    clock = Clock()
    store = Store()
    queue = ApprovalQueue(store, prefix="pruning", clock=clock, allow_credential_issuance=True)
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
        expired = queue.copy_card(original.card_id, holder=owner, expires_at=clock.now + timedelta(hours=1))
        revoked = queue.copy_card(original.card_id, holder=owner, transferable=True)
        [live] = queue.copy_card_template("coder", holder=owner)
        with policy_scope(grant(tmp_path, revoked), approval_actor="worker") as worker:
            assert worker
            leaf = queue.copy_card(revoked.card_id, holder=worker)
            with policy_scope(grant(tmp_path, leaf), approval_actor="sub-worker") as child:
                assert child
                receipt = queue.submit_typed_request(request(child, cleanup()), requester=child)
                assert isinstance(receipt, CardApproval)

    assert queue.prune_card_copies() == 0
    assert queue.revoke_card(revoked.card_id, credential=credential)
    assert expired.expires_at is not None
    clock.now = expired.expires_at
    assert queue.prune_card_copies() == 3
    assert queue.prune_card_copies() == 0
    prefix = "approval-cards:pruning:card:"
    assert {key for key, _ in store.items(prefix=prefix)} == {prefix + original.card_id, prefix + live.card_id}
    assert store.read("approval-cards:pruning:template:coder") == [original.card_id]
    assert queue.list_card_approvals() == [receipt]


@pytest.mark.parametrize("change", ["template", "revocation"])
def test_template_copy_rechecks_template_and_source_at_commit(tmp_path, monkeypatch, change) -> None:
    store = Store()
    queue = ApprovalQueue(store, prefix="template-race", allow_credential_issuance=True)
    credential = queue.issue_approver_credential("owner", {"owner"})
    original = queue.issue_card(
        credential=credential,
        action_class="cleanup",
        tier="routine",
        matcher=CardMatcher(command=("rm",)),
        transferable=True,
    )
    queue.set_card_template("coder", (original.card_id,), credential=credential)
    swap = store.compare_and_swap_many

    def change_before_copy_commits(expected, updates, **kwargs):
        monkeypatch.setattr(store, "compare_and_swap_many", swap)
        if change == "template":
            queue.set_card_template("coder", (), credential=credential)
        else:
            assert queue.revoke_card(original.card_id, credential=credential)
        return swap(expected, updates, **kwargs)

    monkeypatch.setattr(store, "compare_and_swap_many", change_before_copy_commits)
    with policy_scope(resolve("edit-and-run", grant(tmp_path, original)), approval_actor="owner") as owner:
        assert owner
        assert queue.copy_card_template("coder", holder=owner) == ()
    prefix = "approval-cards:template-race:card:"
    assert [key for key, _ in store.items(prefix=prefix)] == [prefix + original.card_id]


def test_template_copy_rolls_back_the_entire_hand_on_a_write_failure(tmp_path: Path) -> None:
    db = str(tmp_path / "templates.db")
    with Store(db=db) as store, closing(sqlite3.connect(db)) as conn:
        queue = ApprovalQueue(store, prefix="template-test", allow_credential_issuance=True)
        credential = queue.issue_approver_credential("owner", {"owner"})
        originals = [
            queue.issue_card(
                credential=credential,
                action_class="cleanup",
                tier="routine",
                matcher=CardMatcher(command=("rm",)),
                transferable=True,
            )
            for _ in range(2)
        ]
        ids = tuple(card.card_id for card in originals)
        queue.set_card_template("coder", ids, credential=credential)
        prefix = "approval-cards:template-test:card:"
        before = store.items(prefix=prefix)
        conn.execute(f"""
            CREATE TRIGGER fail_second_copy BEFORE INSERT ON store
            WHEN NEW.key GLOB '{prefix}*'
              AND (SELECT count(*) FROM store WHERE key GLOB '{prefix}*') = 3
            BEGIN
                SELECT RAISE(ABORT, 'second copy failed');
            END
        """)
        with policy_scope(resolve("edit-and-run", grant(tmp_path, *originals)), approval_actor="owner") as owner:
            assert owner
            with pytest.raises(sqlite3.IntegrityError, match="second copy failed"):
                queue.copy_card_template("coder", holder=owner)
            assert store.items(prefix=prefix) == before
            assert store.read("approval-cards:template-test:template:coder") == list(ids)
            conn.execute("DROP TRIGGER fail_second_copy")
            assert {copy.chain[0] for copy in queue.copy_card_template("coder", holder=owner)} == set(ids)
