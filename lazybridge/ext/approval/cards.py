"""Store-backed issuance, delegation and use of approval cards."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from lazybridge.policy import ApprovalCard, CardAction, CardMatcher, PolicyWidening
from lazybridge.policy.cards import card_matches, validate_card_copy

if TYPE_CHECKING:
    from lazybridge import Store
    from lazybridge.policy import ApprovalTier, PolicyContext

CardTemplate = Literal["coder", "reviewer", "researcher"]
_Result = TypeVar("_Result")
ARCHIVE_AFTER = 100
# A reservation older than this was abandoned (the process died before confirming) and frees its use.
RESERVATION_TTL = timedelta(minutes=5)
_HOLDER_PREFIX = "holder:"


class CardApproval(BaseModel):
    """Audit receipt for a card use committed before the action may execute."""

    model_config = ConfigDict(frozen=True)

    card_id: str
    chain: tuple[str, ...]
    task_id: str
    action: str
    action_class: str
    actor: str
    invocation_id: str
    policy_digest: str
    request_digest: str
    approved_at: datetime
    # Unique per committed card use; None only on receipts written before archiving existed.
    receipt_id: str | None = None


class CardProposal(BaseModel):
    """A suggested restriction set for a parent to consider delegating."""

    model_config = ConfigDict(frozen=True)

    action_class: str
    tier: Literal["routine", "sensitive"]
    matcher: CardMatcher
    task_id: str
    requester: str
    project_id: str | None = None
    occurrences: int = 1


class _CardState(BaseModel):
    cards: dict[str, ApprovalCard] = Field(default_factory=dict)
    revoked: set[str] = Field(default_factory=set)
    used: dict[str, int] = Field(default_factory=dict)
    approvals: list[CardApproval] = Field(default_factory=list)
    templates: dict[CardTemplate, tuple[str, ...]] = Field(default_factory=dict)

    def live(self, card: ApprovalCard, now: datetime) -> bool:
        return all(self.available(card_id, now) for card_id in card.chain)

    def available(self, card_id: str, now: datetime, shared_uses: int = 0) -> bool:
        card = self.cards.get(card_id)
        return (
            card is not None
            and card_id not in self.revoked
            and (card.expires_at is None or card.expires_at > now)
            and (card.max_uses is None or max(self.used.get(card_id, 0), shared_uses) < card.max_uses)
        )

    def compact(self, now: datetime, *, live: Callable[[ApprovalCard, datetime], bool] | None = None) -> None:
        is_live = live or self.live
        while dead := {card_id for card_id, card in self.cards.items() if not is_live(card, now)}:
            for card_id in dead:
                del self.cards[card_id]
            self.revoked.difference_update(dead)
            for card_id in dead:
                self.used.pop(card_id, None)
            self.templates = {
                name: tuple(card_id for card_id in card_ids if card_id not in dead)
                for name, card_ids in self.templates.items()
            }


class _HolderState(_CardState):
    # Pending receipts reserve local uses during validation. Only confirmed
    # receipts enter approvals or the archive.
    pending: list[CardApproval] = Field(default_factory=list)

    def drop_abandoned(self, now: datetime) -> None:
        self.pending = [
            approval
            for approval in self.pending
            if approval.card_id in self.cards and approval.approved_at > now - RESERVATION_TTL
        ]

    def at_capacity(self, card: ApprovalCard, now: datetime) -> bool:
        if card.max_uses is None:
            return False
        self.drop_abandoned(now)
        reserved = sum(approval.card_id == card.card_id for approval in self.pending)
        return self.used.get(card.card_id, 0) + reserved >= card.max_uses


class _CardDocuments:
    """Read each provenance document once per CAS attempt."""

    def __init__(self, store: Store, main_key: str) -> None:
        self.store = store
        self.main_key = main_key
        self.raw: dict[str, dict | None] = {}
        self.states: dict[str, _CardState] = {}

    def key(self, card_id: str) -> str:
        holder_suffix, _, _ = card_id.rpartition(":")
        return f"{self.main_key}{holder_suffix}" if holder_suffix.startswith(_HOLDER_PREFIX) else self.main_key

    def read(self, key: str) -> _CardState:
        if key not in self.states:
            raw = self.store.read(key)
            model = _CardState if key == self.main_key else _HolderState
            self.raw[key] = raw
            self.states[key] = model.model_validate(raw) if raw is not None else model()
        return self.states[key]

    def holder(self, key: str) -> _HolderState:
        state = self.read(key)
        assert isinstance(state, _HolderState)
        return state

    def state(self, card_id: str) -> _CardState:
        return self.read(self.key(card_id))

    def live(self, card: ApprovalCard, now: datetime) -> bool:
        main = self.read(self.main_key)
        return all(self.state(card_id).available(card_id, now, main.used.get(card_id, 0)) for card_id in card.chain)


class _CardRegistry:
    def __init__(
        self, anchored: Callable[[], AbstractContextManager[Store]], key: str, clock: Callable[[], datetime]
    ) -> None:
        self._anchored = anchored
        self._key = key
        self._audit_prefix = f"{key}audit:"
        self._clock = clock

    def _approval_key(self, approval: CardApproval) -> str:
        # Two uses of one card for one request at the same instant are distinct receipts: the unique
        # receipt id keeps them apart. Older receipts (no id) keep their timestamp-card-digest key.
        identity = approval.receipt_id or f"{approval.card_id}-{approval.request_digest[:16]}"
        return f"{self._audit_prefix}{approval.approved_at.isoformat()}-{identity}"

    def _archive(self, store: Store, approvals: list[CardApproval]) -> None:
        for approval in approvals:
            store.write(self._approval_key(approval), approval.model_dump(mode="json"))

    def _save(self, documents: _CardDocuments, key: str) -> bool:
        store = documents.store
        state = documents.read(key)
        if len(state.approvals) > ARCHIVE_AFTER:
            self._archive(store, state.approvals[:-ARCHIVE_AFTER])
            del state.approvals[:-ARCHIVE_AFTER]
        if not store.compare_and_swap(key, documents.raw[key], state.model_dump(mode="json")):
            return False
        self._retire_empty_holder(documents, key)
        return True

    def _retire_empty_holder(self, documents: _CardDocuments, key: str) -> None:
        if key != self._key and not (state := documents.read(key)).cards:
            # Issuance never reuses a holder key. Once empty, a shard cannot gain
            # cards again, so its committed receipts can be archived before deletion.
            self._archive(documents.store, state.approvals)
            documents.store.delete(key)

    def _compact(self, documents: _CardDocuments, key: str, now: datetime) -> None:
        state = documents.read(key)
        if key == self._key:
            state.compact(now)
            # The shared budget CAS also tracks bounded copies in holder shards.
            for card_id in tuple(state.used):
                if card_id not in documents.state(card_id).cards:
                    del state.used[card_id]
        else:
            holder = documents.holder(key)
            holder.compact(now, live=documents.live)
            holder.drop_abandoned(now)

    def _compact_holder(self, documents: _CardDocuments, key: str, now: datetime) -> None:
        if key == self._key or documents.raw[key] is None:
            return
        state = documents.read(key)
        before = len(state.cards)
        self._compact(documents, key, now)
        if len(state.cards) < before:
            self._save(documents, key)
        elif not before:
            self._retire_empty_holder(documents, key)

    def _update(self, change: Callable[[_CardState], _Result | None]) -> _Result | None:
        with self._anchored() as store:
            while True:
                documents = _CardDocuments(store, self._key)
                result = change(documents.read(self._key))
                if result is None:
                    return result
                self._compact(documents, self._key, self._clock())
                if self._save(documents, self._key):
                    return result

    def issue(self, card: ApprovalCard) -> ApprovalCard:
        def add(state: _CardState) -> ApprovalCard:
            if card.expires_at is not None and card.expires_at <= self._clock():
                raise ValueError("card must expire in the future")
            state.cards[card.card_id] = card
            return card

        issued = self._update(add)
        assert issued is not None
        return issued

    def _issue_hand(
        self, store: Store, holder: PolicyContext, cards: tuple[ApprovalCard, ...]
    ) -> tuple[ApprovalCard, ...]:
        # The supplied invocation delegates one hand to a child. A unique hand
        # suffix keeps children of the same invocation in separate documents.
        holder_key = f"{quote(holder.invocation_id, safe='')}-{uuid.uuid4().hex}"
        suffix = f"{_HOLDER_PREFIX}{holder_key}"
        cards = tuple(card.model_copy(update={"card_id": f"{suffix}:{card.card_id}"}) for card in cards)
        state = _HolderState(cards={card.card_id: card for card in cards})
        store.write(f"{self._key}{suffix}", state.model_dump(mode="json"))
        return cards

    def copy(self, source: ApprovalCard, child: ApprovalCard, holder: PolicyContext) -> ApprovalCard:
        validate_card_copy(source, child)

        with self._anchored() as store:
            documents = _CardDocuments(store, self._key)
            now = self._clock()
            if documents.state(source.card_id).cards.get(source.card_id) != source or not documents.live(source, now):
                raise PolicyWidening("source card is unavailable")
            if child.expires_at is not None and child.expires_at <= now:
                raise ValueError("card must expire in the future")
            (copied,) = self._issue_hand(store, holder, (child,))
            return copied

    def revoke(self, card_id: str) -> bool:
        with self._anchored() as store:
            while True:
                documents = _CardDocuments(store, self._key)
                key = documents.key(card_id)
                state = documents.read(key)
                if documents.raw[key] is None:
                    return False
                now = self._clock()
                card = state.cards.get(card_id)
                revoked = (
                    card is not None
                    and card_id not in state.revoked
                    and (key == self._key or documents.live(card, now))
                )
                if key == self._key and not revoked:
                    return False
                if revoked:
                    state.revoked.add(card_id)
                before = len(state.cards)
                self._compact(documents, key, now)
                if len(state.cards) == before:
                    self._retire_empty_holder(documents, key)
                    return False
                if self._save(documents, key):
                    return revoked

    def set_template(self, name: CardTemplate, card_ids: tuple[str, ...]) -> None:
        def set_hand(state: _CardState) -> bool:
            now = self._clock()
            for card_id in card_ids:
                card = state.cards.get(card_id)
                if card is None or card.provenance or not card.transferable or not state.live(card, now):
                    raise PolicyWidening("templates require live transferable original cards")
            state.templates[name] = tuple(dict.fromkeys(card_ids))
            return True

        self._update(set_hand)

    def copy_template(
        self, name: CardTemplate, hand: tuple[ApprovalCard, ...], holder: PolicyContext
    ) -> tuple[ApprovalCard, ...]:
        with self._anchored() as store:
            documents = _CardDocuments(store, self._key)
            now = self._clock()
            roots = documents.read(self._key).templates.get(name, ())
            sources = tuple(
                card
                for card in hand
                if card.chain[0] in roots
                and card.transferable
                and documents.state(card.card_id).cards.get(card.card_id) == card
                and documents.live(card, now)
            )
            if not sources:
                return ()
            copies = tuple(
                card.model_copy(update={"card_id": uuid.uuid4().hex, "provenance": card.chain, "transferable": False})
                for card in sources
            )
            return self._issue_hand(store, holder, copies)

    def approve(
        self,
        requester: PolicyContext,
        action: CardAction,
        paths: tuple[Path, ...],
        *,
        action_name: str,
        action_class: str,
        tier: ApprovalTier,
        task_id: str,
        request_digest: str,
    ) -> CardApproval | None:
        policy = requester.effective_policy
        if policy is None or not policy.approval_cards or tier == "external":
            return None

        with self._anchored() as store:
            for card in policy.approval_cards:
                if not card_matches(card, action, paths, action_class=action_class, tier=tier, task_id=task_id):
                    continue
                while True:
                    documents = _CardDocuments(store, self._key)
                    holder_key = documents.key(card.card_id)
                    holder = documents.read(holder_key)
                    now = self._clock()
                    if holder.cards.get(card.card_id) != card or not documents.live(card, now):
                        self._compact_holder(documents, holder_key, now)
                        break
                    budget_ids = [
                        card_id
                        for card_id in card.chain
                        if documents.state(card_id).cards[card_id].max_uses is not None
                    ]
                    # Bounded ancestors and bounded transferable copies share
                    # a single CAS, whether used directly or via a descendant.
                    shared_budget = bool(budget_ids) and (card.transferable or budget_ids[0] != card.card_id)
                    key = self._key if shared_budget else holder_key
                    if key != self._key and documents.holder(key).at_capacity(card, now):
                        break
                    approval = CardApproval(
                        card_id=card.card_id,
                        chain=card.chain,
                        task_id=task_id,
                        action=action_name,
                        action_class=action_class,
                        actor=requester.actor,
                        invocation_id=requester.invocation_id,
                        policy_digest=policy.digest,
                        request_digest=request_digest,
                        approved_at=now,
                        receipt_id=uuid.uuid4().hex,
                    )
                    if key == self._key:
                        state = documents.read(key)
                        for card_id in budget_ids:
                            state.used[card_id] = state.used.get(card_id, 0) + 1
                        state.approvals.append(approval)
                    else:
                        documents.holder(key).pending.append(approval)
                    self._compact(documents, key, now)
                    if not self._save(documents, key):
                        continue
                    if key == self._key:
                        if key != holder_key:
                            self._compact_holder(documents, holder_key, now)
                        return approval

                    # The holder CAS cannot detect revocations elsewhere in the
                    # chain. Read every document afresh before confirming the use.
                    current = _CardDocuments(store, self._key)
                    for card_id in reversed(card.chain):
                        current.state(card_id)
                    approved = current.live(card, self._clock())
                    self._finish_holder_approval(current, key, approval, approved=approved)
                    return approval if approved else None
        return None

    def _finish_holder_approval(
        self, documents: _CardDocuments, key: str, approval: CardApproval, *, approved: bool
    ) -> None:
        while True:
            holder = documents.holder(key)
            if not holder.cards:
                if approved:
                    # A later revocation can retire the holder after validation.
                    self._archive(documents.store, [approval])
                return

            reserved = len(holder.pending)
            holder.pending = [receipt for receipt in holder.pending if receipt.receipt_id != approval.receipt_id]
            if approved:
                card = holder.cards.get(approval.card_id)
                if card is not None and card.max_uses is not None:
                    holder.used[card.card_id] = holder.used.get(card.card_id, 0) + 1
                holder.approvals.append(approval)
            elif len(holder.pending) == reserved:
                return

            self._compact(documents, key, self._clock())
            if self._save(documents, key):
                return
            documents = _CardDocuments(documents.store, self._key)

    def approvals(self) -> list[CardApproval]:
        with self._anchored() as store:
            documents = _CardDocuments(store, self._key)
            approvals = {self._approval_key(approval): approval for approval in documents.read(self._key).approvals}
            for _, raw in store.items(prefix=f"{self._key}{_HOLDER_PREFIX}"):
                state = _CardState.model_validate(raw)
                approvals.update((self._approval_key(approval), approval) for approval in state.approvals)
            # Read archives last: a receipt can only move from a live document
            # to the archive, and the archive write precedes its removal.
            archived = store.items(prefix=self._audit_prefix)
        approvals.update((key, CardApproval.model_validate(value)) for key, value in archived)
        return sorted(approvals.values(), key=lambda approval: approval.approved_at)
