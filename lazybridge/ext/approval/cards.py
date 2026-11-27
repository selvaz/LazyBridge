"""Store-backed issuance, delegation and use of approval cards."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from lazybridge.policy import ApprovalCard, CardAction, CardMatcher, PolicyWidening
from lazybridge.policy.cards import card_matches, validate_card_copy

if TYPE_CHECKING:
    from lazybridge import Store
    from lazybridge.policy import ApprovalTier, PolicyContext

CardTemplate = Literal["coder", "reviewer", "researcher"]
_MIGRATED = "per-card-v1"


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


class _LegacyCardState(BaseModel):
    cards: dict[str, ApprovalCard] = Field(default_factory=dict)
    revoked: set[str] = Field(default_factory=set)
    used: dict[str, int] = Field(default_factory=dict)
    approvals: list[CardApproval] = Field(default_factory=list)
    templates: dict[CardTemplate, tuple[str, ...]] = Field(default_factory=dict)


class _CardRecord(BaseModel):
    card: ApprovalCard
    revoked: bool = False
    used: int = 0

    def retired(self, now: datetime) -> bool:
        return self.revoked or (self.card.expires_at is not None and self.card.expires_at <= now)

    def live(self, now: datetime) -> bool:
        return not self.retired(now) and (self.card.max_uses is None or self.used < self.card.max_uses)


class _CardRegistry:
    def __init__(
        self, anchored: Callable[[], AbstractContextManager[Store]], key: str, clock: Callable[[], datetime]
    ) -> None:
        self._anchored = anchored
        self._key = key
        self._card_prefix = f"{key}:card:"
        self._receipt_prefix = f"{key}:receipt:"
        self._clock = clock

    def _card_key(self, card_id: str) -> str:
        return self._card_prefix + quote(card_id, safe="")

    def _migrate(self, store: Store) -> None:
        # Replace the old document and publish all its records in one transaction.
        # The marker also makes an older registry fail closed if it reads this key.
        while True:
            raw = store.read(self._key)
            if raw == _MIGRATED:
                return
            state = _LegacyCardState.model_validate(raw) if raw is not None else _LegacyCardState()
            updates: dict[str, object] = {self._key: _MIGRATED}
            updates.update(
                {
                    self._card_key(card_id): _CardRecord(
                        card=card, revoked=card_id in state.revoked, used=state.used.get(card_id, 0)
                    ).model_dump(mode="json")
                    for card_id, card in state.cards.items()
                }
            )
            updates.update({f"{self._key}:template:{name}": list(ids) for name, ids in state.templates.items()})
            updates.update(
                {
                    f"{self._receipt_prefix}0-{index:020d}": receipt.model_dump(mode="json")
                    for index, receipt in enumerate(state.approvals)
                }
            )
            if store.compare_and_swap_many({self._key: raw}, updates):
                return

    def _live_chain(
        self, store: Store, card: ApprovalCard, now: datetime
    ) -> dict[str, tuple[dict, _CardRecord]] | None:
        chain = {}
        for card_id in card.chain:
            key = self._card_key(card_id)
            raw = store.read(key)
            if raw is None:
                return None
            record = _CardRecord.model_validate(raw)
            if not record.live(now) or (card_id == card.card_id and record.card != card):
                return None
            chain[key] = raw, record
        return chain

    def issue(self, card: ApprovalCard) -> ApprovalCard:
        with self._anchored() as store:
            self._migrate(store)
            if card.expires_at is not None and card.expires_at <= self._clock():
                raise ValueError("card must expire in the future")
            key = self._card_key(card.card_id)
            if not store.compare_and_swap(key, None, _CardRecord(card=card).model_dump(mode="json")):
                raise ValueError("card ID already exists")
        return card

    def copy(self, source: ApprovalCard, child: ApprovalCard) -> ApprovalCard:
        validate_card_copy(source, child)
        key = self._card_key(child.card_id)
        updates = {key: _CardRecord(card=child).model_dump(mode="json")}
        with self._anchored() as store:
            self._migrate(store)
            while True:
                now = self._clock()
                chain = self._live_chain(store, source, now)
                if chain is None:
                    raise PolicyWidening("source card is unavailable")
                if child.expires_at is not None and child.expires_at <= now:
                    raise ValueError("card must expire in the future")
                expected: dict[str, object] = {source_key: raw for source_key, (raw, _) in chain.items()}
                expected[key] = None
                if store.compare_and_swap_many(expected, updates):
                    return child
                if store.read(key) is not None:
                    raise ValueError("card ID already exists")

    def revoke(self, card_id: str) -> bool:
        key = self._card_key(card_id)
        with self._anchored() as store:
            self._migrate(store)
            while (raw := store.read(key)) is not None:
                record = _CardRecord.model_validate(raw)
                if record.revoked:
                    return False
                record.revoked = True
                if store.compare_and_swap(key, raw, record.model_dump(mode="json")):
                    return True
        return False

    def set_template(self, name: CardTemplate, card_ids: tuple[str, ...]) -> None:
        key = f"{self._key}:template:{name}"
        ids = list(dict.fromkeys(card_ids))
        with self._anchored() as store:
            self._migrate(store)
            while True:
                expected = {key: store.read(key)}
                now = self._clock()
                for card_id in ids:
                    card_key = self._card_key(card_id)
                    raw = store.read(card_key)
                    record = _CardRecord.model_validate(raw) if raw is not None else None
                    if record is None or record.card.provenance or not record.card.transferable or not record.live(now):
                        raise PolicyWidening("templates require live transferable original cards")
                    expected[card_key] = raw
                if store.compare_and_swap_many(expected, {key: ids}):
                    return

    def copy_template(self, name: CardTemplate, hand: tuple[ApprovalCard, ...]) -> tuple[ApprovalCard, ...]:
        key = f"{self._key}:template:{name}"
        with self._anchored() as store:
            self._migrate(store)
            while roots := store.read(key):
                now = self._clock()
                expected = {key: roots}
                updates = {}
                copies = []
                for card in hand:
                    if card.chain[0] not in roots or not card.transferable:
                        continue
                    chain = self._live_chain(store, card, now)
                    if chain is None:
                        continue
                    child = card.model_copy(
                        update={"card_id": uuid.uuid4().hex, "provenance": card.chain, "transferable": False}
                    )
                    child_key = self._card_key(child.card_id)
                    expected.update((source_key, raw) for source_key, (raw, _) in chain.items())
                    expected[child_key] = None
                    updates[child_key] = _CardRecord(card=child).model_dump(mode="json")
                    copies.append(child)
                if not copies:
                    if store.read(key) == roots:
                        return ()
                elif store.compare_and_swap_many(expected, updates):
                    return tuple(copies)
        return ()

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
            self._migrate(store)
            for card in policy.approval_cards:
                if not card_matches(card, action, paths, action_class=action_class, tier=tier, task_id=task_id):
                    continue
                while True:
                    now = self._clock()
                    chain = self._live_chain(store, card, now)
                    if chain is None:
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
                    )
                    receipt_key = f"{self._receipt_prefix}1-{time_ns():020d}-{uuid.uuid4().hex}"
                    expected: dict[str, object] = {key: raw for key, (raw, _) in chain.items()}
                    updates = {}
                    for key, (_, record) in chain.items():
                        record.used += 1
                        updates[key] = record.model_dump(mode="json")
                    expected[receipt_key] = None
                    updates[receipt_key] = approval.model_dump(mode="json")
                    if store.compare_and_swap_many(expected, updates):
                        return approval
        return None

    def approvals(self) -> list[CardApproval]:
        with self._anchored() as store:
            self._migrate(store)
            items = store.items(prefix=self._receipt_prefix)
        return [
            CardApproval.model_validate(raw)
            for key, raw in sorted(items)
            if ":" not in key[len(self._receipt_prefix) :]
        ]

    def prune_copies(self) -> int:
        with self._anchored() as store:
            self._migrate(store)
            now = self._clock()
            records = {
                key: (raw, _CardRecord.model_validate(raw))
                for key, raw in store.items(prefix=self._card_prefix)
                if ":" not in key[len(self._card_prefix) :]
            }
            unretired = {key for key, (_, record) in records.items() if not record.retired(now)}
            removed = 0
            for key, (raw, record) in records.items():
                if record.card.provenance and any(
                    self._card_key(card_id) not in unretired for card_id in record.card.chain
                ):
                    # Retirement is final. The CAS avoids racing a concurrent
                    # deletion or deleting a row that changed after this scan.
                    removed += store.compare_and_swap_many({key: raw}, {}, delete=(key,))
            return removed
