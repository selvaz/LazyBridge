"""Store-backed issuance, delegation and use of approval cards."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from lazybridge.policy import ApprovalCard, CardAction, CardMatcher, PolicyWidening
from lazybridge.policy.cards import card_matches, validate_card_copy

if TYPE_CHECKING:
    from lazybridge import Store
    from lazybridge.policy import ApprovalTier, PolicyContext

CardTemplate = Literal["coder", "reviewer", "researcher"]
_Result = TypeVar("_Result")
ARCHIVE_AFTER = 100


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


class _CardState(BaseModel):
    cards: dict[str, ApprovalCard] = Field(default_factory=dict)
    revoked: set[str] = Field(default_factory=set)
    used: dict[str, int] = Field(default_factory=dict)
    approvals: list[CardApproval] = Field(default_factory=list)
    templates: dict[CardTemplate, tuple[str, ...]] = Field(default_factory=dict)

    def live(self, card: ApprovalCard, now: datetime) -> bool:
        for card_id in card.chain:
            source = self.cards.get(card_id)
            if source is None or card_id in self.revoked:
                return False
            if source.expires_at is not None and source.expires_at <= now:
                return False
            if source.max_uses is not None and self.used.get(card_id, 0) >= source.max_uses:
                return False
        return True

    def compact(self, now: datetime) -> None:
        while dead := {card_id for card_id, card in self.cards.items() if not self.live(card, now)}:
            for card_id in dead:
                del self.cards[card_id]
            self.revoked.difference_update(dead)
            for card_id in dead:
                self.used.pop(card_id, None)
            self.templates = {
                name: tuple(card_id for card_id in card_ids if card_id not in dead)
                for name, card_ids in self.templates.items()
            }


class _CardRegistry:
    # One CAS binds usage against every ancestor's budget, revocation and the audit record.
    def __init__(
        self, anchored: Callable[[], AbstractContextManager[Store]], key: str, clock: Callable[[], datetime]
    ) -> None:
        self._anchored = anchored
        self._key = key
        self._audit_prefix = f"{key}audit:"
        self._clock = clock

    def _approval_key(self, approval: CardApproval) -> str:
        return (
            f"{self._audit_prefix}{approval.approved_at.isoformat()}-{approval.card_id}-{approval.request_digest[:16]}"
        )

    def _update(self, change: Callable[[_CardState], _Result | None]) -> _Result | None:
        with self._anchored() as store:
            while True:
                raw = store.read(self._key)
                state = _CardState.model_validate(raw) if raw is not None else _CardState()
                result = change(state)
                if result is None:
                    return result
                state.compact(self._clock())
                if len(state.approvals) > ARCHIVE_AFTER:
                    archived = state.approvals[:-ARCHIVE_AFTER]
                    for approval in archived:
                        store.write(self._approval_key(approval), approval.model_dump(mode="json"))
                    del state.approvals[:-ARCHIVE_AFTER]
                if store.compare_and_swap(self._key, raw, state.model_dump(mode="json")):
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

    def copy(self, source: ApprovalCard, child: ApprovalCard) -> ApprovalCard:
        validate_card_copy(source, child)

        def add(state: _CardState) -> ApprovalCard:
            now = self._clock()
            if state.cards.get(source.card_id) != source or not state.live(source, now):
                raise PolicyWidening("source card is unavailable")
            if child.expires_at is not None and child.expires_at <= now:
                raise ValueError("card must expire in the future")
            state.cards[child.card_id] = child
            return child

        copied = self._update(add)
        assert copied is not None
        return copied

    def revoke(self, card_id: str) -> bool:
        def revoke(state: _CardState) -> bool | None:
            if card_id not in state.cards or card_id in state.revoked:
                return None
            state.revoked.add(card_id)
            return True

        return self._update(revoke) is True

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

    def copy_template(self, name: CardTemplate, hand: tuple[ApprovalCard, ...]) -> tuple[ApprovalCard, ...]:
        def copy(state: _CardState) -> tuple[ApprovalCard, ...] | None:
            now = self._clock()
            roots = state.templates.get(name, ())
            sources = tuple(
                card
                for card in hand
                if card.chain[0] in roots
                and card.transferable
                and state.cards.get(card.card_id) == card
                and state.live(card, now)
            )
            if not sources:
                return None
            copies = tuple(
                card.model_copy(update={"card_id": uuid.uuid4().hex, "provenance": card.chain, "transferable": False})
                for card in sources
            )
            state.cards.update((card.card_id, card) for card in copies)
            return copies

        return self._update(copy) or ()

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

        def approve(state: _CardState) -> CardApproval | None:
            now = self._clock()
            for card in policy.approval_cards:
                if (
                    state.cards.get(card.card_id) != card
                    or not state.live(card, now)
                    or not card_matches(card, action, paths, action_class=action_class, tier=tier, task_id=task_id)
                ):
                    continue
                for card_id in card.chain:
                    state.used[card_id] = state.used.get(card_id, 0) + 1
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
                state.approvals.append(approval)
                return approval
            return None

        return self._update(approve)

    def approvals(self) -> list[CardApproval]:
        with self._anchored() as store:
            raw = store.read(self._key)
            archived = store.items(prefix=self._audit_prefix)
        approvals = {key: CardApproval.model_validate(value) for key, value in archived}
        if raw is not None:
            state = _CardState.model_validate(raw)
            approvals.update((self._approval_key(approval), approval) for approval in state.approvals)
        return sorted(approvals.values(), key=lambda approval: approval.approved_at)
