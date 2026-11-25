"""Typed wire fields for version 1 publication receipts."""

from datetime import datetime, timedelta
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class _ReceiptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Issuer(_ReceiptModel):
    key_id: str
    public_key_fingerprint: str
    algorithm: Literal["ed25519"] = "ed25519"


class Repository(_ReceiptModel):
    owner: str
    name: str
    base_branch: str


class Publication(_ReceiptModel):
    head_branch: str
    head_sha: str
    base_sha: str
    pr_number: int | None
    pr_url: str | None
    draft: Literal[True] = True


class AdmissionEvidence(_ReceiptModel):
    eligible: bool
    reasons: list[str]


class PolicyEvidence(_ReceiptModel):
    profile: str
    revision: str
    digest: str
    folders: list[str]
    network: str
    tool_classes: list[str]


class WriterEvidence(_ReceiptModel):
    task_id: str
    attempt: int
    engine: str
    model: str
    launch_mode: str
    admission: AdmissionEvidence
    policy: PolicyEvidence


class ReviewerEvidence(WriterEvidence):
    independence_mode: Literal["opposite-engine", "same-provider-fresh"]
    reviewed_sha: str
    verdict: str


class DelegationEvidence(_ReceiptModel):
    parent_digest: str
    child_within_parent: Literal[True] = True


class ApprovalTarget(_ReceiptModel):
    action: Literal["open_draft_pr"] = "open_draft_pr"
    repository: str
    head_branch: str
    head_sha: str


class ApprovalEvidence(_ReceiptModel):
    ticket_id: str | None
    decided_by: str | None
    decided_at: str | None
    consumed_at: str | None
    request_digest: str | None
    target: ApprovalTarget


class CheckEvidence(_ReceiptModel):
    name: str
    command: str
    status: str
    head_sha: str


class Outcome(_ReceiptModel):
    status: Literal["published", "refused"]
    refusal_reason: str | None


class PublicationReceipt(_ReceiptModel):
    """Publication evidence with UUID4 and UTC ISO-8601 values kept as wire strings."""

    receipt_version: int = 1
    receipt_id: str
    issued_at: str
    issuer: Issuer
    repository: Repository
    publication: Publication
    writer: WriterEvidence
    reviewer: ReviewerEvidence
    delegation: DelegationEvidence
    approval: ApprovalEvidence
    checks: list[CheckEvidence]
    outcome: Outcome
    signature: str = ""

    @field_validator("receipt_id")
    @classmethod
    def _uuid4(cls, value: str) -> str:
        if UUID(value).version != 4:
            raise ValueError("receipt_id must be a UUID4")
        return value

    @field_validator("issued_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        if datetime.fromisoformat(value).utcoffset() != timedelta(0):
            raise ValueError("issued_at must be a UTC ISO-8601 timestamp")
        return value

    @model_validator(mode="after")
    def _refusal_shape(self) -> Self:
        if self.outcome.status == "refused" and not self._has_valid_refusal():
            raise ValueError(
                "A refused receipt requires a refusal reason and may carry a PR number "
                "only for pr_head_mismatch or pr_not_draft"
            )
        return self

    def _has_valid_refusal(self) -> bool:
        reason = self.outcome.refusal_reason
        return bool(reason) and (self.publication.pr_number is None or reason in ("pr_head_mismatch", "pr_not_draft"))
