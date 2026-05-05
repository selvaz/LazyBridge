"""Wave 1.1 — robust judge verdict normalisation.

Historic ``verify_with_retry`` accepted only ``startswith("approved")``;
synonyms ("yes", "ok", "allow", "pass", "looks good") were silently
rejected, making the verify loop fragile.  ``_is_approved`` now
recognises a defined set of synonyms (allowlist-style, fail-safe).
"""

from __future__ import annotations

import pytest

from lazybridge._verify import _is_approved, verify_with_retry
from lazybridge.envelope import Envelope


# ---------------------------------------------------------------------------
# _is_approved — pure normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict",
    [
        "approved",
        "Approved",
        "APPROVED: looks good",
        "approve",
        "accept",
        "accepted: clear",
        "allow",
        "allowed",
        "pass",
        "passed",
        "ok",
        "OK",
        "okay",
        "yes",
        "Yes — accept",
        "good",
        "valid",
        "  approved  ",
        True,
    ],
)
def test_is_approved_recognises_synonyms(verdict):
    assert _is_approved(verdict) is True


@pytest.mark.parametrize(
    "verdict",
    [
        "rejected",
        "rejected: missing citation",
        "reject",
        "deny",
        "denied: off-topic",
        "block",
        "blocked",
        "fail",
        "failed",
        "no",
        "No — wrong format",
        "bad",
        "invalid",
        "",
        "   ",
        None,
        False,
        # Unrecognised → fail-safe rejection.
        "maybe",
        "I'm not sure",
        "the answer might be okay",
    ],
)
def test_is_approved_rejects_negatives_and_unrecognised(verdict):
    assert _is_approved(verdict) is False


# ---------------------------------------------------------------------------
# verify_with_retry end-to-end with synonym verdicts
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal Agent-like with .run() returning a fixed envelope."""

    def __init__(self, payload: str = "ok-output"):
        self.payload = payload
        self.calls = 0

    async def run(self, env):
        self.calls += 1
        return Envelope(task=env.task, payload=self.payload)


@pytest.mark.asyncio
async def test_verify_accepts_yes_synonym_first_attempt():
    agent = _FakeAgent()

    def judge(_text: str) -> str:
        return "yes — looks good"

    env = Envelope.from_task("q")
    result = await verify_with_retry(agent, env, judge, max_verify=5)
    assert result.payload == "ok-output"
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_verify_accepts_ok_synonym_first_attempt():
    agent = _FakeAgent()

    def judge(_text: str) -> str:
        return "OK"

    env = Envelope.from_task("q")
    result = await verify_with_retry(agent, env, judge, max_verify=5)
    assert agent.calls == 1
    assert result.payload == "ok-output"


@pytest.mark.asyncio
async def test_verify_accepts_allow_synonym():
    agent = _FakeAgent()

    def judge(_text: str) -> str:
        return "allow"

    env = Envelope.from_task("q")
    await verify_with_retry(agent, env, judge, max_verify=3)
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_verify_unrecognised_verdict_treated_as_rejection():
    """A judge that fails to produce a clear approval triggers retries."""
    agent = _FakeAgent()
    verdicts = iter(["maybe?", "I'm not sure", "approved"])

    def judge(_text: str) -> str:
        return next(verdicts)

    env = Envelope.from_task("q")
    await verify_with_retry(agent, env, judge, max_verify=5)
    # Two unrecognised verdicts forced retries; third "approved" passed.
    assert agent.calls == 3


@pytest.mark.asyncio
async def test_verify_explicit_reject_prefix_overrides_ambiguity():
    agent = _FakeAgent()
    # "no good" — starts with "no", which is a reject synonym.
    verdicts = iter(["no good", "approved"])

    def judge(_text: str) -> str:
        return next(verdicts)

    env = Envelope.from_task("q")
    await verify_with_retry(agent, env, judge, max_verify=5)
    assert agent.calls == 2  # rejected then approved.


@pytest.mark.asyncio
async def test_verify_existing_approved_rejected_contract_preserved():
    """Backward-compat: the original 'approved'/'rejected' contract still works."""
    agent = _FakeAgent()
    verdicts = iter(["rejected: bad", "rejected: still bad", "approved"])

    def judge(_text: str) -> str:
        return next(verdicts)

    env = Envelope.from_task("q")
    result = await verify_with_retry(agent, env, judge, max_verify=3)
    assert agent.calls == 3
    assert result.payload == "ok-output"
