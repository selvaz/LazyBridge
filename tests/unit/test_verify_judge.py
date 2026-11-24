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


class _RecordingJudge:
    """Agent-like judge that records the prompt it receives."""

    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)
        self.prompts: list[str] = []

    async def run(self, prompt):
        self.prompts.append(prompt if isinstance(prompt, str) else prompt.task)
        return Envelope(task="judge", payload=next(self.verdicts))


@pytest.mark.asyncio
async def test_judge_context_is_opt_in_and_never_includes_feedback():
    agent = _FakeAgent("draft citing [1]")
    judge = _RecordingJudge(["rejected: [1] does not say that", "approved"])
    env = Envelope(task="q", context="[1] https://example.test/source says X")
    await verify_with_retry(agent, env, judge, max_verify=3, judge_context=env.context)
    assert len(judge.prompts) == 2
    for prompt in judge.prompts:
        assert "https://example.test/source says X" in prompt and "Original task: q" in prompt
    assert "Feedback from judge" not in judge.prompts[1]


@pytest.mark.asyncio
async def test_judge_sees_no_context_by_default():
    agent = _FakeAgent()
    judge = _RecordingJudge(["approved"])
    env = Envelope(task="q", context="confidential source material")
    await verify_with_retry(agent, env, judge, max_verify=2)
    assert "confidential source material" not in judge.prompts[0]
    assert "Context the agent was given" not in judge.prompts[0]


def _scripted_agent(**kwargs):
    from lazybridge import Agent

    class _Engine:
        model = "scripted"

        async def run(self, env, **_kw):
            return Envelope(task=env.task, payload="draft")

    return Agent(engine=_Engine(), **kwargs)


@pytest.mark.asyncio
async def test_agent_verify_context_flag_passes_sources_to_judge():
    judge = _RecordingJudge(["approved"])
    agent = _scripted_agent(name="writer", sources=["SOURCE-TEXT-42"], verify=judge, verify_context=True)
    await agent.run("q")
    assert "SOURCE-TEXT-42" in judge.prompts[0]


@pytest.mark.asyncio
async def test_as_tool_verify_context_includes_the_wrapped_agents_sources():
    judge = _RecordingJudge(["approved"])
    agent = _scripted_agent(name="writer", sources=["SOURCE-TEXT-43"])
    tool = agent.as_tool("write", verify=judge, max_verify=2, verify_context=True)
    await tool.func("q")
    assert "SOURCE-TEXT-43" in judge.prompts[0]
    plain = agent.as_tool("write2", verify=_RecordingJudge(["approved"]), max_verify=2)
    assert plain is not None


def test_verify_context_does_not_shift_positional_parameters():
    import inspect

    from lazybridge import Agent

    signature = inspect.signature(Agent.__init__)
    positional = [p.name for p in signature.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[-1] == "verify_context"
    assert positional.index("name") == positional.index("max_verify") + 1
    assert signature.parameters["policy"].kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "verdict",
    [
        "VERDICT: APPROVED\n\nThe revised draft addresses the concerns.",
        "Verdict: approved",
        "Verdict: yes",
        "**VERDICT: APPROVED**",
        "**Verdict:** pass",
        "Final verdict - accepted",
        "Decision: approve",
    ],
)
def test_labelled_approvals_are_recognised(verdict):
    assert _is_approved(verdict) is True


@pytest.mark.parametrize(
    "verdict",
    [
        "VERDICT: REJECTED\n\nTwo claims are unsupported.",
        "**Verdict:** rejected",
        "Decision: no",
        "Result = fail",
        # An unrelated label is not stripped: fail-safe rejection.
        "Overall: approved",
    ],
)
def test_labelled_rejections_and_unknown_labels_fail_safe(verdict):
    assert _is_approved(verdict) is False


@pytest.mark.parametrize(
    "verdict",
    [
        "Verdict: approve only if the dates are fixed",
        "Decision: OK to reject",
        "Review - approved with major issues",
        "Status: not approved",
    ],
)
def test_qualified_labelled_verdicts_fail_safe(verdict):
    assert _is_approved(verdict) is False


@pytest.mark.parametrize(
    "verdict",
    [
        "Verdict: approved \u2014 only if the dates are fixed",
        "Verdict: approved, but fix the dates first",
        "Decision: pass - unless the source is retracted",
        "VERDICT: APPROVED. Only after the second source is added.",
        "Verdict:\napprove only if the dates are fixed",
        "**Verdict:**\n\nrejected",
        "Verdict:\nno",
    ],
)
def test_conditional_or_split_labelled_verdicts_fail_safe(verdict):
    assert _is_approved(verdict) is False


@pytest.mark.parametrize(
    "verdict",
    [
        "Verdict:\nAPPROVED",
        "**Verdict:**\n\n**APPROVED**\nEvery claim matches a quote.",
    ],
)
def test_split_or_rationale_labelled_approvals_pass(verdict):
    assert _is_approved(verdict) is True


@pytest.mark.parametrize(
    "verdict",
    [
        # Anything after a labelled approval on the same line fails safe,
        # rationale included: the allowlist is the bare word.
        "Verdict: approved \u2014 all claims are supported by the quotes",
        "Verdict: approved \u2014 contingent on fixing the dates",
        "VERDICT:\nAPPROVED \u2014 all claims supported",
        # A condition on the line after a bare labelled approval.
        "VERDICT: APPROVED\nOnly if the dates are fixed",
        "VERDICT: APPROVED\n\n- contingent on a second source",
        "VERDICT: APPROVED\n1. Only if the dates are fixed",
        "VERDICT: APPROVED\n2) With the condition that the dates are corrected",
    ],
)
def test_labelled_approval_with_anything_attached_fails_safe(verdict):
    assert _is_approved(verdict) is False
