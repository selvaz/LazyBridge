"""Tests for the v1.0 evals framework."""

from __future__ import annotations

import asyncio

from lazybridge.envelope import Envelope
from lazybridge.ext.evals import (
    EvalCase,
    EvalReport,
    EvalSuite,
    contains,
    exact_match,
    max_length,
    min_length,
    not_contains,
)


def _sync_agent(response: str):
    class _A:
        def __call__(self, task):
            return Envelope(payload=response)

    return _A()


# ── check helpers ─────────────────────────────────────────────────────────────


def test_exact_match_pass():
    assert exact_match("hi")("hi", "hi") is True


def test_exact_match_fail():
    assert exact_match("hi")("bye", "hi") is False


def test_exact_match_strips_whitespace():
    assert exact_match("hi")("  hi  ", "hi") is True


def test_contains_pass():
    assert contains("foo")("foobar") is True


def test_contains_case_insensitive():
    assert contains("FOO")("foobar") is True


def test_contains_fail():
    assert contains("xyz")("foobar") is False


def test_not_contains_pass():
    assert not_contains("error")("all good") is True


def test_not_contains_fail():
    assert not_contains("error")("there was an error") is False


def test_max_length_pass():
    assert max_length(100)("short") is True


def test_max_length_fail():
    assert max_length(3)("toolong") is False


def test_min_length_pass():
    assert min_length(3)("long enough") is True


def test_min_length_fail():
    assert min_length(100)("short") is False


# ── EvalSuite.run ─────────────────────────────────────────────────────────────


def test_suite_all_pass():
    agent = _sync_agent("hello world")
    suite = EvalSuite(
        EvalCase(input="q", check=contains("hello")),
        EvalCase(input="q", check=contains("world")),
    )
    report = suite.run(agent)
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert "2/2" in str(report)


def test_suite_partial_fail():
    agent = _sync_agent("hello")
    suite = EvalSuite(
        EvalCase(input="q", check=contains("hello")),
        EvalCase(input="q", check=contains("world")),
    )
    report = suite.run(agent)
    assert report.passed == 1
    assert report.failed == 1


def test_suite_agent_exception():
    class _Boom:
        def __call__(self, task):
            raise RuntimeError("API down")

    suite = EvalSuite(EvalCase(input="q", check=contains("x")))
    report = suite.run(_Boom())
    assert report.errors == 1
    assert report.passed == 0


def test_suite_with_expected():
    agent = _sync_agent("Paris")
    suite = EvalSuite(EvalCase(input="capital of France?", check=exact_match("Paris"), expected="Paris"))
    report = suite.run(agent)
    assert report.passed == 1


def test_report_str_format():
    report = EvalReport()
    from lazybridge.ext.evals import EvalResult

    report.results.append(EvalResult(case=EvalCase("q", check=contains("x")), output="x", passed=True))
    assert "1/1" in str(report)
    assert "100%" in str(report)


# ── async suite ───────────────────────────────────────────────────────────────


def test_suite_arun():
    class _AsyncAgent:
        async def run(self, task):
            return Envelope(payload="hello world")

    suite = EvalSuite(EvalCase(input="q", check=contains("hello")))
    report = asyncio.run(suite.arun(_AsyncAgent()))
    assert report.passed == 1


# ── llm_judge — robust verdict parsing (W1.1-bis) ─────────────────────────────


class _JudgeAgent:
    """Stub agent whose .text() returns the canned verdict string."""

    def __init__(self, verdict: str):
        self._v = verdict

    def __call__(self, prompt: str):
        v = self._v

        class _E:
            def text(self) -> str:
                return v

        return _E()


def test_llm_judge_accepts_approved_canonical():
    from lazybridge.ext.evals import llm_judge

    judge = llm_judge(_JudgeAgent("approved: looks great"), criteria="be concise")
    assert judge("any output") is True


def test_llm_judge_accepts_synonyms():
    """Pre-W1.1-bis only ``approved`` prefix passed; W1.1-bis recognises
    a documented synonym set (yes / ok / allow / pass / approve /
    accept / good / valid)."""
    from lazybridge.ext.evals import llm_judge

    # NB: "looks good" intentionally NOT in this list — the parser is
    # conservative (fail-safe).  A judge response that doesn't START
    # with a recognised verdict word is treated as rejected, even if
    # the word appears later in the same line; this prevents an
    # ambiguous "looks good but actually bad" from accidentally
    # passing.
    for verdict in ("yes", "OK", "allow", "pass", "approve", "accept", "Good — clear", "Valid"):
        judge = llm_judge(_JudgeAgent(verdict), criteria="x")
        assert judge("output") is True, f"verdict {verdict!r} should be approved"


def test_llm_judge_rejects_explicit_negatives():
    from lazybridge.ext.evals import llm_judge

    for verdict in ("rejected", "deny", "block", "fail", "no", "bad", "invalid"):
        judge = llm_judge(_JudgeAgent(verdict), criteria="x")
        assert judge("output") is False, f"verdict {verdict!r} should be rejected"


def test_llm_judge_unparseable_verdict_fails_closed():
    """If the judge can't produce a recognised verdict, fail-safe to
    rejection — a vague ``maybe`` never accidentally passes a bad
    output."""
    from lazybridge.ext.evals import llm_judge

    for verdict in ("maybe?", "I'm not sure", "the output might be acceptable", ""):
        judge = llm_judge(_JudgeAgent(verdict), criteria="x")
        assert judge("output") is False, f"unparseable verdict {verdict!r} must fail closed"


def test_llm_judge_handles_formatted_verdicts():
    """Markdown / numbered / bullet-prefixed verdicts still parse (the
    underlying normaliser strips leading punctuation)."""
    from lazybridge.ext.evals import llm_judge

    for verdict in ("**approved**", "> allow", "- pass", "1. approve", "Approved\nreason: ok"):
        judge = llm_judge(_JudgeAgent(verdict), criteria="x")
        assert judge("output") is True, f"formatted verdict {verdict!r} should approve"
