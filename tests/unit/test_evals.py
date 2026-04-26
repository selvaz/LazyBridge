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
    suite = EvalSuite(
        EvalCase(input="capital of France?", check=exact_match("Paris"), expected="Paris")
    )
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
