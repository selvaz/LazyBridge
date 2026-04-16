"""Unit tests for the evals framework."""

from __future__ import annotations

from unittest.mock import MagicMock

from lazybridge.evals import (
    EvalCase,
    EvalSuite,
    contains,
    exact_match,
    max_length,
    min_length,
    not_contains,
)


def _mock_agent(responses: dict[str, str]):
    """Mock agent that returns predefined responses based on prompt."""
    agent = MagicMock()

    def _text(prompt, **kw):
        for key, val in responses.items():
            if key in prompt:
                return val
        return "default response"

    agent.text = MagicMock(side_effect=_text)
    return agent


# ---------------------------------------------------------------------------
# Built-in check functions
# ---------------------------------------------------------------------------


def test_exact_match():
    check = exact_match("hello")
    assert check("hello") is True
    assert check("Hello") is True
    assert check("  hello  ") is True
    assert check("hello world") is False


def test_exact_match_case_sensitive():
    check = exact_match("Hello", case_sensitive=True)
    assert check("Hello") is True
    assert check("hello") is False


def test_contains():
    check = contains("cat", "dog")
    assert check("I have a cat") is True
    assert check("I have a dog") is True
    assert check("I have a fish") is False


def test_not_contains():
    check = not_contains("error", "fail")
    assert check("all good") is True
    assert check("there was an error") is False


def test_min_length():
    check = min_length(10)
    assert check("short") is False
    assert check("this is long enough") is True


def test_max_length():
    check = max_length(10)
    assert check("short") is True
    assert check("this is way too long") is False


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------


def test_eval_case_auto_name():
    case = EvalCase("What is the capital of France?", check=exact_match("Paris"))
    assert case.name == "What is the capital of France?"


def test_eval_case_custom_name():
    case = EvalCase("prompt", check=exact_match("x"), name="my test")
    assert case.name == "my test"


# ---------------------------------------------------------------------------
# EvalSuite.run()
# ---------------------------------------------------------------------------


def test_suite_all_pass():
    agent = _mock_agent({"2+2": "4", "capital": "Paris"})
    suite = EvalSuite(
        cases=[
            EvalCase("What is 2+2?", check=exact_match("4")),
            EvalCase("capital of France?", check=contains("Paris")),
        ]
    )
    report = suite.run(agent)
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.pass_rate == 100.0


def test_suite_partial_fail():
    agent = _mock_agent({"2+2": "5", "capital": "Paris"})
    suite = EvalSuite(
        cases=[
            EvalCase("What is 2+2?", check=exact_match("4")),
            EvalCase("capital of France?", check=contains("Paris")),
        ]
    )
    report = suite.run(agent)
    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert len(report.failures) == 1
    assert "2+2" in report.failures[0].case.name


def test_suite_agent_error():
    agent = MagicMock()
    agent.text.side_effect = RuntimeError("API down")
    suite = EvalSuite(cases=[EvalCase("test", check=exact_match("x"))])
    report = suite.run(agent)
    assert report.total == 1
    assert report.passed == 0
    assert report.results[0].error == "API down"


def test_suite_by_tag():
    agent = _mock_agent({"safe": "clean", "accurate": "correct answer"})
    suite = EvalSuite(
        cases=[
            EvalCase("safe?", check=not_contains("harmful"), tags=["safety"]),
            EvalCase("accurate?", check=contains("correct"), tags=["accuracy"]),
        ]
    )
    report = suite.run(agent)
    safety = report.by_tag("safety")
    assert len(safety) == 1
    assert safety[0].passed is True


def test_suite_report_repr():
    agent = _mock_agent({"q": "a"})
    suite = EvalSuite(cases=[EvalCase("q", check=exact_match("a"))])
    report = suite.run(agent)
    assert "1/1 passed" in repr(report)
    assert "100.0%" in repr(report)


def test_suite_duration_tracked():
    agent = _mock_agent({"q": "a"})
    suite = EvalSuite(cases=[EvalCase("q", check=exact_match("a"))])
    report = suite.run(agent)
    assert report.duration_ms >= 0
    assert report.results[0].duration_ms >= 0


# ---------------------------------------------------------------------------
# Async suite
# ---------------------------------------------------------------------------


async def test_suite_arun():
    from unittest.mock import AsyncMock

    agent = MagicMock()
    agent.atext = AsyncMock(return_value="4")
    suite = EvalSuite(cases=[EvalCase("2+2?", check=exact_match("4"))])
    report = await suite.arun(agent)
    assert report.passed == 1
