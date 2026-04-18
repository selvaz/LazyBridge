"""lazybridge.evals — lightweight evaluation framework for agents.

Run test cases against agents and measure output quality. No heavy deps —
uses plain functions as judges.

Quick start::

    from lazybridge import LazyAgent
    from lazybridge.evals import EvalCase, EvalSuite, exact_match, contains

    agent = LazyAgent("anthropic")
    suite = EvalSuite(
        cases=[
            EvalCase("What is 2+2?", check=exact_match("4")),
            EvalCase("Name a planet", check=contains("Earth", "Mars", "Jupiter")),
            EvalCase(
                "Translate 'hello' to French",
                check=lambda output: "bonjour" in output.lower(),
            ),
        ],
    )
    report = suite.run(agent)
    print(report)
    # EvalReport: 3/3 passed (100.0%)

Custom judges::

    def length_check(output: str) -> bool:
        return 10 < len(output) < 500

    EvalCase("Write a haiku", check=length_check)

LLM-as-judge::

    from lazybridge.evals import llm_judge

    judge = LazyAgent("openai", model="gpt-4o-mini")
    EvalCase(
        "Explain quantum computing to a 5-year-old",
        check=llm_judge(judge, criteria="Age-appropriate, accurate, under 100 words"),
    )
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in check functions
# ---------------------------------------------------------------------------


def exact_match(expected: str, *, case_sensitive: bool = False) -> Callable[[str], bool]:
    """Check that output matches expected text exactly."""

    def _check(output: str) -> bool:
        if case_sensitive:
            return output.strip() == expected.strip()
        return output.strip().lower() == expected.strip().lower()

    return _check


def contains(*substrings: str, case_sensitive: bool = False) -> Callable[[str], bool]:
    """Check that output contains at least one of the given substrings."""

    def _check(output: str) -> bool:
        text = output if case_sensitive else output.lower()
        return any((s if case_sensitive else s.lower()) in text for s in substrings)

    return _check


def not_contains(*substrings: str, case_sensitive: bool = False) -> Callable[[str], bool]:
    """Check that output does NOT contain any of the given substrings."""

    def _check(output: str) -> bool:
        text = output if case_sensitive else output.lower()
        return all((s if case_sensitive else s.lower()) not in text for s in substrings)

    return _check


def min_length(n: int) -> Callable[[str], bool]:
    """Check that output has at least n characters."""

    def _check(output: str) -> bool:
        return len(output.strip()) >= n

    return _check


def max_length(n: int) -> Callable[[str], bool]:
    """Check that output has at most n characters."""

    def _check(output: str) -> bool:
        return len(output.strip()) <= n

    return _check


def llm_judge(
    judge_agent: Any,
    criteria: str,
) -> Callable[[str], bool]:
    """Use an LLM as an eval judge.

    The judge receives the output + criteria and returns PASS/FAIL.
    """
    _PROMPT = (
        "You are an evaluation judge. Evaluate the following output against the criteria.\n\n"
        "Criteria: {criteria}\n\n"
        "Output to evaluate: {output}\n\n"
        'Respond with exactly "PASS" or "FAIL". No other text.'
    )

    def _check(output: str) -> bool:
        prompt = _PROMPT.format(criteria=criteria, output=output[:2000])
        try:
            verdict = judge_agent.text(prompt).strip().upper()
            return verdict.startswith("PASS")
        except Exception as exc:
            # Previously this swallowed the exception and returned False,
            # making "judge crashed" indistinguishable from "judge said
            # FAIL" in EvalReport.  Raise through a marker exception so
            # EvalSuite.run / arun captures it in EvalResult.error (audit
            # L11) and EvalReport.errors counts it separately.
            _logger.warning("LLM judge failed: %s", exc)
            raise JudgeError(str(exc)) from exc

    return _check


class JudgeError(RuntimeError):
    """Raised by :func:`llm_judge` when the judge agent itself errors out.

    Distinguishes "the judge crashed" from "the judge returned FAIL" in
    :class:`EvalReport`.  Callers generally don't catch this — the suite
    runner does, and records it under :attr:`EvalResult.error`.
    """


# ---------------------------------------------------------------------------
# EvalCase + EvalResult
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """A single evaluation test case.

    Attributes
    ----------
    prompt : str
        The input to send to the agent.
    check : Callable[[str], bool]
        Function that receives the agent's output text and returns True/False.
    name : str
        Human-readable name (defaults to first 50 chars of prompt).
    tags : list[str]
        Optional tags for filtering (e.g. ["safety", "accuracy"]).
    chat_kwargs : dict
        Extra keyword arguments passed to agent.text().
    """

    prompt: str
    check: Callable[[str], bool]
    name: str = ""
    tags: list[str] = field(default_factory=list)
    chat_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.prompt[:50].strip()


@dataclass
class EvalResult:
    """Result of running a single EvalCase."""

    case: EvalCase
    passed: bool
    output: str
    duration_ms: float
    error: str | None = None

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"EvalResult({status}, {self.case.name!r}, {self.duration_ms:.0f}ms)"


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Aggregated results from an EvalSuite run."""

    results: list[EvalResult]
    duration_ms: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def errors(self) -> int:
        """Cases where the check itself raised (e.g. judge crashed) —
        distinct from pure FAIL verdicts (audit L11)."""
        return sum(1 for r in self.results if r.error is not None)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    @property
    def failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]

    def by_tag(self, tag: str) -> list[EvalResult]:
        return [r for r in self.results if tag in r.case.tags]

    def __repr__(self) -> str:
        return f"EvalReport: {self.passed}/{self.total} passed ({self.pass_rate:.1f}%) in {self.duration_ms:.0f}ms"


# ---------------------------------------------------------------------------
# EvalSuite
# ---------------------------------------------------------------------------


class EvalSuite:
    """Collection of eval cases that can be run against any agent.

    Usage::

        suite = EvalSuite(cases=[
            EvalCase("What is 2+2?", check=exact_match("4")),
            EvalCase("Say hello", check=contains("hello", "hi")),
        ])
        report = suite.run(agent)
        print(report)

        # Filter by tag
        suite = EvalSuite(cases=[
            EvalCase("safe?", check=not_contains("harmful"), tags=["safety"]),
            EvalCase("accurate?", check=contains("correct"), tags=["accuracy"]),
        ])
        report = suite.run(agent)
        for r in report.by_tag("safety"):
            print(r)
    """

    def __init__(self, cases: list[EvalCase]) -> None:
        self._cases = list(cases)

    def run(self, agent: Any, **global_kwargs: Any) -> EvalReport:
        """Run all cases against an agent. Returns an EvalReport."""
        results: list[EvalResult] = []
        suite_start = time.monotonic()

        for case in self._cases:
            kwargs = {**global_kwargs, **case.chat_kwargs}
            start = time.monotonic()
            try:
                output = agent.text(case.prompt, **kwargs)
                passed = case.check(output)
                duration = (time.monotonic() - start) * 1000
                results.append(EvalResult(case=case, passed=passed, output=output, duration_ms=duration))
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                results.append(EvalResult(case=case, passed=False, output="", duration_ms=duration, error=str(exc)))

        total_duration = (time.monotonic() - suite_start) * 1000
        return EvalReport(results=results, duration_ms=total_duration)

    async def arun(self, agent: Any, **global_kwargs: Any) -> EvalReport:
        """Async version — runs cases sequentially with atext()."""
        results: list[EvalResult] = []
        suite_start = time.monotonic()

        for case in self._cases:
            kwargs = {**global_kwargs, **case.chat_kwargs}
            start = time.monotonic()
            try:
                output = await agent.atext(case.prompt, **kwargs)
                passed = case.check(output)
                duration = (time.monotonic() - start) * 1000
                results.append(EvalResult(case=case, passed=passed, output=output, duration_ms=duration))
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                results.append(EvalResult(case=case, passed=False, output="", duration_ms=duration, error=str(exc)))

        total_duration = (time.monotonic() - suite_start) * 1000
        return EvalReport(results=results, duration_ms=total_duration)
