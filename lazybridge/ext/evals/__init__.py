"""Eval framework — alpha.

Pydantic-typed evaluation cases plus a small assertion-helpers library
(``contains`` / ``exact_match`` / ``min_length`` / ``max_length`` /
``not_contains`` / ``llm_judge``) for batch-grading agent outputs.

The runtime ``verify_with_retry`` helper used by ``Agent(verify=...)``
lives in core (private :mod:`lazybridge._verify`) — foundational
plumbing rather than an evaluation feature.
"""

from __future__ import annotations

#: Stability tag — see ``docs/guides/core-vs-ext.md``.
__stability__ = "alpha"
__lazybridge_min__ = "1.0.0"

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    input: str
    check: Callable[..., bool]
    expected: Any = None
    description: str = ""


@dataclass
class EvalResult:
    case: EvalCase
    output: str
    passed: bool
    error: str | None = None


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.error is None)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    def __str__(self) -> str:
        pct = int(self.passed / self.total * 100) if self.total else 0
        return f"{self.passed}/{self.total} passed ({pct}%)"


class EvalSuite:
    """Run a set of EvalCases against any agent callable."""

    def __init__(self, *cases: EvalCase) -> None:
        self.cases = list(cases)

    def run(self, agent: Any) -> EvalReport:
        report = EvalReport()
        for case in self.cases:
            try:
                output = agent(case.input).text()
                if case.expected is not None:
                    passed = case.check(output, case.expected)
                else:
                    passed = case.check(output)
                report.results.append(EvalResult(case=case, output=output, passed=bool(passed)))
            except Exception as exc:
                report.results.append(EvalResult(case=case, output="", passed=False, error=str(exc)))
        return report

    async def arun(self, agent: Any) -> EvalReport:
        # Run all cases concurrently — sequential awaiting would make
        # arun() as slow as the synchronous run() for large suites.
        async def _run_one(case: EvalCase) -> EvalResult:
            try:
                env = await agent.run(case.input)
                output = env.text()
                if case.expected is not None:
                    passed = case.check(output, case.expected)
                else:
                    passed = case.check(output)
                return EvalResult(case=case, output=output, passed=bool(passed))
            except Exception as exc:
                return EvalResult(case=case, output="", passed=False, error=str(exc))

        results = await asyncio.gather(*[_run_one(c) for c in self.cases])
        report = EvalReport()
        report.results = list(results)
        return report


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def exact_match(expected: str) -> Callable[[str, str], bool]:
    return lambda output, exp: output.strip() == exp.strip()


def contains(substring: str) -> Callable[[str], bool]:
    return lambda output: substring.lower() in output.lower()


def max_length(n: int) -> Callable[[str], bool]:
    return lambda output: len(output) <= n


def min_length(n: int) -> Callable[[str], bool]:
    return lambda output: len(output) >= n


def not_contains(substring: str) -> Callable[[str], bool]:
    return lambda output: substring.lower() not in output.lower()


def llm_judge(agent: Any, criteria: str) -> Callable[[str], bool]:
    """Returns a judge function using an agent to evaluate output."""

    def judge(output: str) -> bool:
        verdict = agent(f"Criteria: {criteria}\nOutput to judge: {output}\nVerdict (approved/rejected):").text()
        return verdict.strip().lower().startswith("approved")

    return judge


__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalReport",
    "EvalSuite",
    "exact_match",
    "contains",
    "not_contains",
    "min_length",
    "max_length",
    "llm_judge",
]
