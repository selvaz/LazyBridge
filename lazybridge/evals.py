"""Evals — verify/judge framework with retry loop for agent output validation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    input: str
    check: Callable[[str], bool] | Callable[[str, Any], bool]
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
        report = EvalReport()
        for case in self.cases:
            try:
                env = await agent.run(case.input)
                output = env.text()
                if case.expected is not None:
                    passed = case.check(output, case.expected)
                else:
                    passed = case.check(output)
                report.results.append(EvalResult(case=case, output=output, passed=bool(passed)))
            except Exception as exc:
                report.results.append(EvalResult(case=case, output="", passed=False, error=str(exc)))
        return report


# ---------------------------------------------------------------------------
# Verify loop helpers — used by Agent for output validation with retry
# ---------------------------------------------------------------------------


def exact_match(expected: str) -> Callable[[str, str], bool]:
    return lambda output, exp: output.strip() == exp.strip()


def contains(substring: str) -> Callable[[str], bool]:
    return lambda output: substring.lower() in output.lower()


def llm_judge(agent: Any, criteria: str) -> Callable[[str], bool]:
    """Returns a judge function using an agent to evaluate output."""
    def judge(output: str) -> bool:
        verdict = agent(f"Criteria: {criteria}\nOutput to judge: {output}\nVerdict (approved/rejected):").text()
        return verdict.strip().lower().startswith("approved")
    return judge


async def verify_with_retry(
    agent: Any,
    env: Any,
    verify_agent: Any,
    *,
    max_verify: int = 3,
) -> Any:
    """Run verify_agent on agent output; retry with feedback up to max_verify times."""
    from lazybridge.envelope import Envelope

    for attempt in range(max_verify):
        result = await agent.run(env)
        if not hasattr(verify_agent, "run"):
            # Plain callable judge
            verdict = verify_agent(result.text())
            if str(verdict).lower().startswith("approved"):
                return result
            if attempt == max_verify - 1:
                return result
            # Inject feedback for next attempt
            feedback = str(verdict)
            env = Envelope(task=f"{env.task}\n\nFeedback from judge: {feedback}", context=env.context)
        else:
            verdict_env = await verify_agent.run(
                f"Evaluate this output:\n{result.text()}\n\nOriginal task: {env.task}\n\nApproved or rejected (with reason)?"
            )
            verdict = verdict_env.text()
            if verdict.strip().lower().startswith("approved"):
                return result
            if attempt == max_verify - 1:
                return result
            env = Envelope(task=f"{env.task}\n\nFeedback: {verdict}", context=env.context)

    return result
