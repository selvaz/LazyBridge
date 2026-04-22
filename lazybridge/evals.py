"""Evals — verify/judge framework with retry loop for agent output validation."""

from __future__ import annotations

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
        # F8: run all cases concurrently — sequential awaiting made arun()
        # as slow as the synchronous run() for large eval suites.
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
# Verify loop helpers — used by Agent for output validation with retry
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


async def verify_with_retry(
    agent: Any,
    env: Any,
    verify_agent: Any,
    *,
    max_verify: int = 3,
) -> Any:
    """Run ``agent`` and gate its output through ``verify_agent``.

    If the judge rejects, retry the ORIGINAL task with the judge's
    feedback appended as context.  Up to ``max_verify`` attempts; the
    last attempt is returned as-is even if still rejected.

    Pre-fix, each retry appended feedback to the previous retry's
    ``env.task`` — so by attempt 3 the task was the original prompt
    layered with two feedback paragraphs, and the judge's
    ``"Original task: {env.task}"`` line was showing the already-
    modified task rather than the user's real input.  Now we cache
    the original task / context outside the loop and rebuild a clean
    envelope every attempt.
    """
    from lazybridge.envelope import Envelope

    original_task = getattr(env, "task", None) or ""
    original_context = getattr(env, "context", None)
    current_env = env
    result: Any = None

    for attempt in range(max_verify):
        result = await agent.run(current_env)

        if not hasattr(verify_agent, "run"):
            # Plain callable judge.
            verdict = verify_agent(result.text())
            approved = str(verdict).lower().startswith("approved")
        else:
            verdict_env = await verify_agent.run(
                f"Evaluate this output:\n{result.text()}\n\n"
                f"Original task: {original_task}\n\n"
                f"Approved or rejected (with reason)?"
            )
            verdict = verdict_env.text()
            approved = verdict.strip().lower().startswith("approved")

        if approved or attempt == max_verify - 1:
            return result

        # Rebuild from the pristine original task.  Feedback goes into
        # the context slot rather than concatenated onto the task so
        # the judge always sees the user's real question.
        feedback = str(verdict)
        feedback_ctx = f"Feedback from judge: {feedback}"
        merged_context = (
            f"{original_context}\n\n{feedback_ctx}" if original_context else feedback_ctx
        )
        current_env = Envelope(task=original_task, context=merged_context)

    return result
