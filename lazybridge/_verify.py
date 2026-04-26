"""Private — ``verify_with_retry`` runtime helper for ``Agent(verify=...)``.

This used to live in ``lazybridge.evals`` together with ``EvalSuite`` /
``EvalCase`` / ``llm_judge`` etc.  In 1.0.1 the public eval surface
moved to :mod:`lazybridge.ext.evals` (the eval API is in active
evolution, not core), but ``verify_with_retry`` stayed in core because
``Agent`` invokes it directly when ``verify=`` is set.

The module is private (leading underscore): no external imports.
"""

from __future__ import annotations

from typing import Any


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
