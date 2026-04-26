"""Private — ``verify_with_retry`` runtime helper for ``Agent(verify=...)``.

The public eval surface (``EvalSuite`` / ``EvalCase`` / ``llm_judge`` /
matchers) lives in :mod:`lazybridge.ext.evals`.  This helper stays in
core because ``Agent`` invokes it directly when ``verify=`` is set.

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

    The pristine task / context are cached outside the loop and a
    clean envelope is rebuilt every attempt — so feedback flows via
    ``context``, never accumulating onto ``env.task``.
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
        merged_context = f"{original_context}\n\n{feedback_ctx}" if original_context else feedback_ctx
        current_env = Envelope(task=original_task, context=merged_context)

    return result
