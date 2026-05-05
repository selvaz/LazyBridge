"""Private — ``verify_with_retry`` runtime helper for ``Agent(verify=...)``.

The public eval surface (``EvalSuite`` / ``EvalCase`` / ``llm_judge`` /
matchers) lives in :mod:`lazybridge.ext.evals`.  This helper stays in
core because ``Agent`` invokes it directly when ``verify=`` is set.

The module is private (leading underscore): no external imports.
"""

from __future__ import annotations

import re
from typing import Any

# Verdict normalization.  A judge — whether a plain callable returning
# free-form text or an ``Agent`` whose final ``.text()`` we read — must
# decide ``approved`` / ``rejected``.  Historic behaviour gated on
# ``startswith("approved")`` only; that silently rejected reasonable
# verdicts ("yes", "ok", "looks good", "allow") and made the loop fragile.
#
# We now match a small, well-defined set of synonyms anchored at the
# start of the (lower-cased, stripped) verdict.  Everything else falls
# through to ``rejected`` — including silence, errors, and anything the
# judge couldn't classify.  This is intentionally allowlist-style: a
# judge that fails to produce a recognisable approval is treated as a
# rejection (fail-safe).

_APPROVE_PATTERN = re.compile(
    r"^(approve(d)?|accept(ed)?|allow(ed)?|pass(ed)?|okay|ok|yes|good|valid)\b",
    re.IGNORECASE,
)
_REJECT_PATTERN = re.compile(
    r"^(reject(ed)?|denied|deny|block(ed)?|fail(ed)?|no|bad|invalid)\b",
    re.IGNORECASE,
)


def _is_approved(verdict: Any) -> bool:
    """Normalise a judge verdict to a boolean approval.

    Recognises common synonyms for approve/reject anchored at the start
    of the verdict text (case-insensitive).  Unrecognised verdicts are
    treated as ``rejected`` — fail-safe.

    Tolerant of common formatting that real judges emit: leading
    markdown (``**approved**``), block-quote (``> allow``), bullet /
    numbered list (``- pass``, ``1. approve``), or label prefix
    (``Verdict: yes``).  The first line that yields a recognised
    verdict word after this trim wins; later lines (judge's
    rationale) are ignored.
    """
    if verdict is None:
        return False
    if isinstance(verdict, bool):
        return verdict
    text = str(verdict).strip()
    if not text:
        return False
    # Walk lines so a multi-line judge response (verdict + reason on
    # a separate line) is parsed by its first informative line.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Strip leading markdown / punctuation / numbering so verdicts
        # like ``**approved**``, ``> allow``, ``1. pass`` parse cleanly.
        # The character class mirrors LLMGuard._verdict so both judge
        # paths share the same tolerance.
        stripped = line.lstrip("*>#-_ \t:.0123456789")
        if not stripped:
            continue
        if _REJECT_PATTERN.match(stripped):
            return False
        # First non-blank, non-empty-after-strip line that doesn't
        # match either pattern is the verdict line — fail-safe to
        # rejection rather than scanning down for a maybe-approval
        # buried in a rationale paragraph.
        return bool(_APPROVE_PATTERN.match(stripped))
    return False


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

    Verdict recognition: the judge may return any of
    ``approved`` / ``accept`` / ``allow`` / ``pass`` / ``ok`` / ``yes`` /
    ``good`` / ``valid`` (synonyms, case-insensitive, prefix-anchored)
    to approve.  Explicit reject prefixes (``rejected`` / ``deny`` /
    ``block`` / ``fail`` / ``no`` / ``bad`` / ``invalid``) and any
    unrecognised verdict are treated as rejection (fail-safe).
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
        else:
            verdict_env = await verify_agent.run(
                f"Evaluate this output:\n{result.text()}\n\n"
                f"Original task: {original_task}\n\n"
                f"Approved or rejected (with reason)?"
            )
            verdict = verdict_env.text()

        approved = _is_approved(verdict)

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
