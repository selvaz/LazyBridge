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
#: A label in front of the verdict word (``VERDICT: APPROVED``,
#: ``Decision - rejected``, ``**Result:** pass``). Only these label words
#: are stripped, so an unrelated first word still fails safe to rejection.
_LABEL_PREFIX = re.compile(
    r"^(final\s+)?(verdict|decision|result|outcome|status|judg(e)?ment|review)\s*[:\-=]\s*",
    re.IGNORECASE,
)
#: After a label, an approval counts only when the word stands alone on its
#: line (``VERDICT: APPROVED``, ``**Verdict:** approved.``). Anything else on
#: that line (``approved - only if X``, ``approved, contingent on X``) makes
#: it a qualified verdict, which fails safe: an allowlist, not a list of
#: qualifier words that can never be complete.
_LABELLED_APPROVE = re.compile(
    r"^(approve(d)?|accept(ed)?|allow(ed)?|pass(ed)?|okay|ok|yes|good|valid)[*_]*[.!]?[*_]*$",
    re.IGNORECASE,
)
#: A line right after a bare labelled approval that opens with a condition
#: (``VERDICT: APPROVED`` then ``Only if the dates are fixed``) turns it into
#: a conditional verdict. Free text cannot be parsed completely: a judge that
#: must be exact should put the verdict alone on the first line.
_QUALIFIER = re.compile(
    r"^(only|if|unless|but|provided|pending|once|after|with|conditional(ly)?|contingent|subject|except|assuming|however)\b",
    re.IGNORECASE,
)


def _labelled_verdict(text: str, next_line: str) -> bool:
    """Verdict after a label: a bare approval word with no condition."""
    if _REJECT_PATTERN.match(text):
        return False
    if _LABELLED_APPROVE.match(text.strip()) is None:
        return False
    return not _QUALIFIER.match(next_line.lstrip("*>#-_ \t:.)0123456789"))


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
    lines = [raw.strip() for raw in text.splitlines()]
    pending_label = False
    for index, line in enumerate(lines):
        if not line:
            continue
        # Strip leading markdown / punctuation / numbering so verdicts
        # like ``**approved**``, ``> allow``, ``1. pass`` parse cleanly.
        # The character class mirrors LLMGuard._verdict so both judge
        # paths share the same tolerance.
        stripped = line.lstrip("*>#-_ \t:.0123456789")
        # ``VERDICT: APPROVED`` / ``**Verdict:** rejected``: drop the label,
        # then any markdown left between the label and the verdict word.
        labelled = pending_label or _LABEL_PREFIX.match(stripped) is not None
        stripped = _LABEL_PREFIX.sub("", stripped, count=1).lstrip("*>#-_ \t:.")
        if not stripped:
            # A label alone on its line ("Verdict:") governs the next line.
            pending_label = labelled
            continue
        if labelled:
            # After a label the verdict word must stand alone on its line and
            # the next line must not open with a condition: "Verdict: approve
            # only if X", "Verdict: approved - contingent on X" or "Decision:
            # OK to reject" are qualified, not approvals, and fail safe.
            following = next((rest for rest in lines[index + 1 :] if rest), "")
            return _labelled_verdict(stripped, following)
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
    run: Any = None,
    judge_context: str | None = None,
) -> Any:
    """Run ``agent`` and gate its output through ``verify_agent``.

    If the judge rejects, retry the ORIGINAL task with the judge's
    feedback appended as context.  Up to ``max_verify`` attempts; the
    last attempt is returned as-is even if still rejected.
    An uncertain producer or reviewer outcome stops the loop; ordinary producer failures are judged and retried.

    ``judge_context`` (opt-in, default ``None``): evidence an Agent judge
    should see in addition to the output and the task, typically the
    context the producing agent was given (for example sources from earlier
    plan steps). It is never the accumulated judge feedback. Callers pass it
    only when the user enabled ``verify_context=True``: a judge on another
    provider then receives that material too. Plain callable judges keep
    their text-only signature and never receive it.

    ``run`` overrides the coroutine used to produce each attempt
    (default: ``agent.run``).  ``Agent._run_body`` MUST pass its
    engine-only runner here: its ``verify=`` branch calls this helper,
    and this helper calling back into the full ``agent.run()`` would
    re-enter that same branch — infinite mutual recursion
    (``RecursionError`` on any ``Agent(verify=...).run()``).

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

    if max_verify < 1:
        raise ValueError(f"max_verify must be >= 1, got {max_verify!r}")

    original_task = getattr(env, "task", None) or ""
    original_context = getattr(env, "context", None)
    current_env = env
    result: Any = None
    runner = run if run is not None else agent.run

    for attempt in range(max_verify):
        result = await runner(current_env)
        if (
            isinstance(result, Envelope)
            and not result.ok
            and result.error is not None
            and result.error.outcome_uncertain
        ):
            return result

        if not hasattr(verify_agent, "run"):
            # Plain callable judge.
            verdict = verify_agent(result.text())
        else:
            # Opt-in evidence (judge_context): the context the agent was given,
            # never the accumulated judge feedback, so a judge can check claims
            # against sources instead of judging style only.
            evidence = (
                f"Context the agent was given (use it as evidence):\n{judge_context}\n\n" if judge_context else ""
            )
            verdict_env = await verify_agent.run(
                f"Evaluate this output:\n{result.text()}\n\n"
                f"Original task: {original_task}\n\n"
                f"{evidence}"
                f"Approved or rejected (with reason)?"
            )
            if isinstance(verdict_env, Envelope) and verdict_env.error is not None:
                if verdict_env.error.outcome_uncertain:
                    return verdict_env
                verdict = "Reviewer returned an error"
            else:
                verdict = verdict_env.text()

        approved = _is_approved(verdict)

        if approved or attempt == max_verify - 1:
            return result

        # Rebuild from the pristine original task.  Feedback goes into
        # the context slot rather than concatenated onto the task so
        # the judge always sees the user's real question.  Attachments
        # and payload are carried over from the ORIGINAL env — dropping
        # them meant every post-rejection attempt ran without the
        # images/audio/input the first attempt had.
        feedback = str(verdict)
        feedback_ctx = f"Feedback from judge: {feedback}"
        merged_context = f"{original_context}\n\n{feedback_ctx}" if original_context else feedback_ctx
        current_env = Envelope(
            task=original_task,
            context=merged_context,
            images=getattr(env, "images", None),
            audio=getattr(env, "audio", None),
            payload=getattr(env, "payload", None),
        )

    return result
