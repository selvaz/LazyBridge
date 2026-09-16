"""Shortening text for a human to read, without hiding the part that bites.

Every place that put a value in front of a person used to roll its own cut:
``text[:400]``, ``[:500]``, ``[:300]``, ``[:200]``, ``[:147]``. Five magic
numbers for one job, none of them derived from anything.

Measured against a live store of 397 real approval requests, 155 of them --
**39%** -- reached the operator truncated, and the operator approved 95% of
all requests. Four times in ten a human authorised a command they could read
only the beginning of.

Two things follow, and the second is the one that matters.

**The budget was far tighter than the transport.** These strings travel to
Telegram, whose hard limit is 4096 characters per message; the longest
prompt ever produced was 3617. Cutting at 400 was between eight and twenty
times more aggressive than anything required.

**Keeping the head is backwards.** In ``cd repo && build && rm -rf artifacts``
the consequence is last. A head-only truncation reliably shows the approver
the harmless opening and conceals whatever follows -- so the gate does not
merely inconvenience the human, it manufactures their consent. Anything
elided here therefore keeps both ends.

And the marker says how much went missing. ``...`` cannot distinguish one
dropped line from ten thousand, which is precisely the judgement the reader
needs to decide whether to go and look at the full value.
"""

from __future__ import annotations

#: How much of a value a human is shown before it is elided.
#:
#: Derived, not picked: Telegram's 4096-character per-message limit is the
#: tightest transport these strings travel over, and the lines around them
#: (header, cwd, reply instructions, ticket id) run to a few hundred
#: characters. Three thousand leaves real headroom for those and still shows
#: several times what any of the previous per-call-site numbers did.
DISPLAY_BUDGET = 3000

#: How the budget is split when both ends have to be kept. Weighted towards
#: the head because that is where a command says what it is, while the tail
#: is where it says what it will destroy -- a third is enough to see that.
_HEAD_SHARE = 2 / 3


def elide(text: str, budget: int = DISPLAY_BUDGET) -> str:
    """``text`` shortened to ``budget``, keeping BOTH ends.

    Returns the text unchanged when it fits, so short values are
    byte-for-byte what they always were.
    """
    if budget <= 0 or len(text) <= budget:
        return text
    marker_width = 40  # the elision line costs budget too; pay for it
    usable = max(budget - marker_width, 2)
    head = max(int(usable * _HEAD_SHARE), 1)
    tail = max(usable - head, 1)
    dropped = len(text) - head - tail
    if dropped <= 0:
        return text
    return f"{text[:head]}\n  […{dropped} characters elided…]\n{text[-tail:]}"


__all__ = ["DISPLAY_BUDGET", "elide"]
