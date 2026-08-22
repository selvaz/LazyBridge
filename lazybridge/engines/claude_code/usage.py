"""Claude Code's own ``/usage`` report, parsed into structured budgets.

Why this exists instead of reading a typed field: there isn't one. The Agent
SDK's :class:`~claude_agent_sdk.RateLimitEvent` arrives free on every run, but
on a current account its ``utilization`` is ``None`` — the payload simply
lacks the figure. Scanning every message type of a live run for anything else
carrying a percentage finds nothing (verified 2026-08-21: twelve messages
across ``SystemMessage``/``RateLimitEvent``/``AssistantMessage``/
``ResultMessage``, one usable percentage, and it wasn't in any of them).

The percentage exists in exactly one place: the prose the CLI's own
``/usage`` slash command prints. So this module is fundamentally a
screen-scrape of a CLI's human-facing report, with everything that implies —
the wording can change release to release, and a failed parse must say so
rather than invent a number. Every field this fails to extract is ``None``;
the raw text always travels alongside the parsed one so a caller can fall
back to it or notice drift.

    from lazybridge.engines.claude_code.usage import fetch_claude_usage

    snapshot = await fetch_claude_usage()
    for label, window in snapshot.weekly.items():
        print(label, window.used_percent, "%, resets", window.resets_at)

Or, with a running engine, its own ``model``/``cwd`` and (when supplied) its
injected test client apply automatically::

    snapshot = await engine.usage()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import ClaudeSdkClient

# `/usage` is a meta command the CLI answers directly, not a question routed
# to the model -- it costs no completion of its own, so one turn is enough
# (measured live: returns in ~13s including process startup).
_DEFAULT_MAX_TURNS = 1
_DEFAULT_TIMEOUT = 60.0

_SESSION_RE = re.compile(
    r"Current session:\s*(?P<pct>\d+)%\s*used\s*[·\-]\s*resets\s*(?P<reset>.+)",
    re.IGNORECASE,
)
_WEEK_RE = re.compile(
    r"Current week \((?P<label>[^)]+)\):\s*(?P<pct>\d+)%\s*used\s*[·\-]\s*resets\s*(?P<reset>.+)",
    re.IGNORECASE,
)
# The reset clause itself: "Aug 25, 12pm (America/Los_Angeles)". Minutes and
# the timezone are both optional in principle -- captured as such rather than
# assumed, because an absent one should fail the parse, not fabricate a UTC
# offset or a midnight the CLI never printed.
_RESET_RE = re.compile(
    r"(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2}),?\s*"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)"
    r"(?:\s*\((?P<tz>[^)]+)\))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UsageWindow:
    """One budget figure: how much of it is used, and when it resets.

    ``resets_at`` is ``None`` whenever the reset clause did not match the
    expected shape — most likely because it carried no recognisable timezone,
    or the CLI's wording changed. ``resets_raw`` is never ``None`` once the
    window itself was found, so the original text is never lost even when the
    timestamp could not be derived from it.
    """

    used_percent: int
    resets_raw: str
    resets_at: datetime | None = None


@dataclass(frozen=True)
class ClaudeUsageSnapshot:
    """One read of ``/usage``.

    ``weekly`` is keyed by whatever label the CLI prints in parentheses —
    ``"all models"`` plus one entry per model it breaks out separately (e.g.
    ``"Fable"``, observed live 2026-08-21). The label set is not fixed by
    this module: a new one appearing is a new dict key, not a schema change,
    and an old one disappearing is simply absent from ``weekly``.
    """

    session: UsageWindow | None
    weekly: dict[str, UsageWindow]
    raw_text: str
    fetched_at: datetime

    @property
    def parsed(self) -> bool:
        """Whether at least one recognisable window was found.

        ``False`` means the report's wording no longer matches what this
        module expects — a real possibility for a screen-scrape of a CLI's
        own prose — and callers should fall back to ``raw_text`` rather than
        trust an empty ``weekly``/``session`` as "nothing is being used".
        """
        return self.session is not None or bool(self.weekly)

    def most_used(self) -> tuple[str, UsageWindow] | None:
        """The weekly window closest to its limit — the one worth watching."""
        if not self.weekly:
            return None
        return max(self.weekly.items(), key=lambda item: item[1].used_percent)


def _parse_reset(text: str, *, now: datetime) -> datetime | None:
    """``now`` must be timezone-aware — enforced by :func:`parse_usage_report`."""
    match = _RESET_RE.search(text)
    if match is None or match.group("tz") is None:
        # No timezone in the clause: guessing one (UTC, the machine's local
        # zone) would silently attach an hour that was never printed. Better
        # to report the reset as unparsed than to be precisely wrong.
        return None
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            tz = ZoneInfo(match.group("tz"))
        except ZoneInfoNotFoundError:
            # Either a genuinely unknown zone name, or (on Windows without
            # the `tzdata` package -- see the `claude-code` extra) a real
            # one this interpreter simply cannot resolve. Either way, no
            # timestamp can be built from it.
            return None

        hour = int(match.group("hour"))
        ampm = match.group("ampm").lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        minute = int(match.group("minute") or 0)
        month_day = f"{match.group('month')[:3].title()} {match.group('day')}"
        # The report carries no year. Anchored to `now`'s year IN THE
        # REPORT'S OWN ZONE, not `now`'s -- across the hours where the two
        # zones straddle midnight on 31 December they disagree, and using
        # the wrong one turns "resets in half an hour" into "resets in a
        # year" (a real case: now=2027-01-01 00:30 UTC with a reset of
        # "Dec 31, 5pm (America/Los_Angeles)" is 2026-12-31 in that zone).
        year = now.astimezone(tz).year
        naive = datetime.strptime(f"{month_day} {hour}:{minute:02d} {year}", "%b %d %H:%M %Y")
        # DST fall-back repeats one wall-clock hour. The report gives no UTC
        # offset to tell the two occurrences apart, so silently assuming the
        # first (`fold=0`) would be exactly the kind of invented precision
        # this function otherwise refuses to produce -- leave it unparsed
        # when the hour is genuinely ambiguous instead of guessing.
        first = naive.replace(tzinfo=tz, fold=0)
        if first.utcoffset() != naive.replace(tzinfo=tz, fold=1).utcoffset():
            return None
        stamp = first
    except (ValueError, KeyError):
        return None
    if stamp < now - timedelta(days=1):
        stamp = stamp.replace(year=stamp.year + 1)
    return stamp


def parse_usage_report(text: str, *, now: datetime | None = None) -> ClaudeUsageSnapshot:
    """Turn ``/usage``'s printed report into a :class:`ClaudeUsageSnapshot`.

    Exposed standalone (not only via :func:`fetch_claude_usage`) so a report
    captured elsewhere — a saved transcript, a value pasted from a terminal —
    can be parsed without spending a turn to fetch a fresh one.

    Args:
        text: the report as the CLI printed it.
        now: reference instant for resolving the year and detecting a stale
            reset that must roll into the next one. Defaults to the current
            moment. **Must be timezone-aware** — reset times carry their own
            zone from the report, and comparing them against a naive ``now``
            is a bug this function refuses to attempt rather than raise a
            confusing ``TypeError`` deep inside the parser.

    Raises:
        ValueError: ``now`` was given naive (no ``tzinfo``).
    """
    if now is None:
        now = datetime.now().astimezone()
    elif now.tzinfo is None:
        raise ValueError(
            "parse_usage_report(now=...) must be timezone-aware: a naive datetime cannot be "
            "compared against reset times that carry their own zone from the report."
        )

    session: UsageWindow | None = None
    match = _SESSION_RE.search(text)
    if match:
        reset_raw = match.group("reset").strip()
        session = UsageWindow(int(match.group("pct")), reset_raw, _parse_reset(reset_raw, now=now))

    weekly: dict[str, UsageWindow] = {}
    for match in _WEEK_RE.finditer(text):
        reset_raw = match.group("reset").strip()
        weekly[match.group("label").strip()] = UsageWindow(
            int(match.group("pct")), reset_raw, _parse_reset(reset_raw, now=now)
        )

    return ClaudeUsageSnapshot(session=session, weekly=weekly, raw_text=text, fetched_at=now)


async def fetch_claude_usage(
    *,
    model: str | None = None,
    cwd: str | None = None,
    max_turns: int = _DEFAULT_MAX_TURNS,
    timeout: float = _DEFAULT_TIMEOUT,
    client: ClaudeSdkClient | None = None,
) -> ClaudeUsageSnapshot:
    """Run Claude Code's own ``/usage`` and parse the answer.

    Costs one small turn — verified live at roughly 13 seconds including
    process startup, no completion of its own since ``/usage`` is answered
    by the CLI directly rather than routed to the model.

    Goes through the same :class:`~lazybridge.engines.claude_code.protocol.ClaudeSdkClient`
    boundary :class:`~lazybridge.engines.claude_code.ClaudeCodeEngine` itself
    uses — the real ``AgentSdkClient`` by default, or a caller-supplied fake
    for tests — rather than a second, parallel entry point into the Agent
    SDK. ``ClaudeCodeEngine.usage()`` is a thin wrapper that passes its own
    ``model``/``cwd`` and (when the engine was built with one) its injected
    client.

    ``setting_sources`` is deliberately empty, matching this engine's default
    posture: this call should read the ambient account's usage, not have its
    behaviour altered by a project's local settings.

    Raises:
        ImportError: the Claude Agent SDK is not installed.
        RuntimeError: the SDK ended without a result, or the CLI reported the
            turn itself as an error (auth failure, no active session, etc.) —
            both distinguishable from "usage is simply unparsed", which
            returns normally with :attr:`ClaudeUsageSnapshot.parsed` ``False``.
        TimeoutError: no result within ``timeout`` seconds.
    """
    import asyncio

    from .protocol import ClaudeSdkOptions
    from .sdk_client import AgentSdkClient

    resolved_client = client or AgentSdkClient()
    options = ClaudeSdkOptions(model=model, cwd=cwd, max_turns=max_turns, setting_sources=())
    result = await asyncio.wait_for(resolved_client.run("/usage", options=options), timeout=timeout)
    return parse_usage_report(result.text)


__all__ = ["ClaudeUsageSnapshot", "UsageWindow", "fetch_claude_usage", "parse_usage_report"]
