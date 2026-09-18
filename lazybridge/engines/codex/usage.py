"""Codex subscription quota, read from the App Server.

Unlike the Claude side -- which screen-scrapes the CLI's ``/usage`` prose
because no typed field exists -- Codex answers a structured request::

    {"method": "account/rateLimits/read", "id": 2}

and returns ``usedPercent``, ``windowDurationMins`` and ``resetsAt`` per
limit bucket. No thread and no turn are involved, so reading the quota
costs nothing against it.

Verified live on this machine (18/09/2026, codex-cli under
``%LOCALAPPDATA%\\OpenAI\\Codex``): the ``codex`` bucket reported
``usedPercent=85`` for a ``windowDurationMins=10080`` window. That number
had been invisible to the whole system until it was asked for.

    from lazybridge.engines.codex.usage import fetch_codex_usage

    snapshot = await fetch_codex_usage()
    weekly = snapshot.weekly()
    print(weekly.used_percent, weekly.resets_at)

Two fields, never one: a percentage without its reset time cannot be
judged. 85% is comfortable on the first day of a window and critical on
the last.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .app_server import codex_executable

#: A seven-day window, in the unit the server reports.
WEEKLY_WINDOW_MINUTES = 10080

#: How long to wait for the handshake plus the read. The App Server answers
#: in well under a second when healthy; this bounds a hung child process.
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CodexUsageWindow:
    """One quota bucket: how much is gone, and when it comes back.

    ``resets_at`` is timezone-aware UTC. ``reached_type`` is the server's
    own classification of an exhausted limit, ``None`` while there is room.
    """

    limit_id: str
    kind: str  # "primary" or "secondary"
    used_percent: float
    window_duration_minutes: int
    resets_at: datetime | None
    reached_type: str | None = None

    @property
    def is_weekly(self) -> bool:
        return self.window_duration_minutes == WEEKLY_WINDOW_MINUTES

    def hours_until_reset(self, *, now: datetime | None = None) -> float | None:
        if self.resets_at is None:
            return None
        return (self.resets_at - (now or datetime.now(UTC))).total_seconds() / 3600.0

    def elapsed_fraction(self, *, now: datetime | None = None) -> float | None:
        """How far through the window we are, 0.0 to 1.0.

        The pair to ``used_percent``: consumption only means something
        against the time that produced it.
        """
        remaining = self.hours_until_reset(now=now)
        if remaining is None or self.window_duration_minutes <= 0:
            return None
        total_hours = self.window_duration_minutes / 60.0
        elapsed = (total_hours - remaining) / total_hours
        return min(max(elapsed, 0.0), 1.0)

    def projected_end_percent(self, *, now: datetime | None = None) -> float | None:
        """Where this window lands if consumption continues at this pace.

        ``None`` when too little of the window has passed to say anything:
        early on, the divisor is tiny and the projection is noise. A
        controller must react to the forecast rather than the position, but
        an unstable forecast is worse than none.
        """
        elapsed = self.elapsed_fraction(now=now)
        if elapsed is None or elapsed < 0.05:
            return None
        return self.used_percent / elapsed


@dataclass(frozen=True)
class CodexUsageSnapshot:
    """Every bucket the server reported, plus the raw payload.

    ``raw`` always travels with the parsed form. The response shape has
    already gained fields (``rateLimitsByLimitId`` beside the older
    ``rateLimits``), so a caller that needs something this dataclass does
    not model can still reach it without waiting for a release.
    """

    windows: tuple[CodexUsageWindow, ...]
    plan_type: str | None
    raw: dict[str, Any]

    def weekly(self, *, limit_id: str = "codex") -> CodexUsageWindow | None:
        """The seven-day bucket for one limit, identified by its DURATION.

        Not by position: ``primary`` is not always the weekly one, and a
        controller that assumes it is will silently pace against a
        five-hour window instead.
        """
        for window in self.windows:
            if window.limit_id == limit_id and window.is_weekly:
                return window
        return None


def _window(limit_id: str, kind: str, payload: object) -> CodexUsageWindow | None:
    if not isinstance(payload, dict):
        return None
    used = payload.get("usedPercent")
    duration = payload.get("windowDurationMins")
    if not isinstance(used, (int, float)) or not isinstance(duration, int):
        # A bucket missing its own figures is absent, not zero. Reporting
        # 0% for "the server did not say" is the kind of confident wrong
        # number this module exists to avoid.
        return None
    resets_raw = payload.get("resetsAt")
    resets_at = datetime.fromtimestamp(resets_raw, UTC) if isinstance(resets_raw, (int, float)) else None
    return CodexUsageWindow(
        limit_id=limit_id,
        kind=kind,
        used_percent=float(used),
        window_duration_minutes=duration,
        resets_at=resets_at,
        reached_type=payload.get("rateLimitReachedType"),
    )


def parse_rate_limits(result: dict[str, Any]) -> CodexUsageSnapshot:
    """Parse an ``account/rateLimits/read`` result into buckets."""
    buckets: dict[str, dict[str, Any]] = {}
    by_id = result.get("rateLimitsByLimitId")
    if isinstance(by_id, dict) and by_id:
        # Preferred: the multi-bucket view. Falling back to the single
        # `rateLimits` view would silently drop every limit but one.
        for key, value in by_id.items():
            if isinstance(value, dict):
                buckets[str(value.get("limitId") or key)] = value
    else:
        single = result.get("rateLimits")
        if isinstance(single, dict):
            buckets[str(single.get("limitId") or "codex")] = single

    windows: list[CodexUsageWindow] = []
    plan_type: str | None = None
    for limit_id, bucket in buckets.items():
        plan_type = plan_type or bucket.get("planType")
        for kind in ("primary", "secondary"):
            window = _window(limit_id, kind, bucket.get(kind))
            if window is not None:
                windows.append(window)
    return CodexUsageSnapshot(windows=tuple(windows), plan_type=plan_type, raw=result)


async def fetch_codex_usage(
    *, executable: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CodexUsageSnapshot:
    """Ask the App Server for the account's quota.

    Raises ``RuntimeError`` if the server cannot be reached or refuses the
    method -- an older binary answers "method not found", and the caller
    must treat that as *unknown*, never as *unlimited*.
    """
    exe = executable or codex_executable()
    process = await asyncio.create_subprocess_exec(
        exe,
        "app-server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def send(message: dict[str, Any]) -> None:
        assert process.stdin is not None
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()

    async def await_id(wanted: int) -> dict[str, Any]:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                raise RuntimeError("codex app-server closed the stream before answering")
            try:
                message: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue  # a non-JSON line is noise, not an answer
            if message.get("id") == wanted:
                return message

    try:
        async with asyncio.timeout(timeout):
            await send(
                {
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "clientInfo": {"name": "lazybridge", "title": "LazyBridge", "version": "1"},
                        "capabilities": {"experimentalApi": True},
                    },
                }
            )
            await await_id(1)
            # The documented lifecycle: `initialized` must follow the
            # handshake before an account request is accepted.
            await send({"method": "initialized", "params": {}})
            await send({"method": "account/rateLimits/read", "id": 2})
            answer = await await_id(2)
    except TimeoutError as exc:
        raise RuntimeError(f"codex app-server did not answer within {timeout:g}s") from exc
    finally:
        try:
            process.kill()
        except ProcessLookupError:
            # Already gone -- it answered and exited on its own. Nothing to
            # clean up, and raising here would replace a successful read
            # with a failure about the teardown.
            pass
        await process.wait()

    if "error" in answer:
        raise RuntimeError(f"codex app-server refused the quota read: {answer['error']}")
    result = answer.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("codex app-server returned no result for the quota read")
    return parse_rate_limits(result)
