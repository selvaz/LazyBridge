"""ApprovalQueue / StoreApprovalChannel — durable, Store-backed approval and
escalation tickets.

Ported from the project this queue was promoted from (LazyCEO's
``lazyceo.approvals``), including the live-incident regression tests (a
stuck sqlite snapshot re-surfacing an already-resolved ticket as pending)
that motivated both this queue's own ``_fresh_read_store`` guard and
``Store.write``/``delete``/``clear``'s rollback-on-failure recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import timedelta

import pytest

from lazybridge import Store
from lazybridge.ext.approval import ApprovalQueue, ApprovalTicket, Rule, StoreApprovalChannel, TieredGate, ticket_gist


async def _wait_until(predicate, *, timeout: float = 1.0, interval: float = 0.01) -> None:
    """Cancellation cleanup that races a cancelled `ask()` runs on its own,
    independent task (see `queue.py`'s `_retire_if_unclaimed`) precisely so
    that no amount of repeated cancellation can interrupt it -- which also
    means `await task` no longer blocks until that cleanup is done. Tests
    that assert on its effect (the ticket appearing / being retired) poll
    for it instead of asserting immediately after `await task`."""
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(interval)


def test_ticket_gist_falls_back_to_first_line_without_an_objective_marker() -> None:
    """A plain 'ask'-tier ticket's prompt IS just the request -- nothing to
    skip past, unlike a caller whose template puts boilerplate before the
    real objective."""
    assert ticket_gist("git commit -m 'fix the thing'") == "git commit -m 'fix the thing'"


def test_ticket_gist_finds_the_objective_past_boilerplate() -> None:
    prompt = (
        "Delegate to Codex with REAL write access. Codex's own git/file actions inside the "
        "workspace will NOT be individually gated once this is approved -- it can write, commit, etc. "
        "without asking again.\n\nObjective: build the walk-forward module"
    )
    assert ticket_gist(prompt) == "build the walk-forward module"


def test_ticket_gist_truncates_a_long_objective() -> None:
    assert ticket_gist("Objective: " + "x" * 300, max_len=50) == "x" * 47 + "..."


def test_ticket_gist_handles_an_empty_prompt() -> None:
    """create_ticket doesn't reject an empty/whitespace-only prompt --
    splitlines()[0] on one would raise IndexError instead of returning an
    empty gist. Found by Codex review before this ever shipped."""
    assert ticket_gist("") == ""
    assert ticket_gist("   \n  ") == ""


def test_ticket_gist_respects_a_max_len_too_small_to_fit_the_ellipsis() -> None:
    """gist[:max_len - 3] + "..." goes negative for max_len <= 3, and a
    negative slice returns almost the WHOLE string -- the opposite of
    truncating. Found by Codex review before this ever shipped."""
    gist = ticket_gist("Objective: " + "x" * 300, max_len=2)
    assert gist == "xx"
    assert len(gist) <= 2


def test_ticket_gist_rejects_a_negative_max_len() -> None:
    """max_len=-1 still hits a negative slice (gist[:-1]) unless the <= 0
    case is handled before any slicing is attempted. Found by Codex review
    before this ever shipped."""
    assert ticket_gist("Objective: " + "x" * 300, max_len=-1) == ""
    assert ticket_gist("short", max_len=0) == ""


def test_get_ticket_survives_a_cwd_change_from_a_different_thread(tmp_path, monkeypatch) -> None:
    """Store's connection is thread-local: a naive re-resolution of
    store._db per read would, from a NEW thread that has never called
    store._conn() before, open ITS OWN first connection relative to
    whatever cwd is active on THAT thread at THAT moment -- wrong if cwd
    already changed since the queue's ApprovalQueue was constructed on the
    main thread. Resolving once at construction and never touching
    store._db/store._conn() again sidesteps this regardless of which
    thread later calls get_ticket(). Found by Codex review (three rounds)
    before this ever shipped."""
    import threading

    monkeypatch.chdir(tmp_path)
    queue = ApprovalQueue(Store(db="approvals.sqlite"))
    ticket = queue.create_ticket(task_id="t1", prompt="p")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    results: list[ApprovalTicket | None] = []

    def _read_from_worker_thread() -> None:
        results.append(queue.get_ticket(ticket.approval_id))

    worker = threading.Thread(target=_read_from_worker_thread)
    worker.start()
    worker.join(timeout=5)

    assert results == [ticket]


def test_approve_ticket_survives_a_cwd_change_from_a_different_thread(tmp_path, monkeypatch) -> None:
    """The decision methods (approve/reject) must route through the SAME
    anchored database as the read methods -- not the raw ``store`` object
    directly, which a new thread (after a cwd change) would connect to
    relative to the wrong directory, silently creating/using an unrelated
    empty database and leaving the real, original ticket permanently
    pending. Found by Codex review (three rounds) before this ever
    shipped."""
    import threading

    monkeypatch.chdir(tmp_path)
    queue = ApprovalQueue(Store(db="approvals.sqlite"))
    ticket = queue.create_ticket(task_id="t1", prompt="p")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    results: list[bool] = []

    def _approve_from_worker_thread() -> None:
        results.append(queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram"))

    worker = threading.Thread(target=_approve_from_worker_thread)
    worker.start()
    worker.join(timeout=5)

    assert results == [True]
    assert queue.get_ticket(ticket.approval_id).status == "approved"


def test_get_ticket_survives_a_cwd_change_after_a_relative_db_path(tmp_path, monkeypatch) -> None:
    """Store constructed with a relative db path, then the process's cwd
    changes (a realistic thing for a long-running task to do) -- a naive
    re-resolve of the raw relative string against the NEW cwd would open a
    different, empty database and make every ticket vanish.
    _fresh_read_store must instead ask the original connection what file
    it actually has open. Found by Codex review (twice) before this ever
    shipped."""
    monkeypatch.chdir(tmp_path)
    queue = ApprovalQueue(Store(db="approvals.sqlite"))
    ticket = queue.create_ticket(task_id="t1", prompt="p")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert queue.get_ticket(ticket.approval_id) == ticket
    assert [t.approval_id for t in queue.list_pending_tickets()] == [ticket.approval_id]
    assert queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram") is True


def test_create_ticket_is_pending_and_hashed() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="approve Bash 'git push'?")
    assert ticket.status == "pending"
    assert ticket.kind == "approval"  # default, unchanged behavior for the existing gate
    assert len(ticket.prompt_hash) == 64
    assert queue.get_ticket(ticket.approval_id) == ticket


def test_create_ticket_accepts_an_escalation_kind() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="specialist-x", prompt="found something structurally off", kind="escalation")
    assert ticket.kind == "escalation"
    assert queue.get_ticket(ticket.approval_id).kind == "escalation"


def test_default_in_memory_store_is_safe_across_threads() -> None:
    """Store(db=None) (NOT Store(db=":memory:"), which is thread-local and
    NOT safe for this -- see ApprovalQueue's own docstring) is the
    recommended in-memory choice specifically because it IS safe from
    another thread, the normal shape for this queue (one thread files a
    ticket, another surface resolves it)."""
    import threading

    queue = ApprovalQueue(Store())  # db=None, the library's own thread-safe in-memory mode
    ticket = queue.create_ticket(task_id="t1", prompt="p")

    results: list[bool] = []

    def _approve_from_worker_thread() -> None:
        results.append(queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram"))

    worker = threading.Thread(target=_approve_from_worker_thread)
    worker.start()
    worker.join(timeout=5)

    assert results == [True]
    assert queue.get_ticket(ticket.approval_id).status == "approved"


def test_queue_works_with_sqlite_memory_special_filename() -> None:
    """SQLite's own convention: every sqlite3.connect(":memory:") call opens
    a brand new, unrelated anonymous database. _anchor_db_path must not
    treat db=":memory:" as a real path to reopen, or every ticket becomes
    invisible immediately after creation. Found by Codex review before
    this ever shipped.

    Single-threaded only, deliberately -- Store(db=":memory:") is NOT safe
    across threads (see ApprovalQueue's own docstring); this test proves
    same-thread reuse still works, not that the mode is generally safe."""
    queue = ApprovalQueue(Store(db=":memory:"))
    ticket = queue.create_ticket(task_id="t1", prompt="p")

    assert queue.get_ticket(ticket.approval_id) == ticket
    assert [t.approval_id for t in queue.list_pending_tickets()] == [ticket.approval_id]
    assert queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram") is True


def test_two_queues_with_different_prefixes_do_not_collide_on_one_store() -> None:
    """The whole point of a configurable prefix: two independent queues can
    share one Store."""
    store = Store()
    queue_a = ApprovalQueue(store, prefix="app-a:")
    queue_b = ApprovalQueue(store, prefix="app-b:")
    ticket_a = queue_a.create_ticket(task_id="t1", prompt="p1")
    queue_b.create_ticket(task_id="t1", prompt="p2")

    assert [t.approval_id for t in queue_a.list_pending_tickets()] == [ticket_a.approval_id]
    assert queue_b.get_ticket(ticket_a.approval_id) is None  # wrong queue, must not see it


async def test_store_approval_channel_stamps_its_configured_kind_on_tickets_it_creates() -> None:
    store = Store()
    queue = ApprovalQueue(store)

    async def approve_soon() -> None:
        await asyncio.sleep(0.03)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    channel = StoreApprovalChannel(queue, task_id="specialist-x", poll_seconds=0.01, kind="escalation")
    await asyncio.gather(channel.ask("please look into this"), approve_soon())

    [(_key, raw)] = store.items(prefix="approval:")
    assert raw["kind"] == "escalation"


async def test_store_approval_channel_bounds_a_stalled_notify_callback() -> None:
    """A notify callback that hangs (a stalled network transport, e.g.)
    must not block ask() from ever reaching its own status-polling/expiry
    loop -- otherwise a ticket approved through another surface a moment
    later would sit unnoticed for as long as the notify call never
    returns. Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())

    async def stalled_notify(ticket, message):
        await asyncio.sleep(999)  # never completes within this test

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=stalled_notify, notify_timeout=0.02)

    async def approve_soon() -> None:
        await asyncio.sleep(0.05)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.wait_for(asyncio.gather(channel.ask("please approve"), approve_soon()), timeout=2.0)
    assert result is True


async def test_notify_wait_does_not_overshoot_a_short_ttl() -> None:
    """notify_timeout alone (10s default) can still make ask() overshoot a
    short ttl by nearly that whole amount whenever a notify call stalls --
    the wait must be capped by whichever is shorter: notify_timeout, or
    the ticket's own remaining lifetime. Found by Codex review before this
    ever shipped."""
    queue = ApprovalQueue(Store())

    async def stalled_notify(ticket, message):
        await asyncio.sleep(999)  # never completes within this test

    channel = StoreApprovalChannel(
        queue,
        task_id="t1",
        poll_seconds=0.01,
        ttl=timedelta(seconds=0.05),
        notify=stalled_notify,
        # Default notify_timeout (10s) is deliberately left unset here --
        # the fix is that ttl (50ms) wins regardless, not that this test
        # passes a short notify_timeout to dodge the bug.
    )

    result = await asyncio.wait_for(channel.ask("please approve"), timeout=2.0)

    assert result is False


def test_approve_ticket_is_cas_second_call_fails() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="p")
    assert queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram") is True
    assert queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram") is False
    assert queue.get_ticket(ticket.approval_id).status == "approved"


def test_reject_after_approve_fails() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="p")
    queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")
    assert queue.reject_ticket(ticket.approval_id, actor="marco", channel="telegram", reason="too late") is False


def test_approve_unknown_ticket_returns_false() -> None:
    queue = ApprovalQueue(Store())
    assert queue.approve_ticket("nope", actor="marco", channel="telegram") is False


def test_approve_expired_ticket_fails() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="p", ttl=timedelta(seconds=-1))
    assert queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram") is False


def test_list_pending_tickets_excludes_resolved_and_expired() -> None:
    queue = ApprovalQueue(Store())
    pending = queue.create_ticket(task_id="t1", prompt="p1")
    resolved = queue.create_ticket(task_id="t1", prompt="p2")
    queue.approve_ticket(resolved.approval_id, actor="marco", channel="telegram")
    queue.create_ticket(task_id="t1", prompt="p3", ttl=timedelta(seconds=-1))

    ids = [t.approval_id for t in queue.list_pending_tickets()]
    assert ids == [pending.approval_id]


def test_get_ticket_never_re_surfaces_an_already_resolved_ticket_from_a_stuck_sqlite_snapshot(tmp_path) -> None:
    """Live regression (3x in the project this queue was promoted from, one
    case running ~24h with zero progress after the real approval): a ticket
    already ``approved``/``rejected`` in the store kept getting relayed as
    still-pending. Root cause: ``Store`` caches one sqlite3 connection per
    thread; a read transaction left open on that connection (e.g. after a
    failed write with no rollback -- ``Store.write``/``delete``/``clear``
    now recover from exactly that, see ``lazybridge/store/__init__.py``) is
    pinned to whatever snapshot existed when it first touched the ``store``
    table, and in WAL mode will never see a later commit from another
    connection for as long as that transaction stays open -- exactly the
    profile of a long-lived agent polling its own escalation channel for
    hours.

    This reproduces the stuck-transaction snapshot directly (an explicit
    ``BEGIN`` plus one real read of the table) rather than waiting out
    ``busy_timeout`` for the actual "database is locked" trigger, and
    proves both halves: the raw ``Store.read`` on the wedged connection IS
    stale (so the bug is real, not a straw man), while ``get_ticket`` --
    going through ``_fresh_read_store`` -- is not."""
    db = str(tmp_path / "approvals.sqlite")
    store = Store(db=db)
    queue = ApprovalQueue(store)
    ticket = queue.create_ticket(task_id="t1", prompt="p")

    # Wedge this Store's cached connection into an open read transaction
    # pinned to the current snapshot -- the same end state a failed
    # write/delete/clear (before the rollback-on-failure fix) leaves behind
    # on a long-lived Store.
    conn = store._conn()
    conn.execute("BEGIN")
    conn.execute("SELECT value FROM store WHERE key=?", (queue._key(ticket.approval_id),)).fetchone()

    # Resolved from a totally different connection, exactly like an
    # /approve handler talking to its own Store instance would.
    other_queue = ApprovalQueue(Store(db=db))
    assert other_queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram") is True
    other_queue._store.close()

    # The wedged connection's OWN read is provably stale -- confirms the
    # scenario is real, not just asserting the fix in isolation.
    assert store.read(queue._key(ticket.approval_id))["status"] == "pending"

    # get_ticket / list_pending_tickets must never be fooled by it.
    assert queue.get_ticket(ticket.approval_id).status == "approved"
    assert queue.list_pending_tickets() == []
    store.close()


async def test_store_approval_channel_calls_notify_with_ticket_and_message() -> None:
    """Without this, StoreApprovalChannel creates a ticket and polls it, but
    nothing ever tells a human it exists."""
    queue = ApprovalQueue(Store())
    notified = []

    async def notify(ticket, message):
        notified.append((ticket, message))

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=notify)

    async def approve_soon():
        await asyncio.sleep(0.03)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve this"), approve_soon())
    assert result is True
    assert len(notified) == 1
    ticket, message = notified[0]
    assert ticket.approval_id in message
    assert "please approve this" in message
    assert "/approve" in message and ticket.approval_id in message


async def test_store_approval_channel_survives_a_failing_notify() -> None:
    """A notify failure (e.g. a push provider down) must not fail the
    approval itself -- the ticket still exists and is still answerable
    through any surface that reads the queue directly."""
    queue = ApprovalQueue(Store())

    async def broken_notify(ticket, message):
        raise RuntimeError("notification channel unreachable")

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=broken_notify)

    async def approve_soon():
        await asyncio.sleep(0.03)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve"), approve_soon())
    assert result is True  # the broken notify did not prevent the approval from working


async def test_store_approval_channel_renotifies_a_ticket_still_pending_past_the_interval() -> None:
    """Found live: a ticket notified exactly once, then never mentioned
    again, left a whole turn blocked for close to an hour because the human
    simply never saw (or forgot) the first message. A short
    renotify_interval here stands in for the real 5-minute default."""
    queue = ApprovalQueue(Store())
    notified = []

    async def notify(ticket, message):
        notified.append(message)

    channel = StoreApprovalChannel(
        queue, task_id="t1", poll_seconds=0.01, notify=notify, renotify_interval=timedelta(seconds=0.03)
    )

    async def approve_after_two_reminders() -> None:
        while len(notified) < 3:  # the original notify + 2 reminders
            await asyncio.sleep(0.01)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve this"), approve_after_two_reminders())
    assert result is True
    assert len(notified) >= 3
    assert "please approve this" in notified[0]
    assert "Still waiting" in notified[1] and "please approve this" in notified[1]


async def test_store_approval_channel_stops_renotifying_once_resolved() -> None:
    queue = ApprovalQueue(Store())
    notified = []

    async def notify(ticket, message):
        notified.append(message)

    channel = StoreApprovalChannel(
        queue, task_id="t1", poll_seconds=0.01, notify=notify, renotify_interval=timedelta(seconds=0.03)
    )

    async def approve_soon() -> None:
        await asyncio.sleep(0.015)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    await asyncio.gather(channel.ask("please approve"), approve_soon())
    count_after_resolution = len(notified)
    await asyncio.sleep(0.1)  # long enough for several renotify intervals, if they fired
    assert len(notified) == count_after_resolution == 1


async def test_store_approval_channel_renotify_none_matches_old_single_notify_behaviour() -> None:
    queue = ApprovalQueue(Store())
    notified = []

    async def notify(ticket, message):
        notified.append(message)

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=notify, renotify_interval=None)

    async def approve_after_a_while() -> None:
        await asyncio.sleep(0.05)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve"), approve_after_a_while())
    assert result is True
    assert len(notified) == 1


async def test_store_approval_channel_returns_true_when_approved() -> None:
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    async def approve_soon() -> None:
        await asyncio.sleep(0.03)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve"), approve_soon())
    assert result is True


async def test_store_approval_channel_returns_false_when_rejected() -> None:
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    async def reject_soon() -> None:
        await asyncio.sleep(0.03)
        [ticket] = queue.list_pending_tickets()
        queue.reject_ticket(ticket.approval_id, actor="marco", channel="telegram", reason="no")

    result, _ = await asyncio.gather(channel.ask("please approve"), reject_soon())
    assert result is False


def test_poll_seconds_must_be_positive() -> None:
    """A nonpositive poll_seconds makes min(poll_seconds, remaining_ttl)
    resolve to sleep(0-or-negative) every iteration -- a hot loop of
    Store reads (and, for a file-backed queue, a fresh SQLite connection
    opened and closed every single iteration) until the ticket's TTL, up
    to hours, finally expires. Found by Codex review before this ever
    shipped."""
    queue = ApprovalQueue(Store())
    with pytest.raises(ValueError, match="poll_seconds"):
        StoreApprovalChannel(queue, task_id="t1", poll_seconds=0)
    with pytest.raises(ValueError, match="poll_seconds"):
        StoreApprovalChannel(queue, task_id="t1", poll_seconds=-1)


def test_renotify_interval_must_be_positive_or_none() -> None:
    """None is already the documented way to disable reminders -- zero or
    negative is not a smaller version of that, it's "always overdue":
    the reminder condition is true on EVERY poll, so a reminder fires at
    the full polling rate for up to the ticket's whole TTL. Found by
    Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    with pytest.raises(ValueError, match="renotify_interval"):
        StoreApprovalChannel(queue, task_id="t1", renotify_interval=timedelta(0))
    with pytest.raises(ValueError, match="renotify_interval"):
        StoreApprovalChannel(queue, task_id="t1", renotify_interval=timedelta(seconds=-1))
    # None itself must still be accepted -- it's the documented way to
    # disable reminders, not rejected by this same guard.
    StoreApprovalChannel(queue, task_id="t1", renotify_interval=None)


async def test_ask_does_not_block_the_event_loop() -> None:
    """ApprovalQueue's methods are synchronous Store I/O -- calling them
    directly from this async method would block the WHOLE event loop for
    as long as the call takes (Store's busy_timeout is 5s under write
    contention -- a shared Store with many agents filing tickets/
    heartbeats is exactly what produces that), freezing every other
    coroutine sharing it, not just this ticket's own progress.

    Asserts progress WHILE the slow call is still in flight, not just
    the final tick count after both coroutines have finished -- a
    gather()-then-check-the-total assertion passes either way (blocking
    only delays when the ticks happen, not whether all of them
    eventually do), so it wouldn't actually catch a regression back to
    calling ApprovalQueue directly. Found by Codex review before this
    ever shipped (twice: the production fix, and this test's own first
    version not actually distinguishing blocking from non-blocking)."""
    queue = ApprovalQueue(Store())
    real_create_ticket = queue.create_ticket

    def _slow_create_ticket(*args, **kwargs):
        time.sleep(0.2)
        return real_create_ticket(*args, **kwargs)

    queue.create_ticket = _slow_create_ticket  # type: ignore[method-assign]
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, ttl=timedelta(seconds=0.05))
    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    ask_task = asyncio.create_task(channel.ask("please approve"))
    ticker_task = asyncio.create_task(_ticker())

    # Checked at the halfway point of the 0.2s slow call, while it is
    # still running: if create_ticket ran directly on this event loop,
    # NOTHING else could execute until it returned, and ticks would
    # still be exactly 0 here.
    await asyncio.sleep(0.1)
    assert ticks > 0

    ticker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        _ = await ticker_task
    with contextlib.suppress(Exception):
        _ = await ask_task


async def test_ask_retires_the_ticket_on_a_non_cancellation_exception() -> None:
    """Any exceptional exit from the poll loop -- not just cancellation --
    must retire the ticket: a transient Store error (or anything else
    get_ticket can raise) would otherwise leave an actionable ticket
    nobody is waiting on anymore. The retiring reject_ticket call itself
    runs on a detached task (see queue.py's `_retire` inside ask()'s
    `except BaseException` handler), so it can outlive a repeated
    cancellation of ask() -- which also means it isn't guaranteed done the
    instant ask() itself raises; poll for it instead. Found by Codex
    review before this ever shipped."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    real_get_ticket = queue.get_ticket
    call_count = 0

    def _flaky_get_ticket(approval_id: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return real_get_ticket(approval_id)
        raise RuntimeError("transient sqlite error")

    queue.get_ticket = _flaky_get_ticket  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="transient sqlite error"):
            await channel.ask("please approve")
    finally:
        queue.get_ticket = real_get_ticket  # type: ignore[method-assign]

    await _wait_until(lambda: queue.list_pending_tickets() == [])
    assert queue.list_pending_tickets() == []


async def test_ask_retires_the_ticket_when_cancelled() -> None:
    """A cancelled ask() (a caller's own timeout, task shutdown, process
    teardown) leaves nothing to consume the eventual decision -- the
    ticket must not stay "pending" (still listed, still actionable by an
    operator who has no idea the coroutine that asked is gone) until its
    full TTL expiry hours later. The retiring reject_ticket call runs on a
    detached task so it can survive a repeated cancellation of ask()
    itself, which also means `await task` no longer waits for it to
    finish; poll for it instead. Found by Codex review before this ever
    shipped."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    task = asyncio.create_task(channel.ask("please approve"))
    await asyncio.sleep(0.03)  # let it create the ticket and start polling
    [ticket] = queue.list_pending_tickets()

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        _ = await task  # discarded on purpose -- awaiting only to let cancellation propagate/settle

    await _wait_until(lambda: queue.get_ticket(ticket.approval_id).status == "rejected")
    resolved = queue.get_ticket(ticket.approval_id)
    assert resolved.status == "rejected"
    assert queue.list_pending_tickets() == []


async def test_ask_retires_the_ticket_when_cancelled_during_creation() -> None:
    """A cancellation landing WHILE create_ticket's offloaded call is
    still in flight must not orphan the ticket that lands moments
    later -- the underlying thread-pool work can't actually be stopped
    once started, so this only works because ask() shields that specific
    call and hands its real outcome to an independent watcher task that
    retires it, regardless of what happens to ask() itself afterwards.
    Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    real_create_ticket = queue.create_ticket
    created: list[ApprovalTicket] = []

    def _slow_create_ticket(*args, **kwargs):
        time.sleep(0.1)
        ticket = real_create_ticket(*args, **kwargs)
        created.append(ticket)
        return ticket

    queue.create_ticket = _slow_create_ticket  # type: ignore[method-assign]
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    task = asyncio.create_task(channel.ask("please approve"))
    await asyncio.sleep(0.02)  # cancel WHILE create_ticket's own 0.1s sleep is still running
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        _ = await task  # ask() unwinds immediately -- the retiring watcher runs on its own, separate task

    await _wait_until(lambda: len(created) == 1)
    await _wait_until(lambda: queue.get_ticket(created[0].approval_id).status == "rejected")
    resolved = queue.get_ticket(created[0].approval_id)
    assert resolved.status == "rejected"


async def test_ask_retires_the_ticket_when_cancelled_twice_during_creation() -> None:
    """A SECOND cancellation (and, in principle, any further one after
    that) landing while ask() is still unwinding from the first must not
    abandon the pending ticket -- asyncio.TaskGroup cancelling every
    remaining sibling when one fails is a real way this can happen, not
    just theoretical. Retiring it can't depend on ask()'s own coroutine
    surviving long enough to do the cleanup itself: the ticket is retired
    by an independent watcher task that nothing here ever cancels, so it
    finishes regardless of how many more times `task` itself is cancelled.
    Found by Codex review before this ever shipped; the first fix attempt
    (nesting a second `asyncio.shield()` inside ask()'s own cancellation
    handler) was proven insufficient by a standalone repro -- shield()
    protects the awaited future from cancellation, not the coroutine
    doing the awaiting from being cancelled again."""
    queue = ApprovalQueue(Store())
    real_create_ticket = queue.create_ticket
    created: list[ApprovalTicket] = []

    def _slow_create_ticket(*args, **kwargs):
        time.sleep(0.15)
        ticket = real_create_ticket(*args, **kwargs)
        created.append(ticket)
        return ticket

    queue.create_ticket = _slow_create_ticket  # type: ignore[method-assign]
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    task = asyncio.create_task(channel.ask("please approve"))
    await asyncio.sleep(0.02)
    task.cancel()  # first cancellation -- create_ticket's 0.15s sleep is still running
    await asyncio.sleep(0.02)
    task.cancel()  # second cancellation -- ask() is still unwinding from the first
    with contextlib.suppress(asyncio.CancelledError):
        _ = await task

    await _wait_until(lambda: len(created) == 1)
    await _wait_until(lambda: queue.get_ticket(created[0].approval_id).status == "rejected")
    resolved = queue.get_ticket(created[0].approval_id)
    assert resolved.status == "rejected"


async def test_ask_retires_the_ticket_when_cancelled_twice_after_creation() -> None:
    """The same double-cancellation hazard exists on the OTHER retirement
    path: a second cancellation landing while ask()'s post-creation
    cleanup is (slowly) rejecting an already-created ticket must not
    abandon that rejection either -- `contextlib.suppress(Exception)`
    alone does not catch a second `asyncio.CancelledError` (a
    `BaseException`) raised into that await. Detaching the reject_ticket
    call onto its own task, exactly like the creation-path fix, is what
    makes ask() itself unwind on the first cancellation without leaving
    anything further for a second one to interrupt. Found by Codex review
    before this ever shipped."""
    queue = ApprovalQueue(Store())
    real_reject_ticket = queue.reject_ticket

    def _slow_reject_ticket(*args, **kwargs):
        time.sleep(0.1)
        return real_reject_ticket(*args, **kwargs)

    queue.reject_ticket = _slow_reject_ticket  # type: ignore[method-assign]
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    task = asyncio.create_task(channel.ask("please approve"))
    await asyncio.sleep(0.03)  # let it create the ticket and start polling
    [ticket] = queue.list_pending_tickets()

    task.cancel()  # first cancellation -- triggers the (slow) retiring reject_ticket call
    await asyncio.sleep(0.02)
    task.cancel()  # second cancellation -- lands after ask() has already unwound
    with contextlib.suppress(asyncio.CancelledError):
        _ = await task

    await _wait_until(lambda: queue.get_ticket(ticket.approval_id).status == "rejected")
    resolved = queue.get_ticket(ticket.approval_id)
    assert resolved.status == "rejected"


async def test_ask_works_with_sqlite_memory_special_filename() -> None:
    """StoreApprovalChannel must not unconditionally offload to a worker
    thread: Store(db=":memory:") relies on reusing the SAME thread's own
    connection, and asyncio.to_thread hands each call to a (possibly
    different) thread-pool worker, each opening ITS OWN unrelated empty
    in-memory database. Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store(db=":memory:"))
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    async def approve_soon() -> None:
        await asyncio.sleep(0.03)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve"), approve_soon())
    assert result is True


async def test_message_includes_the_ticket_id_in_the_reply_commands() -> None:
    """No bare "/approve"-resolves-the-one-pending-ticket convenience
    exists in this library -- the message must be self-sufficient with
    more than one pending ticket, or for a stateless CLI/API handler.
    Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    notified = []

    async def notify(ticket, message):
        notified.append((ticket, message))

    async def approve_soon() -> None:
        await asyncio.sleep(0.03)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=notify)
    await asyncio.gather(channel.ask("please approve"), approve_soon())

    ticket, message = notified[0]
    assert f"/approve {ticket.approval_id}" in message
    assert f"/reject {ticket.approval_id} <reason>" in message


async def test_reminder_send_time_does_not_delay_the_ttl_check() -> None:
    """The post-reminder sleep must be computed from the time AFTER that
    reminder was sent, not the stale timestamp from before it -- otherwise
    a slow (but not stalled) reminder notify pushes ask()'s return well
    past the configured ttl. Found by Codex review before this ever
    shipped."""
    queue = ApprovalQueue(Store())

    async def slow_notify(ticket, message):
        await asyncio.sleep(0.08)

    channel = StoreApprovalChannel(
        queue,
        task_id="t1",
        poll_seconds=0.11,
        ttl=timedelta(seconds=0.2),
        notify=slow_notify,
        renotify_interval=timedelta(seconds=0.05),
    )

    start = time.monotonic()
    result = await asyncio.wait_for(channel.ask("please approve"), timeout=2.0)
    elapsed = time.monotonic() - start

    assert result is False
    # Unfixed: ~0.2s ttl + ~0.08s stale-reminder overshoot =~ 0.28s+.
    # Fixed: close to the true 0.2s ttl. 0.26s leaves comfortable margin
    # on both sides without making the test flaky.
    assert elapsed < 0.26


async def test_store_approval_channel_times_out_to_false() -> None:
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, ttl=timedelta(seconds=0.02))
    assert await channel.ask("please approve") is False


async def test_store_approval_channel_does_not_overshoot_ttl_when_poll_seconds_is_larger() -> None:
    """An unconditional sleep(poll_seconds) would overshoot expires_at by
    up to a full poll interval whenever poll_seconds > ttl -- a valid, if
    unusual, combination since both are independent caller-set parameters.
    Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=60.0, ttl=timedelta(seconds=0.05))

    result = await asyncio.wait_for(channel.ask("please approve"), timeout=2.0)

    assert result is False


async def test_tiered_gate_end_to_end_through_store_approval_channel() -> None:
    """The actual integration point: TieredGate's `ask` tier, backed by the
    shared ticket queue instead of a direct human prompt."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash", "git commit*"),))

    from lazybridge.engines.coding import ApprovalRequest

    async def resolve_soon() -> None:
        await asyncio.sleep(0.03)
        pending = queue.list_pending_tickets()
        assert len(pending) == 1
        queue.approve_ticket(pending[0].approval_id, actor="marco", channel="telegram")

    request = ApprovalRequest(
        provider="claude-code", kind="tool", name="Bash", arguments={"command": "git commit -m x"}, cwd="C:/repo"
    )
    decision, _ = await asyncio.gather(gate(request), resolve_soon())
    assert decision.action == "allow"
