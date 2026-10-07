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
import itertools
import logging
import os
import time
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from lazybridge import Store
from lazybridge.ext.approval import (
    ApprovalQueue,
    ApprovalTicket,
    Rule,
    StoreApprovalChannel,
    TerminalChannel,
    TieredGate,
    ticket_gist,
)
from lazybridge.ext.approval.queue import DEFAULT_DENIAL_PREFIX, DEFAULT_NOTIFY_FAILURE_PREFIX


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


def test_safe_to_call_from_any_thread_unwraps_an_encrypted_store() -> None:
    """EncryptedStoreAdapter documents itself as usable wherever a plain
    Store is -- but exposes none of Store's own _db/_local attributes
    itself (those belong to the Store it delegates to), so inspecting the
    adapter directly would misreport an encrypted Store(db=":memory:") as
    thread-safe: exactly the cross-thread breakage this property exists
    to prevent, just one layer removed. Found by Codex review before this
    ever shipped."""
    # pytest.importorskip (not a plain try/except ImportError + pytest.skip)
    # so `Fernet` is unconditionally bound afterward -- static analysis
    # can't know pytest.skip() never returns, and flags the try/except
    # shape as leaving `Fernet` possibly-uninitialized on the line below.
    # Found by CodeQL before this ever shipped.
    Fernet = pytest.importorskip("cryptography.fernet").Fernet
    from lazybridge.store.encryption import EncryptedStoreAdapter

    key = Fernet.generate_key()
    unsafe = ApprovalQueue(EncryptedStoreAdapter(Store(db=":memory:"), key=key))
    assert unsafe.safe_to_call_from_any_thread is False

    safe = ApprovalQueue(EncryptedStoreAdapter(Store(db=None), key=key))
    assert safe.safe_to_call_from_any_thread is True


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
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
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
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.wait_for(asyncio.gather(channel.ask("please approve"), approve_soon()), timeout=2.0)
    assert result is True


async def test_notify_wait_does_not_overshoot_a_short_ttl() -> None:
    """notify_timeout alone (10s default) can still make ask() overshoot a
    short ttl by nearly that whole amount whenever a notify call stalls --
    the INITIAL send must be capped by whichever is shorter: notify_timeout,
    or the ticket's own remaining lifetime. Found by Codex review before
    this ever shipped.

    notify_timeout IS set here, unlike this test's own original form --
    still far longer than ttl (so the INITIAL send's own ttl-wins guarantee
    stays exercised), but no longer the 10s default: the ticket now also
    gets an EXPIRY notice once ttl elapses (a later addition), and that
    terminal notice is deliberately bounded by notify_timeout alone (there
    is no "remaining ttl" left to cap it by; see
    StoreApprovalChannel._send's own docstring), so a stalled notifier
    paired with the 10s default would make ask() overshoot this test's own
    outer wait_for budget for a reason unrelated to what this test checks.
    """
    queue = ApprovalQueue(Store())

    async def stalled_notify(ticket, message):
        await asyncio.sleep(999)  # never completes within this test

    channel = StoreApprovalChannel(
        queue,
        task_id="t1",
        poll_seconds=0.01,
        ttl=timedelta(seconds=0.05),
        notify=stalled_notify,
        notify_timeout=0.5,
    )

    result = await asyncio.wait_for(channel.ask("please approve"), timeout=2.0)

    assert result is False


async def test_send_does_not_wait_for_a_notify_that_ignores_cancellation() -> None:
    """Plain `asyncio.wait_for(self._notify(...), timeout=...)` cancels the
    notify call on timeout and then WAITS for it to actually finish
    cancelling before raising -- a notifier that catches CancelledError to
    run its own cleanup (or simply ignores it for a while) can make that
    wait, and therefore notify_timeout itself, take far longer than
    configured. Shielding the wait from the notify task, and never
    awaiting the notify task's own cancellation, is what keeps this
    bounded regardless of how the notifier behaves. Found by Codex review
    before this ever shipped."""
    queue = ApprovalQueue(Store())
    notify_finished = asyncio.Event()

    async def stubborn_notify(ticket, message):
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)  # ignores the cancellation for a while
            raise
        finally:
            notify_finished.set()

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=stubborn_notify, notify_timeout=0.02)

    async def approve_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    loop = asyncio.get_event_loop()
    started = loop.time()
    result, _ = await asyncio.wait_for(asyncio.gather(channel.ask("please approve"), approve_soon()), timeout=1.0)
    elapsed = loop.time() - started

    assert result is True
    # Well under the notifier's 0.3s cancellation-ignoring delay: the old,
    # unshielded wait_for would have had to sit through that delay before
    # ask() could even reach its polling loop.
    assert elapsed < 0.25

    # stubborn_notify's own detached cleanup (see _send()'s `_detach`)
    # keeps running in the background after ask() has already returned --
    # wait for it to actually finish before this test (and its event loop)
    # tears down, or pytest-asyncio can destroy it mid-flight ("Task was
    # destroyed but it is pending!"). Found by Codex review before this
    # ever shipped.
    await asyncio.wait_for(notify_finished.wait(), timeout=1.0)


async def test_send_does_not_pile_up_notify_tasks_for_a_permanently_stuck_notifier() -> None:
    """A notifier that never respects cancellation at all (not just a slow
    one -- one that ignores it forever) must not accumulate one detached
    background task per renotification: every renotify interval firing a
    NEW _send() on top of an already-stuck previous one would leak an
    unbounded number of permanently-running tasks (and whatever resources
    the notifier itself holds -- a socket, a thread) over a long-lived
    ticket. Refusing to start a second notify FOR THIS TICKET while the
    first is still in flight caps the damage at the ONE lingering task the
    very first stuck call leaves behind, not one per reminder. Found by
    Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    call_count = 0
    release = asyncio.Event()

    async def stuck_notify(ticket, message):
        nonlocal call_count
        call_count += 1
        while not release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0.01)

    channel = StoreApprovalChannel(
        queue,
        task_id="t1",
        poll_seconds=0.01,
        notify=stuck_notify,
        notify_timeout=0.02,
        renotify_interval=timedelta(seconds=0.05),
    )

    async def approve_after(delay: float) -> None:
        await asyncio.sleep(delay)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    # 0.3s / a 0.05s renotify_interval is roughly six reminder ticks --
    # without the single-flight guard, each would spawn its own
    # permanently-stuck notify task.
    result = await asyncio.wait_for(asyncio.gather(channel.ask("please approve"), approve_after(0.3)), timeout=2.0)

    assert result[0] is True
    assert call_count == 1

    # Let the one lingering stuck_notify task (still looping until
    # released) actually exit before this test's event loop tears down.
    release.set()
    await asyncio.sleep(0.05)


async def test_stuck_notify_for_one_ticket_does_not_suppress_another() -> None:
    """One channel instance can file MORE than one ticket over its life --
    the class docstring's "every ticket this channel files" is deliberately
    plural, and TieredGate can call ask() repeatedly, or concurrently, on
    the same channel. The single-flight stuck-notify guard must key on
    each ticket's own approval_id, not the channel as a whole -- otherwise
    one ticket's stuck notifier would silently swallow the notification
    (and every reminder) for a completely unrelated second ticket this
    same channel is also handling. Found by Codex review before this ever
    shipped."""
    queue = ApprovalQueue(Store())
    notified: list[str] = []
    release = asyncio.Event()

    async def notify(ticket: ApprovalTicket, message: str) -> None:
        if ticket.prompt == "stuck one":
            while not release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(0.01)
            return
        notified.append(ticket.prompt)

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=notify, notify_timeout=0.02)

    # A separate, explicitly-managed task rather than something gathered
    # to completion: this ticket's own notify never releases on its own
    # (DEFAULT_TTL is hours), so ask() for it would never return.
    stuck_task = asyncio.create_task(channel.ask("stuck one"))
    await asyncio.sleep(0.05)  # let its ticket get created, notified, time out, and get detached

    async def approve_the_normal_one() -> None:
        def _find() -> list[ApprovalTicket]:
            return [t for t in queue.list_pending_tickets() if t.prompt == "normal one"]

        while not _find():
            await asyncio.sleep(0.005)
        [normal_ticket] = _find()
        queue.approve_ticket(normal_ticket.approval_id, actor="marco", channel="telegram")

    normal_result, _ = await asyncio.wait_for(
        asyncio.gather(channel.ask("normal one"), approve_the_normal_one()), timeout=2.0
    )

    assert normal_result is True
    assert notified == ["normal one"]

    release.set()
    stuck_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        _ = await stuck_task
    await asyncio.sleep(0.05)  # let the released stuck_notify loop actually exit


async def test_send_swallows_a_notifier_that_cancels_itself() -> None:
    """A notify callback whose OWN internal implementation ends up
    cancelled -- it manages a sub-task and cancels IT, unrelated to
    anything cancelling ask()/_send() from outside -- must be treated like
    any other notify failure: logged and swallowed, not mistaken for
    _send() itself having been cancelled. shield() is what makes the two
    distinguishable: cancelling the code AWAITING a shielded future never
    touches the shielded task, so notify_task can only be `.cancelled()`
    here if it finished that way on its own. Found by Codex review before
    this ever shipped."""
    queue = ApprovalQueue(Store())

    async def self_cancelling_notify(ticket, message):
        inner = asyncio.ensure_future(asyncio.sleep(999))
        inner.cancel()
        _ = await inner  # raises CancelledError -- self-inflicted, not from outside

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=self_cancelling_notify)

    async def approve_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    # If the bug were still present, ask() would abort right after the
    # notify call instead of reaching its polling loop, and this gather
    # would raise CancelledError instead of returning normally.
    result, _ = await asyncio.wait_for(asyncio.gather(channel.ask("please approve"), approve_soon()), timeout=1.0)

    assert result is True


async def test_send_swallows_a_notify_that_raises_cancelled_before_returning_an_awaitable() -> None:
    """A `notify` callable can raise CancelledError SYNCHRONOUSLY, when
    merely called, before it ever returns an awaitable for
    asyncio.ensure_future to wrap -- no notify_task exists yet at that
    point for the shield()-based distinction to inspect. This must still
    be treated as an ordinary notify failure (logged and swallowed), not
    mistaken for ask() itself having been cancelled, per this class' own
    documented contract that notify failures never fail the approval
    itself. Found by Codex review before this ever shipped: the earlier
    fix only covered a self-cancelling notify TASK, missing that the
    factory call constructing it can raise the exact same way before one
    ever exists."""
    queue = ApprovalQueue(Store())

    def notify_that_raises_cancelled_on_call(ticket, message):
        raise asyncio.CancelledError("self-inflicted, before returning anything")

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=notify_that_raises_cancelled_on_call)

    async def approve_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    # If the bug were still present, ask() would abort right after the
    # notify call instead of reaching its polling loop, and this gather
    # would raise CancelledError instead of returning normally.
    result, _ = await asyncio.wait_for(asyncio.gather(channel.ask("please approve"), approve_soon()), timeout=1.0)

    assert result is True


async def test_send_preserves_an_external_cancellation_racing_notifier_self_cancellation() -> None:
    """Checking only `notify_task.cancelled()` isn't enough to tell "the
    notifier cancelled itself" apart from "ask() ALSO has a pending
    external cancellation at the very same moment" -- the latter must
    still terminate ask() and retire its ticket, not be swallowed just
    because notify_task also happened to end up cancelled in the same
    event-loop turn. `Task.cancelling()` (3.11+, this project's floor) is
    what distinguishes the two regardless of timing. Found by Codex review
    before this ever shipped."""
    queue = ApprovalQueue(Store())
    real_create_ticket = queue.create_ticket
    created: list[ApprovalTicket] = []

    def _tracking_create_ticket(*args, **kwargs):
        ticket = real_create_ticket(*args, **kwargs)
        created.append(ticket)
        return ticket

    queue.create_ticket = _tracking_create_ticket  # type: ignore[method-assign]

    ask_task_holder: list[asyncio.Task] = []

    async def racing_notify(ticket, message):
        # Both cancellations are requested synchronously, back to back,
        # so the event loop delivers them in the same round: ask_task's
        # own pending cancellation, and notify_task's self-inflicted one,
        # racing exactly as described above.
        ask_task_holder[0].cancel()
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        await asyncio.sleep(999)  # unreachable -- cancelled at this suspension point

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=racing_notify)

    ask_task = asyncio.create_task(channel.ask("please approve"))
    ask_task_holder.append(ask_task)

    with contextlib.suppress(asyncio.CancelledError):
        _ = await ask_task

    # The external cancellation must win: ask() itself terminates instead
    # of continuing to poll (which the bug this test guards against would
    # cause, by swallowing the CancelledError here as if it were purely
    # notify_task's own doing).
    assert ask_task.cancelled()

    await _wait_until(lambda: len(created) == 1)
    await _wait_until(lambda: queue.get_ticket(created[0].approval_id).status == "rejected")
    resolved = queue.get_ticket(created[0].approval_id)
    assert resolved.status == "rejected"


async def test_send_swallows_notifier_self_cancellation_despite_a_stale_cancelling_count() -> None:
    """Task.cancelling() is cumulative and never auto-resets on its own --
    a task that absorbed some EARLIER, unrelated cancellation (caught it
    and kept going, without ever calling uncancel(), exactly like a
    caller's own cleanup path might) keeps a nonzero count for the rest of
    its life. Checking that raw count instead of comparing it across just
    THIS specific await would treat every LATER notifier self-cancellation
    on that same task as if it were a brand new external cancellation of
    ask(), rejecting a perfectly fine ticket. Found by Codex review before
    this ever shipped."""
    queue = ApprovalQueue(Store())

    async def self_cancelling_notify(ticket, message):
        inner = asyncio.ensure_future(asyncio.sleep(999))
        inner.cancel()
        _ = await inner  # raises CancelledError -- self-inflicted, not from outside

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=self_cancelling_notify)

    async def run() -> bool:
        current = asyncio.current_task()
        assert current is not None
        # Absorb an unrelated, already-resolved cancellation first,
        # WITHOUT calling uncancel() -- current.cancelling() stays
        # elevated for the rest of this task's life from here on.
        current.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert current.cancelling() > 0

        async def approve_soon() -> None:
            while not queue.list_pending_tickets():
                await asyncio.sleep(0.005)
            [ticket] = queue.list_pending_tickets()
            queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

        # approve_soon() runs as its OWN task, but ask() itself is awaited
        # DIRECTLY (not gather()'d into a task of its own) so it runs on
        # THIS coroutine's own task -- gather() would silently give it a
        # fresh task with a clean cancelling() count, defeating the whole
        # point of this test.
        approve_task = asyncio.create_task(approve_soon())
        result = await channel.ask("please approve")
        _ = await approve_task
        return result

    # A bare `current.cancelling() == 0` check (rather than comparing the
    # count across this specific await) would see the stale nonzero count
    # above and wrongly treat the notifier's self-cancellation as ask()
    # being cancelled, raising CancelledError out of run() instead of
    # returning True.
    result = await asyncio.wait_for(asyncio.create_task(run()), timeout=1.0)
    assert result is True


async def test_send_snapshots_cancelling_count_before_invoking_notify() -> None:
    """A notify callable that cancels the ask()-task as a synchronous side
    effect of merely being CALLED -- before it even returns its own
    awaitable -- must still terminate ask() and retire its ticket.
    Snapshotting cancelling() any later than the very start of _send()
    (in particular, after `self._notify` has already run) would already
    include that fresh increment in the "before" snapshot, so the
    subsequent before/after delta would show no NEW increase and wrongly
    swallow this real external cancellation, leaving ask() to poll
    forever instead. Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    ask_task_holder: list[asyncio.Task] = []

    def cancel_on_call(ticket, message):
        # Cancels ask() -- and hands back an ALREADY-CANCELLED future --
        # entirely as a side effect of being CALLED, before this factory
        # has even returned an awaitable of its own.
        ask_task_holder[0].cancel()
        fut = asyncio.get_event_loop().create_future()
        fut.cancel()
        return fut

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=cancel_on_call)

    ask_task = asyncio.create_task(channel.ask("please approve"))
    ask_task_holder.append(ask_task)

    # Deliberately NOT wait_for()/cancel()'d by this test itself -- either
    # would force a cancellation of its own and mask the very thing being
    # tested. If the bug is present, ask() swallows the cancellation above
    # and keeps polling (the ticket's own TTL is hours), so this is just a
    # bounded window to let the FIXED behavior settle.
    await asyncio.sleep(0.2)

    try:
        assert ask_task.cancelled()
    finally:
        if not ask_task.done():
            ask_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                _ = await ask_task


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
    # create_ticket() rejects a nonpositive ttl outright (it would write
    # an already-expired ticket that's durably stuck as "pending"
    # forever), so a tiny POSITIVE ttl that elapses for real is how this
    # test gets a genuinely expired ticket instead.
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="p", ttl=timedelta(microseconds=1))
    time.sleep(0.01)
    assert queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram") is False


def test_list_pending_tickets_excludes_resolved_and_expired() -> None:
    queue = ApprovalQueue(Store())
    pending = queue.create_ticket(task_id="t1", prompt="p1")
    resolved = queue.create_ticket(task_id="t1", prompt="p2")
    queue.approve_ticket(resolved.approval_id, actor="marco", channel="telegram")
    # A tiny POSITIVE ttl that elapses for real -- create_ticket() rejects
    # a nonpositive one outright, see test_approve_expired_ticket_fails.
    queue.create_ticket(task_id="t1", prompt="p3", ttl=timedelta(microseconds=1))
    time.sleep(0.01)

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
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
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
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
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
        # Polls for the ticket to exist rather than a fixed sleep: ask() now
        # checks claim_earlier_approval (one more offloaded Store round
        # trip) before create_ticket even runs, and that extra hop can cost
        # an occasional double-digit-ms jump on Windows' own scheduler
        # granularity -- a fixed short sleep raced that and flaked.
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
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
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve"), approve_after_a_while())
    assert result is True
    assert len(notified) == 1


async def test_store_approval_channel_returns_true_when_approved() -> None:
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    async def approve_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve"), approve_soon())
    assert result is True


async def test_store_approval_channel_returns_false_when_rejected() -> None:
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    async def reject_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
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


def test_poll_seconds_rejects_nan() -> None:
    """float("nan") compares False against BOTH `<= 0` and `> 0` -- a
    plain `poll_seconds <= 0` guard lets it straight through, and
    asyncio.sleep(nan) may never wake at all, silently missing both the
    ticket's own resolution and its TTL expiry. Found by Codex review
    before this ever shipped."""
    queue = ApprovalQueue(Store())
    with pytest.raises(ValueError, match="poll_seconds"):
        StoreApprovalChannel(queue, task_id="t1", poll_seconds=float("nan"))


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


def test_notify_timeout_must_be_positive() -> None:
    """A nonpositive notify_timeout hands _send() an immediate deadline --
    the notifier gets cancelled at its very first suspension point
    (typically a network call), before it can ever deliver the one
    message that tells a human this ticket exists, while ask() keeps
    polling unnoticed for up to the ticket's full TTL. Found by Codex
    review before this ever shipped."""
    queue = ApprovalQueue(Store())
    with pytest.raises(ValueError, match="notify_timeout"):
        StoreApprovalChannel(queue, task_id="t1", notify_timeout=0)
    with pytest.raises(ValueError, match="notify_timeout"):
        StoreApprovalChannel(queue, task_id="t1", notify_timeout=-1)


def test_notify_timeout_rejects_nan() -> None:
    """float("nan") compares False against BOTH `<= 0` and `> 0` -- a
    plain `notify_timeout <= 0` guard lets it straight through. Found by
    Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    with pytest.raises(ValueError, match="notify_timeout"):
        StoreApprovalChannel(queue, task_id="t1", notify_timeout=float("nan"))


def test_create_ticket_rejects_a_nonpositive_ttl() -> None:
    """A nonpositive ttl writes a ticket whose expires_at is already at or
    before created_at -- immediately invisible to list_pending_tickets()
    and unapprovable/unrejectable, yet still durably stored as "pending"
    forever: a configuration mistake would silently deny every request
    through this queue while accumulating misleading pending-looking
    records nothing can ever act on. Found by Codex review before this
    ever shipped."""
    queue = ApprovalQueue(Store())
    with pytest.raises(ValueError, match="ttl"):
        queue.create_ticket(task_id="t1", prompt="p", ttl=timedelta(0))
    with pytest.raises(ValueError, match="ttl"):
        queue.create_ticket(task_id="t1", prompt="p", ttl=timedelta(seconds=-1))


def test_store_approval_channel_rejects_a_nonpositive_ttl_at_construction() -> None:
    """Same check as ApprovalQueue.create_ticket()'s own, but surfaced at
    StoreApprovalChannel construction time -- consistent with every other
    timing parameter this class already validates up front, rather than
    waiting for ask() to eventually call create_ticket() and hit it
    there. Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    with pytest.raises(ValueError, match="ttl"):
        StoreApprovalChannel(queue, task_id="t1", ttl=timedelta(0))
    with pytest.raises(ValueError, match="ttl"):
        StoreApprovalChannel(queue, task_id="t1", ttl=timedelta(seconds=-1))


async def test_ask_does_not_block_the_event_loop() -> None:
    """ApprovalQueue's methods are synchronous Store I/O -- calling them
    directly from this async method would block the WHOLE event loop for
    as long as the call takes (Store's busy_timeout is 5s under write
    contention -- a shared Store with many agents filing tickets/
    heartbeats is exactly what produces that), freezing every other
    coroutine sharing it, not just this ticket's own progress.

    Checks whether a tick lands strictly BETWEEN create_ticket's own
    recorded start/end timestamps, not just whether ticks eventually
    happen at all, and not a fixed elapsed-time threshold either -- both
    of those looked right but were found NOT to actually distinguish
    blocking from non-blocking when checked empirically (repeatedly
    reverting the fix under test and confirming each style still passed):
    a gather()-then-check-the-total assertion passes either way (blocking
    only delays when the ticks happen, not whether all of them eventually
    do); a fixed `await asyncio.sleep(0.1)` then `assert ticks > 0` ALSO
    passes either way, because every timer that would have fired during a
    genuinely-blocked window (the ticker's own included) simply becomes
    overdue and fires in one catch-up batch the instant the block ends.
    Anchoring the window to create_ticket's OWN measured start/end is
    what actually ties a tick to being concurrent with it specifically.
    Found by Codex review before this ever shipped (three rounds: the
    production fix, this test's own first version not actually
    distinguishing blocking from non-blocking, and this second version
    -- an elapsed-time-threshold check -- turning out to be flaky the
    same way for a different reason, confirmed by running it repeatedly
    against the reverted fix)."""
    queue = ApprovalQueue(Store())
    real_create_ticket = queue.create_ticket
    loop = asyncio.get_event_loop()
    create_start: list[float] = []
    create_end: list[float] = []

    def _slow_create_ticket(*args, **kwargs):
        create_start.append(loop.time())
        time.sleep(0.2)
        create_end.append(loop.time())
        return real_create_ticket(*args, **kwargs)

    queue.create_ticket = _slow_create_ticket  # type: ignore[method-assign]
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, ttl=timedelta(seconds=0.05))
    tick_times: list[float] = []

    async def _ticker() -> None:
        while True:
            await asyncio.sleep(0.02)
            tick_times.append(loop.time())

    ask_task = asyncio.create_task(channel.ask("please approve"))
    ticker_task = asyncio.create_task(_ticker())

    with contextlib.suppress(Exception):
        _ = await ask_task
    ticker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        _ = await ticker_task

    assert create_start and create_end
    window_start, window_end = create_start[0], create_end[0]
    # If create_ticket ran directly on this event loop, NOTHING else
    # could execute for the whole [window_start, window_end) it was
    # running -- no tick could possibly land inside it.
    assert any(window_start < t < window_end for t in tick_times), (
        f"no tick landed inside create_ticket's own [{window_start}, {window_end}) window; "
        f"tick timestamps: {tick_times}"
    )


def test_non_async_def_notify_warns_at_construction() -> None:
    """`notify` must be an `async def` -- calling one only constructs a
    coroutine object, running none of its body yet, which is what keeps
    calling it directly in _send() always cheap regardless of what the
    coroutine goes on to do. A plain synchronous factory that performs
    real, blocking work before ever returning one runs that work
    directly on the event loop's own thread, freezing every other
    coroutine sharing it -- with neither notify_timeout nor cancellation
    able to help.

    Offloading such a call to a worker thread (to defend against exactly
    that) was tried and reverted: it broke an equally valid pattern, a
    synchronous factory that legitimately needs the RUNNING loop to
    build its result (`asyncio.get_event_loop().create_future()`, e.g.),
    with `RuntimeError: no running event loop` -- confirmed with a
    standalone repro. There is no way to tell the two apart via
    introspection, so a loud, one-time warning at construction (where a
    human will actually see it) is the whole defense; this test checks
    that the warning fires for a non-coroutine-function notify, does NOT
    fire for a proper async def OR a callable OBJECT whose own __call__
    is async def (inspect.iscoroutinefunction(obj) alone reports False
    for the latter even though calling it has exactly the safe,
    non-blocking-construction property this check exists to confirm --
    found by Codex review as a false positive in this warning's first
    version). Found by Codex review before this ever shipped (three
    times: once for the missing protection, again for the first fix
    attempt's own regression, and again for this warning's own false
    positive on async-__call__ objects)."""
    queue = ApprovalQueue(Store())

    def synchronous_notify(ticket, message):
        async def _noop() -> None:
            return None

        return _noop()

    caught: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            caught.append(record)

    logger = logging.getLogger("lazybridge.ext.approval.queue")
    handler = _Handler()
    logger.addHandler(handler)
    try:
        StoreApprovalChannel(queue, task_id="t1", notify=synchronous_notify)
        assert any("not an `async def`" in r.getMessage() for r in caught)

        caught.clear()

        async def proper_async_notify(ticket, message) -> None:
            return None

        StoreApprovalChannel(queue, task_id="t1", notify=proper_async_notify)
        assert not any("not an `async def`" in r.getMessage() for r in caught)

        caught.clear()

        class AsyncCallableNotifier:
            async def __call__(self, ticket, message) -> None:
                return None

        StoreApprovalChannel(queue, task_id="t1", notify=AsyncCallableNotifier())
        assert not any("not an `async def`" in r.getMessage() for r in caught)
    finally:
        logger.removeHandler(handler)


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
    full TTL expiry hours later.

    The retiring reject_ticket call is made SYNCHRONOUSLY here, not on a
    detached task -- an earlier version detached it (to survive a
    repeated cancellation of ask() itself), but that let ask()'s own
    CancelledError reach its caller BEFORE the detached task's own
    reject_ticket CAS had actually run, so another surface could approve
    the still-"pending" ticket in that exact window: the CAS would
    succeed, silently, for a decision nobody was listening for anymore
    (confirmed as a real gap by Codex review). A synchronous call has no
    `await` for a repeated cancellation to land in either, so it keeps
    the same immunity to that WITHOUT the race -- no `_wait_until`
    polling needed for the RESULT anymore: retirement is guaranteed
    complete the instant `await task` returns.

    Waits for `get_ticket` to have been CALLED at least once (rather than
    for the ticket to merely be VISIBLE via list_pending_tickets(), which
    this test used to wait for, and before that a fixed `await
    asyncio.sleep(0.03)`) before cancelling -- both of those are racy in
    the same way: create_ticket runs on a worker thread and writes
    straight to the Store, so the ticket can become visible to a direct
    read like list_pending_tickets() SLIGHTLY BEFORE ask()'s own
    `await asyncio.shield(create)` has actually resumed and moved on
    (that requires a separate round trip through the event loop's
    callback queue). Cancelling inside that narrow window lands on the
    OTHER retirement path instead -- cancellation DURING creation, a
    still-detached task, since the ticket doesn't exist FROM ASK()'S OWN
    POINT OF VIEW yet to reject synchronously -- silently testing the
    wrong thing and flaking intermittently (confirmed empirically: this
    exact race reproduced consistently once isolated). `get_ticket` can
    only ever be called from inside the polling loop, past that whole
    handler, so waiting for it is a reliable proxy for "ask() has
    committed to the post-creation code path" that a Store-level read
    is not. Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    real_get_ticket = queue.get_ticket
    get_ticket_calls = 0

    def _counting_get_ticket(*args, **kwargs):
        nonlocal get_ticket_calls
        get_ticket_calls += 1
        return real_get_ticket(*args, **kwargs)

    queue.get_ticket = _counting_get_ticket  # type: ignore[method-assign]
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    task = asyncio.create_task(channel.ask("please approve"))
    await _wait_until(lambda: get_ticket_calls > 0)
    queue.get_ticket = real_get_ticket  # type: ignore[method-assign]
    [ticket] = queue.list_pending_tickets()

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        _ = await task

    resolved = queue.get_ticket(ticket.approval_id)
    assert resolved.status == "rejected"
    assert queue.list_pending_tickets() == []


async def test_ask_retires_the_ticket_when_cancelled_during_creation() -> None:
    """A cancellation landing WHILE create_ticket's offloaded call is
    still in flight must not orphan the ticket that lands moments
    later -- the underlying thread-pool work can't actually be stopped
    once started, so this only works because ask() shields that specific
    call and hands its real outcome to an independent watcher task that
    retires it.

    `await task` now waits for that watcher (shielded, so a repeated
    cancellation of `task` can't interrupt the wait) rather than
    returning the instant the first cancellation lands -- an earlier
    version detached AND returned immediately, which let another surface
    approve the ticket the moment create_ticket's write landed, before
    the watcher got a chance to reject it (confirmed as a real gap by
    Codex review). No `_wait_until` polling needed here anymore:
    retirement is guaranteed complete by the time `await task` returns.
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
        _ = await task

    assert len(created) == 1
    resolved = queue.get_ticket(created[0].approval_id)
    assert resolved.status == "rejected"


async def test_ask_retires_the_ticket_when_cancelled_twice_during_creation() -> None:
    """A SECOND cancellation (and, in principle, any further one after
    that) landing while ask() is still WAITING ON its own retiring
    watcher -- not just while the watcher itself is running -- must not
    let that wait give up early: asyncio.TaskGroup cancelling every
    remaining sibling when one fails is a real way a repeated
    cancellation happens, not just theoretical.

    Two earlier fix attempts got this wrong in opposite directions: the
    first nested a second `asyncio.shield()` inside ask()'s own
    cancellation handler and awaited it ONCE, proven insufficient by a
    standalone repro (shield() protects the awaited future from
    cancellation, not the coroutine doing the awaiting from being
    cancelled again). The second detached the watcher correctly but then
    re-raised immediately, before the watcher was necessarily done,
    opening a window for another surface to approve the ticket before
    retirement won the CAS (confirmed as a real gap by Codex review).
    ask()'s own `except` block now loops, re-shielding its wait on the
    watcher after each further cancellation instead of giving up after
    one -- no `_wait_until` polling needed here anymore either:
    retirement is guaranteed complete by the time `await task` returns,
    however many cancellations landed along the way. Found by Codex
    review before this ever shipped."""
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
    task.cancel()  # second cancellation -- ask() is now waiting on its own retiring watcher
    with contextlib.suppress(asyncio.CancelledError):
        _ = await task

    assert len(created) == 1
    resolved = queue.get_ticket(created[0].approval_id)
    assert resolved.status == "rejected"


async def test_ask_retires_the_ticket_when_cancelled_twice_after_creation() -> None:
    """The post-creation retirement call is made SYNCHRONOUSLY (see
    test_ask_retires_the_ticket_when_cancelled's own docstring for why:
    an earlier, detached-task version of this cleanup let ask()'s own
    CancelledError reach its caller before the retirement CAS had
    actually run, letting another surface approve a ticket nobody was
    listening for anymore). A synchronous call has no `await` inside it
    for a second `.cancel()` to land in at all, so a repeated
    cancellation here is a non-event by construction -- this test exists
    to confirm that stays true, not to exercise a race that (unlike the
    creation-path cleanup, which genuinely still has to wait
    asynchronously for creation to finish) no longer has anywhere left to
    land. Found by Codex review before this ever shipped (this test
    itself changed meaning once: an earlier version relied on a detached
    task for this same cleanup and specifically exercised a second
    cancellation racing that task's own await)."""
    queue = ApprovalQueue(Store())
    real_get_ticket = queue.get_ticket
    get_ticket_calls = 0

    def _counting_get_ticket(*args, **kwargs):
        nonlocal get_ticket_calls
        get_ticket_calls += 1
        return real_get_ticket(*args, **kwargs)

    queue.get_ticket = _counting_get_ticket  # type: ignore[method-assign]
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    task = asyncio.create_task(channel.ask("please approve"))
    await _wait_until(lambda: get_ticket_calls > 0)  # confirms we're past ticket creation
    queue.get_ticket = real_get_ticket  # type: ignore[method-assign]
    [ticket] = queue.list_pending_tickets()

    task.cancel()  # first cancellation
    task.cancel()  # second -- a no-op, nothing async left in the cleanup path to interrupt
    with contextlib.suppress(asyncio.CancelledError):
        _ = await task

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
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve"), approve_soon())
    assert result is True


async def test_ask_works_with_an_encrypted_sqlite_memory_store() -> None:
    """The exact scenario Codex flagged: StoreApprovalChannel offloading
    ticket creation to a worker thread because an EncryptedStoreAdapter
    wrapping Store(db=":memory:") looked "thread-safe" from the outside
    (it exposes none of Store's own _db attribute itself), landing on a
    different thread's own unrelated in-memory database and raising
    sqlite3.OperationalError: no such table: store. Found by Codex review
    before this ever shipped."""
    # pytest.importorskip (not a plain try/except ImportError + pytest.skip)
    # so `Fernet` is unconditionally bound afterward -- static analysis
    # can't know pytest.skip() never returns, and flags the try/except
    # shape as leaving `Fernet` possibly-uninitialized on the line below.
    # Found by CodeQL before this ever shipped.
    Fernet = pytest.importorskip("cryptography.fernet").Fernet
    from lazybridge.store.encryption import EncryptedStoreAdapter

    queue = ApprovalQueue(EncryptedStoreAdapter(Store(db=":memory:"), key=Fernet.generate_key()))
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    async def approve_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
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
        # Polls rather than a fixed sleep -- ask() now checks
        # claim_earlier_approval (one more offloaded Store round trip)
        # before create_ticket even runs, see
        # test_store_approval_channel_stops_renotifying_once_resolved's
        # own identical comment.
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
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
    # Fixed: close to the true 0.2s ttl, PLUS one more ~0.08s round trip
    # through the same slow_notify for the ticket's own EXPIRY notice (a
    # later addition -- the ticket now gets told, once, that it expired).
    # 0.4s leaves comfortable margin on both sides without making the test
    # flaky.
    assert elapsed < 0.4


async def test_reminder_spacing_accounts_for_notify_delivery_time() -> None:
    """last_notified must be stamped from a FRESH timestamp taken after
    the reminder's own notify call completes, not from the timestamp
    captured before it -- otherwise a notifier that takes real time to
    deliver (bounded by notify_timeout, which can be a meaningful
    fraction of a short renotify_interval) has its own delivery time
    eaten into the NEXT interval too, collapsing consecutive reminders
    closer together than configured.

    Checks the gap from one delivery's FINISH to the next one's START,
    not from START to START -- the bug doesn't actually change start-to-
    start spacing at all (each reminder still fires `renotify_interval`
    after the PREVIOUS one's own pre-send timestamp, so consecutive
    starts stay `renotify_interval` apart regardless of the bug); what it
    collapses is the gap a human actually experiences between one
    message finishing and the next one beginning, exactly as Codex's own
    example put it: "subsequent notifications begin only ~10ms after the
    preceding delivery completes" for a 50ms interval and a 40ms
    notifier. A start-to-start check was tried first and found not to
    distinguish the two at all, confirmed empirically by reverting the
    fix and observing it still pass. Found by Codex review before this
    ever shipped."""
    queue = ApprovalQueue(Store())
    loop = asyncio.get_event_loop()
    notify_windows: list[tuple[float, float]] = []

    async def slow_notify(ticket, message):
        start = loop.time()
        await asyncio.sleep(0.04)
        notify_windows.append((start, loop.time()))

    channel = StoreApprovalChannel(
        queue,
        task_id="t1",
        poll_seconds=0.01,
        ttl=timedelta(seconds=1),
        notify=slow_notify,
        renotify_interval=timedelta(seconds=0.05),
    )

    async def approve_after_a_few_reminders() -> None:
        await _wait_until(lambda: len(notify_windows) >= 4, timeout=2.0)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.wait_for(
        asyncio.gather(channel.ask("please approve"), approve_after_a_few_reminders()), timeout=3.0
    )
    assert result is True

    # Each delivery's own finish time to the NEXT delivery's start time
    # must be close to the configured renotify_interval -- not collapsed
    # by the previous delivery's own 0.04s eating into the next interval.
    gaps = [next_start - this_finish for (_, this_finish), (next_start, _) in itertools.pairwise(notify_windows)]
    assert all(gap >= 0.045 for gap in gaps), gaps


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
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        pending = queue.list_pending_tickets()
        assert len(pending) == 1
        queue.approve_ticket(pending[0].approval_id, actor="marco", channel="telegram")

    request = ApprovalRequest(
        provider="claude-code", kind="tool", name="Bash", arguments={"command": "git commit -m x"}, cwd="C:/repo"
    )
    decision, _ = await asyncio.gather(gate(request), resolve_soon())
    assert decision.action == "allow"


# --- claim_earlier_approval ---------------------------------------------


def test_claim_earlier_approval_spends_a_matching_unconsumed_approval() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="run it")
    queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    assert queue.claim_earlier_approval(task_id="t1", prompt="run it") is True
    assert queue.get_ticket(ticket.approval_id).consumed_at is not None


def test_claim_earlier_approval_is_single_use() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="run it")
    queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    assert queue.claim_earlier_approval(task_id="t1", prompt="run it") is True
    assert queue.claim_earlier_approval(task_id="t1", prompt="run it") is False


def test_claim_earlier_approval_requires_exact_prompt_match() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="run it")
    queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    assert queue.claim_earlier_approval(task_id="t1", prompt="run it, please") is False


def test_claim_earlier_approval_requires_matching_task_id() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="run it")
    queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    assert queue.claim_earlier_approval(task_id="other-task", prompt="run it") is False


def test_claim_earlier_approval_ignores_pending_and_rejected_tickets() -> None:
    queue = ApprovalQueue(Store())
    pending = queue.create_ticket(task_id="t1", prompt="run it")
    rejected = queue.create_ticket(task_id="t1", prompt="reject it")
    queue.reject_ticket(rejected.approval_id, actor="marco", channel="telegram", reason="no")

    assert queue.claim_earlier_approval(task_id="t1", prompt="run it") is False
    assert queue.claim_earlier_approval(task_id="t1", prompt="reject it") is False
    assert queue.get_ticket(pending.approval_id).status == "pending"


def test_claim_earlier_approval_ignores_an_expired_approval() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="run it", ttl=timedelta(microseconds=1))
    # approve_ticket itself refuses an expired ticket -- force the row
    # into "approved" directly to isolate claim_earlier_approval's OWN
    # expiry check from approve_ticket's.
    raw = queue._store.read(queue._key(ticket.approval_id))
    queue._store.write(queue._key(ticket.approval_id), {**raw, "status": "approved"})
    # A tiny POSITIVE ttl that elapses for real -- see
    # test_approve_expired_ticket_fails' own identical comment.
    time.sleep(0.01)

    assert queue.claim_earlier_approval(task_id="t1", prompt="run it") is False


# --- expire_ticket -------------------------------------------------------


def test_expire_ticket_transitions_a_pending_ticket() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="p")
    assert queue.expire_ticket(ticket.approval_id) is True
    assert queue.get_ticket(ticket.approval_id).status == "expired"


def test_expire_ticket_is_cas_second_call_fails() -> None:
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="p")
    assert queue.expire_ticket(ticket.approval_id) is True
    assert queue.expire_ticket(ticket.approval_id) is False


def test_expire_ticket_loses_to_a_concurrent_approval() -> None:
    """A human answering between the read and the write wins -- expiring
    an already-approved ticket would silently turn a real "yes" into a
    "did not run"."""
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="p")
    queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")
    assert queue.expire_ticket(ticket.approval_id) is False
    assert queue.get_ticket(ticket.approval_id).status == "approved"


def test_expire_ticket_on_unknown_ticket_returns_false() -> None:
    queue = ApprovalQueue(Store())
    assert queue.expire_ticket("nope") is False


def test_list_pending_tickets_excludes_an_expired_ticket() -> None:
    queue = ApprovalQueue(Store())
    pending = queue.create_ticket(task_id="t1", prompt="p1")
    expiring = queue.create_ticket(task_id="t1", prompt="p2")
    queue.expire_ticket(expiring.approval_id)

    assert [t.approval_id for t in queue.list_pending_tickets()] == [pending.approval_id]


# --- operator_only --------------------------------------------------------


def test_create_ticket_operator_only_defaults_to_false_and_is_stored_when_set() -> None:
    queue = ApprovalQueue(Store())
    default_ticket = queue.create_ticket(task_id="t1", prompt="p")
    flagged_ticket = queue.create_ticket(task_id="t1", prompt="p2", operator_only=True)

    assert default_ticket.operator_only is False
    assert flagged_ticket.operator_only is True
    assert queue.get_ticket(flagged_ticket.approval_id).operator_only is True


async def test_store_approval_channel_stamps_operator_only_on_tickets_it_creates() -> None:
    queue = ApprovalQueue(Store())
    captured: list[ApprovalTicket] = []

    async def approve_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        captured.append(ticket)
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, operator_only=True)
    await asyncio.gather(channel.ask("please approve"), approve_soon())

    assert captured[0].operator_only is True


# --- record_denial ---------------------------------------------------------


def test_record_denial_ignores_non_deny_actions() -> None:
    queue = ApprovalQueue(Store())
    queue.record_denial(SimpleNamespace(action="allow", tool_name="Bash", message="fine"))
    assert queue._store.items(prefix=DEFAULT_DENIAL_PREFIX) == []


def test_record_denial_writes_a_denial_for_deny_and_denied_actions() -> None:
    queue = ApprovalQueue(Store())
    queue.record_denial(
        SimpleNamespace(
            action="deny",
            tool_name="Bash",
            kind="tool",
            tier="ask",
            message="'Bash' denied by the human approver",
            responder="store-approval-queue",
            cwd="C:/repo",
            arguments_preview="git push",
        )
    )
    queue.record_denial(SimpleNamespace(action="denied", tool_name="Write", message="no rule matched"))

    records = [raw for _key, raw in queue._store.items(prefix=DEFAULT_DENIAL_PREFIX)]
    assert len(records) == 2
    by_tool = {r["tool_name"]: r for r in records}
    assert by_tool["Bash"]["message"] == "'Bash' denied by the human approver"
    assert by_tool["Bash"]["arguments_preview"] == "git push"
    assert by_tool["Write"]["message"] == "no rule matched"


def test_record_denial_prefix_is_configurable() -> None:
    queue = ApprovalQueue(Store(), denial_prefix="custom-denial:")
    queue.record_denial(SimpleNamespace(action="deny", tool_name="Bash", message="x"))
    assert len(list(queue._store.items(prefix="custom-denial:"))) == 1
    assert queue._store.items(prefix=DEFAULT_DENIAL_PREFIX) == []


def test_record_denial_never_raises() -> None:
    """Best-effort: a failure to record WHY something was refused must not
    become a second fault on top of the refusal itself."""

    class _BrokenStore:
        def write(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("store is down")

        def read(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def items(self, *_args: Any, **_kwargs: Any) -> list:
            return []

    queue = ApprovalQueue(_BrokenStore())  # type: ignore[arg-type]
    queue.record_denial(SimpleNamespace(action="deny", tool_name="Bash", message="x"))  # must not raise


def test_tiered_gate_on_record_wires_directly_into_record_denial() -> None:
    """record_denial is generic enough to be TieredGate's own ``on_record``
    callback directly -- no adapter needed."""
    queue = ApprovalQueue(Store())
    gate = TieredGate(channel=TerminalChannel(), rules=(), on_record=queue.record_denial)

    from lazybridge.engines.coding import ApprovalRequest

    request = ApprovalRequest(provider="claude-code", kind="tool", name="Bash", arguments={}, cwd="C:/repo")
    decision = asyncio.run(gate(request))

    assert decision.action == "deny"
    records = [raw for _key, raw in queue._store.items(prefix=DEFAULT_DENIAL_PREFIX)]
    assert len(records) == 1
    assert records[0]["tool_name"] == "Bash"
    assert "no rule" in records[0]["message"]


# --- record_notify_failure -------------------------------------------------


def test_record_notify_failure_writes_a_durable_record() -> None:
    queue = ApprovalQueue(Store())
    queue.record_notify_failure(source="StoreApprovalChannel:approval", text="hello", error="boom")

    records = [raw for _key, raw in queue._store.items(prefix=DEFAULT_NOTIFY_FAILURE_PREFIX)]
    assert len(records) == 1
    assert records[0]["source"] == "StoreApprovalChannel:approval"
    assert records[0]["text"] == "hello"
    assert records[0]["error"] == "boom"
    assert records[0]["failed_at"]


def test_record_notify_failure_prefix_is_configurable() -> None:
    queue = ApprovalQueue(Store(), notify_failure_prefix="custom-nf:")
    queue.record_notify_failure(source="x", text="y", error="z")
    assert len(list(queue._store.items(prefix="custom-nf:"))) == 1


async def test_store_approval_channel_records_a_durable_notify_failure() -> None:
    queue = ApprovalQueue(Store())

    async def broken_notify(ticket, message):
        raise RuntimeError("notification channel unreachable")

    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, notify=broken_notify)

    async def approve_soon():
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    await asyncio.gather(channel.ask("please approve"), approve_soon())

    records = [raw for _key, raw in queue._store.items(prefix=DEFAULT_NOTIFY_FAILURE_PREFIX)]
    assert len(records) == 1
    assert "notification channel unreachable" in records[0]["error"]


# --- rejection/expiry are recorded as denials through ask() ---------------


async def test_ask_records_a_rejection_with_the_real_actor() -> None:
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    async def reject_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.reject_ticket(ticket.approval_id, actor="marco", channel="telegram", reason="too risky")

    result, _ = await asyncio.gather(channel.ask("please approve"), reject_soon())
    assert result is False

    records = [raw for _key, raw in queue._store.items(prefix=DEFAULT_DENIAL_PREFIX)]
    assert len(records) == 1
    assert records[0]["responder"] == "marco"
    assert "rejected by marco" in records[0]["message"]
    assert "too risky" in records[0]["message"]


async def test_ask_expires_and_records_a_ticket_nobody_answered() -> None:
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, ttl=timedelta(seconds=0.03))

    result = await channel.ask("please approve")
    assert result is False

    [(_key, raw)] = queue._store.items(prefix="approval:")
    assert raw["status"] == "expired"

    records = [raw for _key, raw in queue._store.items(prefix=DEFAULT_DENIAL_PREFIX)]
    assert len(records) == 1
    assert records[0]["responder"] == "nobody"
    assert "expired" in records[0]["message"]


# --- wait_budget / ttl split -----------------------------------------------


async def test_wait_budget_gives_up_without_expiring_the_ticket() -> None:
    """A caller that stops WAITING does not kill the REQUEST -- the ticket
    stays pending, listed, and answerable well past wait_budget, as long
    as it's still within ttl."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(
        queue,
        task_id="t1",
        poll_seconds=0.01,
        ttl=timedelta(seconds=2),
        wait_budget=timedelta(seconds=0.03),
    )

    result = await asyncio.wait_for(channel.ask("please approve"), timeout=1.0)

    assert result is False
    [ticket] = queue.list_pending_tickets()
    assert ticket.status == "pending"


async def test_wait_budget_default_matches_the_old_full_ttl_wait() -> None:
    """Leaving wait_budget unset reproduces this module's own pre-split
    behaviour exactly: a caller waits the full ttl, and a ticket nobody
    ever answers ends up "expired", not merely abandoned mid-wait."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, ttl=timedelta(seconds=0.03))

    result = await channel.ask("please approve")

    assert result is False
    [(_key, raw)] = queue._store.items(prefix="approval:")
    assert raw["status"] == "expired"


async def test_a_later_ask_claims_an_approval_given_after_an_earlier_waiter_gave_up() -> None:
    """The whole point of splitting wait_budget from ttl: an answer given
    after one caller stopped waiting is not wasted -- a LATER call with
    the identical prompt picks it up via claim_earlier_approval instead of
    asking the same question into a void a second time."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(
        queue,
        task_id="t1",
        poll_seconds=0.01,
        ttl=timedelta(seconds=2),
        wait_budget=timedelta(seconds=0.03),
    )

    first_result = await asyncio.wait_for(channel.ask("please approve"), timeout=1.0)
    assert first_result is False

    [ticket] = queue.list_pending_tickets()
    queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    second_result = await channel.ask("please approve")
    assert second_result is True


def test_wait_budget_must_be_positive() -> None:
    queue = ApprovalQueue(Store())
    with pytest.raises(ValueError, match="wait_budget"):
        StoreApprovalChannel(queue, task_id="t1", wait_budget=timedelta(0))
    with pytest.raises(ValueError, match="wait_budget"):
        StoreApprovalChannel(queue, task_id="t1", wait_budget=timedelta(seconds=-1))


def test_wait_budget_is_clamped_to_ttl() -> None:
    """A wait_budget longer than ttl would just be waiting for a corpse --
    silently clamped to ttl rather than treated as an error, the same way
    the promoted source's own wait_budget/ttl relationship works."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", ttl=timedelta(seconds=1), wait_budget=timedelta(seconds=999))
    assert channel._wait_budget == timedelta(seconds=1)


# --- TieredGate-aware ticket_gist -------------------------------------------


def test_ticket_gist_extracts_the_real_bash_command_from_a_tiered_gate_prompt() -> None:
    prompt = (
        '[TieredGate] agent asks to run tool \'Bash\'\n  arguments: {"command": "git push origin main"}\n  cwd: C:/repo'
    )
    assert ticket_gist(prompt) == "git push origin main"


def test_ticket_gist_tiered_gate_prompt_with_objective_in_the_command_itself() -> None:
    """A commit message containing the literal substring "Objective:" must
    not be mistaken for the Objective:-marker fallback -- the TieredGate
    shape is detected and handled FIRST, before that generic logic ever
    runs."""
    prompt = (
        "[TieredGate] agent asks to run tool 'Bash'\n"
        '  arguments: {"command": "git commit -m \'Objective: ship it\'"}\n'
        "  cwd: C:/repo"
    )
    assert "Objective: ship it" in ticket_gist(prompt)
    assert not ticket_gist(prompt).startswith("ship it")


def test_ticket_gist_tiered_gate_prompt_falls_back_to_raw_text_when_not_json() -> None:
    """An elided/truncated arguments line can't parse as JSON -- returns
    the raw (still truncated, still useful) text rather than raising."""
    prompt = "[TieredGate] agent asks to run tool 'Bash'\n  arguments: not valid json…\n  cwd: C:/repo"
    assert ticket_gist(prompt) == "not valid json…"


def test_ticket_gist_non_tiered_gate_prompt_uses_the_generic_fallback() -> None:
    """A plain prompt that doesn't start with the TieredGate marker is
    untouched by the new extraction -- existing non-TieredGate behaviour
    (the Objective:/first-line logic) is unchanged."""
    assert ticket_gist("just a plain question") == "just a plain question"


# --- Codex-review fixes -----------------------------------------------------


async def test_ask_consumes_its_own_approval_so_a_later_retry_cannot_reclaim_it() -> None:
    """A ticket resolved through ask()'s OWN polling loop (not a later
    retry's claim_earlier_approval) must still end up consumed -- otherwise
    a LATER call with the identical prompt could silently re-claim and
    re-run this same approval a second time, off one human decision.
    Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, ttl=timedelta(seconds=2))

    async def approve_soon() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    result, _ = await asyncio.gather(channel.ask("please approve"), approve_soon())
    assert result is True

    [ticket] = [ApprovalTicket.model_validate(raw) for _k, raw in queue._store.items(prefix="approval:")]
    assert ticket.consumed_at is not None

    # A second ask() with the IDENTICAL prompt must file a brand-new
    # ticket, not silently re-claim the already-spent approval.
    second = asyncio.create_task(channel.ask("please approve"))
    await asyncio.sleep(0.03)
    assert second.done() is False
    second.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await second


async def test_claim_earlier_approval_survives_cancellation_without_losing_the_outcome() -> None:
    """Cancelling ask() while the claim_earlier_approval call it makes at
    the very top is still in flight must not let that CAS land, unobserved,
    after the cancellation has already propagated -- the same hazard the
    ticket-creation path already guards against. Found by Codex review
    before this ever shipped."""
    queue = ApprovalQueue(Store())
    ticket = queue.create_ticket(task_id="t1", prompt="please approve")
    queue.approve_ticket(ticket.approval_id, actor="marco", channel="telegram")

    real_claim = queue.claim_earlier_approval
    started = asyncio.Event()

    def _slow_claim(*args: Any, **kwargs: Any) -> bool:
        started.set()
        time.sleep(0.05)
        return real_claim(*args, **kwargs)

    queue.claim_earlier_approval = _slow_claim  # type: ignore[method-assign]
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01)

    task = asyncio.create_task(channel.ask("please approve"))
    await started.wait()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # Resolved one way or the other -- not left mid-flight with ask()
    # having already moved on.
    assert queue.get_ticket(ticket.approval_id).consumed_at is not None


async def test_ask_treats_an_externally_expired_ticket_as_terminal() -> None:
    """expire_ticket() doesn't check elapsed time -- another caller can
    expire a ticket long before its own ttl would naturally elapse.
    Without an explicit check for this, ask()'s own polling loop fell
    through to its ttl-based expiry check (which stayed false) and kept
    polling -- and even re-notifying -- an already-terminal ticket
    indefinitely. Found by Codex review before this ever shipped."""
    queue = ApprovalQueue(Store())
    channel = StoreApprovalChannel(queue, task_id="t1", poll_seconds=0.01, ttl=timedelta(seconds=5))

    async def expire_it_early() -> None:
        while not queue.list_pending_tickets():
            await asyncio.sleep(0.005)
        [ticket] = queue.list_pending_tickets()
        queue.expire_ticket(ticket.approval_id)

    result, _ = await asyncio.wait_for(asyncio.gather(channel.ask("please approve"), expire_it_early()), timeout=2.0)
    assert result is False


def test_job_owner_is_alive_rejects_a_recycled_pid_with_a_different_boot_id() -> None:
    """A record whose owner_pid equals THIS process's own pid but whose
    owner_boot_id does NOT match is a DEAD earlier owner that happened to
    be handed this exact pid before -- not this process. Falling through
    to a process-table check would trivially see THIS process (it has
    that pid) and wrongly report the old, dead owner as alive. Found by
    Codex review before this ever shipped."""
    from lazybridge.ext.delegation.jobs import _job_owner_is_alive

    assert _job_owner_is_alive({"owner_pid": os.getpid(), "owner_boot_id": "a-different-boot-id"}) is False
