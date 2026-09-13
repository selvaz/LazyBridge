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
import logging
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
        await asyncio.sleep(0.05)
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
        await asyncio.sleep(0.05)
        [normal_ticket] = [t for t in queue.list_pending_tickets() if t.prompt == "normal one"]
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
        await asyncio.sleep(0.05)
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
        await asyncio.sleep(0.05)
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
            await asyncio.sleep(0.05)
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
        await asyncio.sleep(0.03)
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
