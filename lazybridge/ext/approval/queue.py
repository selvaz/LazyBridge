"""Durable, Store-backed approval/escalation ticket queue.

Promoted from LazyCEO's ``lazyceo.approvals`` (a sibling project built on
this package) after live production use. Files a ticket and polls the Store
until some OTHER surface resolves it -- a ``Channel`` implementation here
never polls an external service itself, so a Telegram bot, an HTTP API, a
CLI, or all three at once can share exactly one pending-ticket queue with no
risk of two pollers racing the same external inbox for the same decision.

``StoreApprovalChannel`` implements this package's own ``Channel`` protocol
(``async def ask(self, prompt: str) -> bool``, see ``tiered.py``). That
protocol only hands the channel rendered prompt text -- not the structured
request a caller like ``TieredGate`` matched -- so ``ApprovalTicket`` stores
``prompt_hash`` as an AUDIT field (what exactly was shown), not a verified
security binding: nothing re-checks it against anything at approve/reject
time (there is nothing to check it against -- the prompt never changes once
a ticket is created). What actually keeps an approval from being credited to
the wrong action is the CAS ``pending`` -> ``approved``/``rejected``
transition being one-shot per ``approval_id``.

Known accepted gap: the expiry check and the CAS that follows it are not one
atomic step, so a request that crosses ``expires_at`` in the microseconds
between them can still succeed. Given ``DEFAULT_TTL`` is hours, not seconds,
this is a real but currently accepted race, not a practical one.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from lazybridge import Store

#: "approval" (the original meaning: a yes/no gate on one action) vs
#: "escalation" (an agent flagging something for a human to look at -- still
#: resolved through the exact same approve/reject verbs, since a human
#: answering either is still "go ahead" / "don't"). A plain label, not a
#: different state machine -- kept in the same ticket shape and queue on
#: purpose: not enough distinct behavior to justify a second mechanism.
TicketKind = Literal["approval", "escalation"]
TicketStatus = Literal["pending", "approved", "rejected"]

#: Default key prefix when a caller doesn't need multiple independent
#: queues sharing one Store. Pass ``prefix=`` to :class:`ApprovalQueue`
#: for a namespaced queue instead (e.g. LazyCEO's facade uses
#: ``"ceo:approval:"`` to preserve its own on-disk key format).
DEFAULT_PREFIX = "approval:"

#: How long a ticket stays actionable before a stuck/forgotten approval
#: request stops being answerable and the caller must treat it as denied.
DEFAULT_TTL = timedelta(hours=2)

#: Found live: a ticket notified once at creation and never mentioned again
#: left a whole turn blocked for the better part of an hour on a routine
#: 'rm' of the agent's own scratch file -- the human simply never saw (or
#: forgot) the original message. A ticket still pending after this long
#: gets re-notified, on the same schedule, until it's resolved or expires.
DEFAULT_RENOTIFY_INTERVAL = timedelta(minutes=5)


class ApprovalTicket(BaseModel):
    approval_id: str
    task_id: str
    prompt: str
    #: sha256 of ``prompt`` -- an audit field (what exactly was shown when
    #: this ticket was created), NOT a verified security binding: nothing
    #: currently checks it against anything at approve/reject time, since
    #: the prompt is immutable for a ticket's whole lifetime anyway.
    prompt_hash: str
    status: TicketStatus
    kind: TicketKind = "approval"
    actor: str | None = None
    channel: str | None = None
    reason: str | None = None
    created_at: datetime
    expires_at: datetime


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def ticket_gist(prompt: str, *, max_len: int = 220) -> str:
    """The part of a ticket's ``prompt`` that actually distinguishes it from
    another pending one -- NOT just its first line. A caller's own
    confirmation template can put identical boilerplate on line 1 of every
    ticket it files (e.g. "Delegate to Codex with REAL write access..."),
    with the real objective further down after an "Objective:" marker.
    Falls back to the first line when there's no such marker (a plain
    yes/no ticket's prompt IS just the request itself, already the whole
    distinguishing content, nothing to skip past)."""
    marker = "Objective:"
    idx = prompt.find(marker)
    if idx != -1:
        gist = prompt[idx + len(marker) :].strip()
    else:
        # An empty/whitespace-only prompt (create_ticket doesn't reject
        # one) has no lines at all -- splitlines()[0] would raise
        # IndexError. Found by Codex review before this ever shipped.
        lines = prompt.splitlines()
        gist = lines[0] if lines else ""
    gist = " ".join(gist.split())  # collapse embedded newlines/whitespace to one line
    if max_len <= 0:
        # A zero/negative limit has no valid non-empty result; falling
        # through to gist[:negative] would return almost the WHOLE string
        # (Python slice semantics), the opposite of truncating. Found by
        # Codex review before this ever shipped.
        return ""
    if len(gist) <= max_len:
        return gist
    if max_len <= 3:
        # max_len is positive here (the <= 0 case already returned), so
        # this slice is safe -- max_len - 3 going negative is only a
        # problem for the "..." branch below.
        return gist[:max_len]
    return gist[: max_len - 3] + "..."


def _anchor_db_path(store: Store) -> str | None:
    """The one absolute file path this queue will use for EVERY operation
    (reads and writes) for its whole lifetime, computed exactly once, at
    :class:`ApprovalQueue` construction -- never re-derived from
    ``store._db``/``store._conn()`` per call. ``None`` for an in-memory
    store (``db=None`` or ``db=":memory:"``; the latter because SQLite's
    own convention is that every ``sqlite3.connect(":memory:")`` call opens
    a BRAND NEW, unrelated anonymous database), meaning callers use the
    original ``store`` object directly for everything (no sqlite snapshot
    to go stale there, and no separate file to anchor to).

    IMPORTANT distinction between the two in-memory modes: ``db=None`` is
    genuinely thread-safe for cross-thread use -- ``Store`` backs it with a
    plain dict guarded by ``Store._lock``, shared by every thread. ``db=
    ":memory:"`` is NOT: ``Store``'s connection is thread-local, and
    SQLite gives each thread's first connection to ``":memory:"`` its OWN
    brand-new, unrelated anonymous database (not guarded by
    ``Store._lock`` at all -- that lock only covers the dict path). A
    ticket created on one thread against a ``":memory:"`` Store is
    therefore invisible (or worse, ``sqlite3.OperationalError: no such
    table: store``, an uninitialized schema) to ``ApprovalQueue`` calls
    made from any OTHER thread -- this is a pre-existing ``Store``
    limitation this queue cannot paper over from the outside. Use
    ``db=None`` for an in-memory queue that must be usable from more than
    one thread (e.g. behind ``StoreApprovalChannel``, which is explicitly
    meant to be resolved from a different surface/thread than the one
    that called ``ask()``); reserve ``db=":memory:"`` for genuinely
    single-threaded use (e.g. exercising the SQL code paths in a test
    without touching disk). Found by Codex review (four rounds) before
    this ever shipped.

    Why this can't just be ``Path(store._db).resolve()`` at call time:
    ``Store``'s connection is thread-local (a relative db path resolves
    against whatever cwd is active the first time EACH thread calls
    ``store._conn()``), and a process's cwd can change at any point over
    the store's lifetime -- either one alone can make a later
    re-resolution diverge from the file the caller's OTHER code is
    actually reading/writing through, silently pointing this queue at a
    different, empty database instead.

    If ``store`` already has a connection open ON THIS THREAD (the common
    case: constructed and used, then wrapped in an ``ApprovalQueue``, all
    on the same thread), ask SQLite's own catalog (``PRAGMA
    database_list``) what file that connection ACTUALLY has open --
    authoritative regardless of any cwd drift, past or future, since
    SQLite resolves and records the absolute path once, at connection
    time. Only if no connection exists yet on this thread (Store never
    used before now) does this fall back to resolving the raw path
    against the CURRENT cwd, which correctly predicts where Store's own
    eventual first connection will land as long as no further cwd change
    happens between this call and that first connection -- a much
    narrower, best-effort assumption for the "never used yet" case only.
    Found by Codex review (three rounds) before this ever shipped.
    """
    db_path = getattr(store, "_db", None)
    if not db_path or db_path == ":memory:":
        return None
    local = getattr(store, "_local", None)
    if local is not None and hasattr(local, "conn"):
        try:
            for row in store._conn().execute("PRAGMA database_list"):
                # (seq, name, file) -- "main" is the one Store itself
                # reads/writes through; skip any ATTACHed database (Store
                # never attaches one today, but don't assume that stays true).
                if row[1] == "main" and row[2]:
                    return row[2]
        except Exception:
            pass
    from pathlib import Path

    return str(Path(db_path).resolve())


@contextlib.contextmanager
def _anchored_store(store: Store, *, resolved_db_path: str | None) -> Any:
    """A Store anchored to ``resolved_db_path`` for one operation, so every
    read AND write this queue performs targets the SAME physical file
    regardless of which thread calls it or when -- see
    :func:`_anchor_db_path` for why per-call re-resolution is unsafe.

    Doubles as the fix for the original stale-sqlite-snapshot bug this
    queue was promoted with a workaround for (opening fresh each time
    guarantees no long-lived cached connection can be pinned to a stale
    WAL snapshot after a failed write with no rollback -- ``Store.write``/
    ``delete``/``clear`` now recover from that directly too, see
    ``lazybridge/store/__init__.py``, but this stays as an extra,
    self-contained guarantee). Falls back to the original ``store`` object
    when ``resolved_db_path`` is ``None`` (in-memory): reads/writes there
    are guarded by ``Store._lock``, no sqlite snapshot to go stale.
    """
    if resolved_db_path is None:
        yield store
        return
    from lazybridge import Store as _Store

    fresh = _Store(db=resolved_db_path)
    try:
        yield fresh
    finally:
        fresh.close()


def _actionable(raw: dict, now: datetime) -> ApprovalTicket | None:
    """The parsed ticket if it's still pending and unexpired, else None.

    A stale-but-still-"pending" record is left untouched here -- expiry is
    read-time, not a background sweep -- so approve/reject on an expired
    ticket simply fails (returns False) the same way a second decision on an
    already-resolved one does.
    """
    if raw.get("status") != "pending":
        return None
    ticket = ApprovalTicket.model_validate(raw)
    return ticket if ticket.expires_at > now else None


class ApprovalQueue:
    """A durable, Store-backed queue of approval/escalation tickets.

    Keyed under a configurable ``prefix`` so multiple independent queues
    (e.g. one per application, or one shared fleet-wide queue) can live in
    one ``Store`` without key collisions -- construct one ``ApprovalQueue``
    per prefix a caller needs; the object itself holds no connection of its
    own beyond the ``Store`` it's given, so constructing one is cheap.

    ``prefix`` matching is a literal string prefix (same semantics as
    :meth:`Store.items`'s own ``prefix=``), NOT a namespace boundary: a
    queue whose prefix is itself a prefix of another queue's prefix (e.g.
    ``"app:"`` and ``"app:team:"``) is NOT isolated from it -- listing the
    outer queue's tickets will include the inner queue's too, since every
    inner key literally starts with the outer prefix. Choose prefixes that
    are NOT string-prefixes of each other (e.g. distinct leaf names like
    ``"app-a:"``/``"app-b:"``, or always terminate every prefix with a
    character no other prefix in the same Store starts with) if isolation
    between two queues matters. Found by Codex review before this ever
    shipped.

    For an in-memory ``store`` used from more than one thread (the normal
    shape: one thread files a ticket, another surface resolves it), use
    ``Store(db=None)``, NOT ``Store(db=":memory:")`` -- see
    :func:`_anchor_db_path`'s docstring for why the two are not
    interchangeable.
    """

    def __init__(self, store: Store, *, prefix: str = DEFAULT_PREFIX) -> None:
        self._store = store
        self._prefix = prefix
        # Resolved ONCE, here -- see _anchor_db_path's own docstring for
        # why re-resolving per-call/per-thread is unsafe. Every operation
        # below (reads AND writes) routes through this same anchor.
        self._resolved_db_path = _anchor_db_path(store)

    def _key(self, approval_id: str) -> str:
        return f"{self._prefix}{approval_id}"

    def _anchored(self) -> Any:
        return _anchored_store(self._store, resolved_db_path=self._resolved_db_path)

    def create_ticket(
        self, *, task_id: str, prompt: str, kind: TicketKind = "approval", ttl: timedelta = DEFAULT_TTL
    ) -> ApprovalTicket:
        now = datetime.now(UTC)
        ticket = ApprovalTicket(
            approval_id=str(uuid.uuid4()),
            task_id=task_id,
            prompt=prompt,
            prompt_hash=_hash_prompt(prompt),
            status="pending",
            kind=kind,
            created_at=now,
            expires_at=now + ttl,
        )
        with self._anchored() as anchored:
            anchored.write(self._key(ticket.approval_id), ticket.model_dump(mode="json"))
        return ticket

    def get_ticket(self, approval_id: str) -> ApprovalTicket | None:
        with self._anchored() as anchored:
            raw = anchored.read(self._key(approval_id))
        return ApprovalTicket.model_validate(raw) if isinstance(raw, dict) else None

    def list_pending_tickets(self, *, limit: int = 100) -> list[ApprovalTicket]:
        """Unexpired tickets still awaiting a decision, oldest first (FIFO queue)."""
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        now = datetime.now(UTC)
        with self._anchored() as anchored:
            items = anchored.items(prefix=self._prefix)
        tickets = [
            ApprovalTicket.model_validate(raw)
            for _key, raw in items
            if isinstance(raw, dict) and raw.get("status") == "pending"
        ]
        tickets = [t for t in tickets if t.expires_at > now]
        tickets.sort(key=lambda t: t.created_at)
        return tickets[:limit]

    def approve_ticket(self, approval_id: str, *, actor: str, channel: str) -> bool:
        """Approve a pending, unexpired ticket. CAS: a second approval, a
        rejection, or an expired ticket all return False."""
        key = self._key(approval_id)
        with self._anchored() as anchored:
            raw = anchored.read(key)
            if not isinstance(raw, dict):
                return False
            ticket = _actionable(raw, datetime.now(UTC))
            if ticket is None:
                return False
            updated = ticket.model_copy(update={"status": "approved", "actor": actor, "channel": channel})
            return anchored.compare_and_swap(key, raw, updated.model_dump(mode="json"))

    def reject_ticket(self, approval_id: str, *, actor: str, channel: str, reason: str) -> bool:
        key = self._key(approval_id)
        with self._anchored() as anchored:
            raw = anchored.read(key)
            if not isinstance(raw, dict):
                return False
            ticket = _actionable(raw, datetime.now(UTC))
            if ticket is None:
                return False
            updated = ticket.model_copy(
                update={"status": "rejected", "actor": actor, "channel": channel, "reason": reason}
            )
            return anchored.compare_and_swap(key, raw, updated.model_dump(mode="json"))


class StoreApprovalChannel:
    """A ``lazybridge.ext.approval`` ``Channel`` backed by an :class:`ApprovalQueue`.

    Files a ticket and polls the queue until some other surface (a Telegram
    command, an HTTP API, an MCP tool) resolves it -- never runs its own
    poll of an external service. One instance is scoped to a single
    ``task_id``: construct a fresh channel (and thus a fresh ``TieredGate``)
    per dispatched task, so every ticket this channel files already carries
    the right task_id.

    ``notify``, when given, is awaited once per ticket right after it's
    created, with the ticket and a ready-to-send message (the prompt plus
    the approval_id) -- WITHOUT it, a human has no way to learn a ticket
    exists at all: this channel only ever writes to the queue and polls it
    back; nothing here pushes anything anywhere. A ``notify`` failure is
    logged and swallowed, never allowed to fail the approval itself -- the
    ticket still exists and is still answerable through any surface that
    reads the queue directly, even if the push notification didn't make it.

    Past the FIRST notification, ``renotify_interval`` re-sends the same
    kind of message on a fixed schedule for as long as the ticket stays
    pending -- found live: a ticket notified exactly once, then never
    mentioned again, left an entire turn blocked for close to an hour
    because the human simply never saw (or forgot) that first message. Set
    to ``None`` to go back to a single notification.
    """

    name = "store-approval-queue"

    def __init__(
        self,
        queue: ApprovalQueue,
        *,
        task_id: str,
        poll_seconds: float = 2.0,
        ttl: timedelta = DEFAULT_TTL,
        notify: Callable[[ApprovalTicket, str], Awaitable[None]] | None = None,
        notify_timeout: float = 10.0,
        renotify_interval: timedelta | None = DEFAULT_RENOTIFY_INTERVAL,
        kind: TicketKind = "approval",
    ):
        if poll_seconds <= 0:
            # A poll interval that isn't strictly positive means
            # min(poll_seconds, remaining_ttl) resolves to
            # asyncio.sleep(0-or-negative) on every iteration -- a hot
            # loop of Store reads (and, for a file-backed queue, a fresh
            # SQLite connection opened and closed on every single
            # iteration) hammering the database until the ticket's TTL
            # (up to hours) finally expires. Found by Codex review before
            # this ever shipped.
            raise ValueError(f"poll_seconds must be positive, got {poll_seconds}")
        self._queue = queue
        self._task_id = task_id
        self._poll_seconds = poll_seconds
        self._ttl = ttl
        self._notify = notify
        self._notify_timeout = notify_timeout
        self._renotify_interval = renotify_interval
        #: Stamped on every ticket this instance creates -- "escalation" for
        #: an agent's own channel, "approval" for a default TieredGate "ask"
        #: gate. See TicketKind's docstring.
        self._kind = kind

    async def _send(self, ticket: ApprovalTicket, message: str) -> None:
        if self._notify is None:
            return
        # Bounded by whichever is SHORTER: notify_timeout, or the ticket's
        # own remaining lifetime. notify_timeout alone (10s default) can
        # still make ask() overshoot a short ttl by nearly that whole
        # amount whenever a notify call stalls -- e.g. ttl=50ms with a
        # hanging notifier would return ~10s late instead of ~50ms late,
        # contradicting the timeout this channel documents. Found by
        # Codex review before this ever shipped.
        remaining = (ticket.expires_at - datetime.now(UTC)).total_seconds()
        timeout = min(self._notify_timeout, max(remaining, 0.0))
        try:
            # A bounded wait, not just a try/except: a notify callback that
            # hangs (a stalled network transport, e.g.) would otherwise
            # block `ask()` from EVER reaching its own status-polling loop
            # below -- so even a ticket approved through another surface
            # a second later would sit unnoticed for as long as the
            # notify call never returns. A timeout turns that into an
            # ordinary swallowed notify failure instead. Found by Codex
            # review before this ever shipped.
            await asyncio.wait_for(self._notify(ticket, message), timeout=timeout)
        except Exception:
            logging.getLogger(__name__).exception(
                "notify failed for ticket %s -- it still exists and is still answerable", ticket.approval_id
            )

    async def ask(self, prompt: str) -> bool:
        ticket = self._queue.create_ticket(task_id=self._task_id, prompt=prompt, kind=self._kind, ttl=self._ttl)
        # The id is part of the COMMAND, not just shown alongside it: this
        # library has no bare-"/approve"-resolves-the-one-pending-ticket
        # convenience of its own (a caller like LazyCEO may layer that on
        # top of its OWN command parser, but this message must be
        # self-sufficient without assuming one exists) -- with more than
        # one pending ticket sharing a queue, or a stateless CLI/API
        # handler receiving the reply, "/approve" alone can't be resolved
        # to a specific approval_id at all. Found by Codex review before
        # this ever shipped.
        message = f"{prompt}\n\nReply /approve {ticket.approval_id} or /reject {ticket.approval_id} <reason>."
        try:
            await self._send(ticket, message)
            last_notified = datetime.now(UTC)
            while True:
                current = self._queue.get_ticket(ticket.approval_id)
                if current is None or current.status == "rejected":
                    return False
                if current.status == "approved":
                    return True
                now = datetime.now(UTC)
                if current.expires_at <= now:
                    return False
                if self._renotify_interval is not None and now - last_notified >= self._renotify_interval:
                    waited_minutes = int((now - ticket.created_at).total_seconds() // 60)
                    reminder = f"⏰ Still waiting on this one ({waited_minutes} min) -- {message}"
                    await self._send(current, reminder)
                    last_notified = now
                # Capped at the ticket's remaining lifetime: an
                # unconditional sleep(poll_seconds) would overshoot
                # expires_at by up to a full poll interval whenever
                # poll_seconds is longer than the configured ttl (a
                # valid, if unusual, combination -- both are independent
                # caller-set parameters). Found by Codex review before
                # this ever shipped.
                #
                # Recomputed AFTER the reminder _send above, not reusing
                # the `now` captured before it: a reminder notify can
                # itself take real time (bounded by notify_timeout, which
                # is itself capped by the ticket's remaining lifetime --
                # see _send -- but that can still be a meaningful chunk of
                # a short ttl). Sleeping against the pre-notify `now`
                # would overshoot expires_at by however long that send
                # actually took. Found by Codex review before this ever
                # shipped.
                now = datetime.now(UTC)
                seconds_until_expiry = (current.expires_at - now).total_seconds()
                await asyncio.sleep(min(self._poll_seconds, max(seconds_until_expiry, 0.0)))
        except BaseException as exc:
            # Best-effort: if this coroutine exits WITHOUT returning --
            # cancellation (a caller's own timeout, task shutdown, process
            # teardown) OR any other exception (a transient SQLite error
            # from get_ticket(), a validation failure on a damaged record,
            # anything else polling can raise) -- nothing is left to
            # consume the eventual decision, but the ticket itself would
            # otherwise sit "pending" (still listed by
            # list_pending_tickets(), still actionable, still re-notified
            # by anything ELSE polling it directly) until its full TTL
            # expiry, hours later, with an operator able to "approve" a
            # request nobody is waiting on anymore. Retire it now instead,
            # then re-raise the ORIGINAL exception unchanged -- this is
            # cleanup, not error handling. Found by Codex review before
            # this ever shipped.
            reason = "ask() was cancelled" if isinstance(exc, asyncio.CancelledError) else f"ask() raised: {exc!r}"
            with contextlib.suppress(Exception):
                self._queue.reject_ticket(ticket.approval_id, actor="system", channel=self.name, reason=reason)
            raise
