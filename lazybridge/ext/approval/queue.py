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

Known accepted gap: ``ask()`` retires an orphaned ticket -- one whose
``create_ticket`` call was still in flight when ``ask()`` itself was
cancelled -- via a detached asyncio task that ``ask()`` then waits for
(shielded, in a retry loop immune to however many further times ``ask()``
itself gets cancelled) before letting its own cancellation become
observable to its caller (see ``StoreApprovalChannel.ask()``'s own
comments for the earlier attempts that didn't hold up). That task is
still an ordinary asyncio task, though, not immune to the event loop
itself being torn down -- ``asyncio.run()`` exiting cancels every
remaining task, this retiring one included, before the underlying
offloaded ``create_ticket`` thread is guaranteed to have finished and been
consumed. In that specific compound case (loop shutdown racing a
cancelled-during-creation ``ask()``) the ticket can land, get written, and
never be retired, sitting pending until its own TTL expiry hours later --
the same category of consequence as the CAS/expiry race above, not a
correctness or security issue, and not worth a dedicated thread-native
(non-asyncio) callback to close given how narrow and low-stakes it is.

Known accepted gap: waiting for that retirement task guarantees it has
been ATTEMPTED and fully resolved before ``ask()``'s cancellation becomes
observable -- not that it necessarily WON. Its own ``reject_ticket`` call
is an ordinary CAS that can lose to a genuinely concurrent external
approve reaching the Store first, the instant the ticket becomes visible
(some OTHER surface already polling this same queue, independent of
``ask()`` entirely). No code inside ``ask()`` can prevent that outright --
only shrink the window for it, which retiring as the very next thing once
creation completes, with nothing else interposed, already does about as
tightly as this queue's own design allows.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
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
#: "expired" is distinct from "rejected": nobody said no, nobody said
#: anything -- :meth:`ApprovalQueue.expire_ticket` is how a caller that
#: gave up waiting (see :class:`StoreApprovalChannel`'s ``wait_budget``)
#: records that distinction durably instead of leaving the ticket at
#: "pending" forever (still listed by ``list_pending_tickets``, still
#: actionable, with no sign anything ever stopped waiting on it) or
#: silently treating it the same as a human refusal (which it provably was
#: not, and deserves a different repair).
TicketStatus = Literal["pending", "approved", "rejected", "expired"]

#: Default key prefix when a caller doesn't need multiple independent
#: queues sharing one Store. Pass ``prefix=`` to :class:`ApprovalQueue`
#: for a namespaced queue instead (e.g. LazyCEO's facade uses
#: ``"ceo:approval:"`` to preserve its own on-disk key format).
DEFAULT_PREFIX = "approval:"

#: Default prefix for the durable denial audit trail -- see
#: :meth:`ApprovalQueue.record_denial`. Configurable per :class:`ApprovalQueue`
#: the same way ``prefix`` is, for a caller preserving an existing on-disk
#: layout.
DEFAULT_DENIAL_PREFIX = "approval-denial:"

#: Default prefix for the durable notify-failure audit trail -- see
#: :meth:`ApprovalQueue.record_notify_failure`.
DEFAULT_NOTIFY_FAILURE_PREFIX = "approval-notify-failure:"

#: How long a ticket stays actionable before a stuck/forgotten approval
#: request stops being answerable and the caller must treat it as denied.
#: This is NOT how long a waiter blocks for -- see ``wait_budget`` on
#: :class:`StoreApprovalChannel` for that separate, usually much shorter,
#: number. A ticket can stay actionable for hours while the agent that
#: filed it has long since stopped waiting and moved on.
DEFAULT_TTL = timedelta(hours=2)

#: How long :meth:`StoreApprovalChannel.ask` waits before giving up on a
#: ticket that is still pending and still well within its ``ttl``.
#:
#: ``ask()`` blocks whatever called it -- usually a tool call inside a
#: bounded turn -- so a wait long enough to survive a human being away for
#: hours would make that turn itself hang for hours. Splitting "how long is
#: this request answerable" (``ttl``) from "how long will THIS caller wait
#: for it" (``wait_budget``) is what lets a ticket stay open far longer than
#: any one wait: giving up here does not expire the ticket -- it stays
#: pending and answerable until ``ttl``, so a human who answers after the
#: caller stopped waiting still lands on a live ticket, not a corpse. Equal
#: to ``DEFAULT_TTL`` by default, which reproduces this module's own
#: pre-split behaviour exactly: a caller that never sets ``wait_budget``
#: waits the full ``ttl``, same as before this existed.
DEFAULT_WAIT_BUDGET = DEFAULT_TTL

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
    #: A caller-defined flag meaning whatever the caller needs it to mean
    #: by way of a policy layered on top of this queue -- e.g. "this ticket
    #: must be resolved by a specific human channel, never auto-settled by
    #: anything else with queue access." This module does not enforce
    #: anything about it; it is pure metadata, carried through
    #: :meth:`ApprovalQueue.create_ticket` and
    #: :class:`StoreApprovalChannel`'s constructor so a caller's own
    #: resolution logic can read it back off the ticket.
    operator_only: bool = False
    actor: str | None = None
    channel: str | None = None
    reason: str | None = None
    created_at: datetime
    expires_at: datetime
    #: When an approval given AFTER a waiter stopped waiting for it (see
    #: ``wait_budget``) was actually spent by a later attempt -- see
    #: :meth:`ApprovalQueue.claim_earlier_approval`. An approval sitting
    #: unconsumed is worth nothing: without this field nothing anywhere
    #: records whether a "yes" that arrived late was ever actually acted
    #: on, so the audit trail could show an approval for work that never
    #: ran. Set by compare-and-swap, so two attempts racing to claim the
    #: same approval cannot both win.
    consumed_at: datetime | None = None


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


#: The tool/command name inside a TieredGate-rendered prompt's first line
#: (``lazybridge.ext.approval.tiered._render``'s own
#: ``"[TieredGate] agent asks to run {kind} '{name}'"`` wording).
_TOOL_IN_PROMPT = re.compile(r"asks to run \S+ '([^']+)'")


def _tool_name_from_prompt(prompt: str) -> str | None:
    """Best-effort tool/command name dug out of a TieredGate-rendered
    prompt, for an audit record that needs it (see
    :meth:`ApprovalQueue.record_denial`) but was only ever handed the
    rendered text -- :class:`~lazybridge.ext.approval.tiered.Channel` never
    sees the structured ``ApprovalRequest`` TieredGate matched, only this
    string."""
    match = _TOOL_IN_PROMPT.search(prompt or "")
    return match.group(1) if match else None


def _tiered_gate_gist(prompt: str) -> str | None:
    """Extract the real command/arguments from a TieredGate-rendered prompt
    (``lazybridge.ext.approval.tiered._render``), or ``None`` if ``prompt``
    isn't shaped like one -- the caller falls back to the generic logic in
    that case.

    A TieredGate-created ticket's first line is the generic
    ``"[TieredGate] agent asks to run tool 'Bash'"`` sentence for every
    single Bash ticket, with the actual command buried in an
    ``"arguments: {...}"`` line below -- there is no ``"Objective:"``
    marker to find, so falling back to the first line (the plain,
    non-TieredGate behaviour) would show that identical sentence for every
    pending Bash ticket, indistinguishable from one another. Checked
    first, and never falling through to the ``Objective:``/first-line
    logic below for a prompt shaped this way: if the command text itself
    happens to contain the literal substring ``"Objective:"`` (a commit
    message can), searching for that marker across the WHOLE prompt would
    match inside the arguments JSON and hide the tool/kind context that
    came before it.

    Best-effort past that: an elided/truncated ``arguments:`` line can't
    parse as JSON, so this returns the raw (still truncated, still useful)
    text rather than raising."""
    import json

    if not prompt.startswith("[TieredGate] "):
        return None
    args_line = next((line for line in prompt.splitlines() if line.strip().startswith("arguments:")), None)
    if args_line is None:
        return None
    raw = args_line.split("arguments:", 1)[1].strip()
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(parsed, dict) and isinstance(parsed.get("command"), str):
        return parsed["command"]
    return raw


def ticket_gist(prompt: str, *, max_len: int = 220) -> str:
    """The part of a ticket's ``prompt`` that actually distinguishes it from
    another pending one -- NOT just its first line. A caller's own
    confirmation template can put identical boilerplate on line 1 of every
    ticket it files (e.g. "Delegate to Codex with REAL write access..."),
    with the real objective further down after an "Objective:" marker.

    A prompt rendered by :class:`~lazybridge.ext.approval.tiered.TieredGate`
    has the same problem in a different shape -- see
    :func:`_tiered_gate_gist`, checked first here -- so that case is
    handled before the ``Objective:``/first-line fallback below ever runs.

    Falls back to the first line when neither marker applies (a plain
    yes/no ticket's prompt IS just the request itself, already the whole
    distinguishing content, nothing to skip past)."""
    gist = _tiered_gate_gist(prompt)
    if gist is None:
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


def _unwrap_store(store: Any) -> Any:
    """Follow a wrapper's own public ``inner`` attribute (e.g.
    :class:`~lazybridge.store.encryption.EncryptedStoreAdapter`, which
    documents itself as usable anywhere a plain ``Store`` is) down to
    whatever object actually looks like a real ``Store`` -- i.e. exposes
    its own ``_db`` -- recursively, in case of nested wrappers. A wrapper
    exposes none of ``Store``'s private ``_db``/``_local`` attributes
    itself (those belong to the ``Store`` it delegates to), so inspecting
    a wrapper directly for either silently misreports it: an encrypted
    ``Store(db=":memory:")`` would look "thread-safe" to
    :attr:`ApprovalQueue.safe_to_call_from_any_thread` (it is not -- see
    that property's own docstring), producing exactly the
    cross-thread-``:memory:`` breakage this module exists to avoid, just
    one layer removed. Only used for THAT determination, not for
    :func:`_anchor_db_path`'s own resolution -- a wrapper's anchored
    fresh-connection Store would have to be re-wrapped in the SAME
    encryption (or whatever else) to stay correct, which this module has
    no principled way to do generically; returning ``None`` there (every
    operation staying on the original wrapped ``store`` instead) is the
    safe, intentional fallback for anything that isn't a real ``Store``.
    Found by Codex review before this ever shipped.

    Known accepted gap this leaves: a wrapped store backed by a FILE
    (``EncryptedStoreAdapter(Store(db="relative.sqlite"), ...)``) never
    gets the cwd-drift protection :func:`_anchor_db_path` exists for --
    each thread's own first connection to it (offloading IS correct here;
    see above) resolves the relative path against WHATEVER cwd is active
    on that thread at that moment, exactly the hazard this whole anchoring
    mechanism was built to close for an unwrapped ``Store``. Given no
    generic re-wrap, the only real fix is avoiding the combination:
    prefer an absolute path (or a cwd that never changes for the life of
    the process) when handing a file-backed ``Store`` to a wrapper this
    queue will also use. Found by Codex review before this ever shipped.
    """
    seen: set[int] = set()
    while not hasattr(store, "_db") and hasattr(store, "inner") and id(store) not in seen:
        seen.add(id(store))
        store = store.inner
    return store


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
            # PRAGMA database_list failed for any reason (a closed
            # connection, a driver quirk) -- fall through to the
            # best-effort raw-path resolution below rather than raising,
            # since this is a best-effort anchor, not a required success.
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

    with _Store(db=resolved_db_path) as fresh:
        yield fresh


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

    def __init__(
        self,
        store: Store,
        *,
        prefix: str = DEFAULT_PREFIX,
        denial_prefix: str = DEFAULT_DENIAL_PREFIX,
        notify_failure_prefix: str = DEFAULT_NOTIFY_FAILURE_PREFIX,
    ) -> None:
        self._store = store
        self._prefix = prefix
        self._denial_prefix = denial_prefix
        self._notify_failure_prefix = notify_failure_prefix
        # Resolved ONCE, here -- see _anchor_db_path's own docstring for
        # why re-resolving per-call/per-thread is unsafe. Every operation
        # below (reads AND writes) routes through this same anchor.
        self._resolved_db_path = _anchor_db_path(store)

    def _key(self, approval_id: str) -> str:
        return f"{self._prefix}{approval_id}"

    def _anchored(self) -> Any:
        return _anchored_store(self._store, resolved_db_path=self._resolved_db_path)

    @property
    def safe_to_call_from_any_thread(self) -> bool:
        """``False`` only for ``Store(db=":memory:")`` -- the one
        in-memory mode that is NOT safe to call from a different thread
        than whichever one already has a connection open, because it
        relies on reusing that thread's own thread-local SQLite
        connection (see :func:`_anchor_db_path`'s docstring). ``Store
        (db=None)`` is safe (guarded by ``Store._lock``, not a
        connection); any real file path is safe too (every operation
        opens a FRESH connection from the resolved absolute path, not a
        cached thread-local one). ``StoreApprovalChannel`` reads this to
        decide whether it's safe to offload this queue's calls to a
        worker thread.

        Unwraps a transparent wrapper (e.g. ``EncryptedStoreAdapter``)
        down to the real ``Store`` first -- encryption is a pure value
        transform with no threading semantics of its own, so this
        question is really about whatever ``Store`` is actually doing the
        I/O underneath, not the wrapper sitting in front of it. Found by
        Codex review before this ever shipped.
        """
        return getattr(_unwrap_store(self._store), "_db", None) != ":memory:"

    def create_ticket(
        self,
        *,
        task_id: str,
        prompt: str,
        kind: TicketKind = "approval",
        ttl: timedelta = DEFAULT_TTL,
        operator_only: bool = False,
    ) -> ApprovalTicket:
        if ttl <= timedelta(0):
            # A nonpositive ttl writes a ticket whose expires_at is
            # already at or before created_at -- immediately invisible to
            # list_pending_tickets() (which filters on `expires_at >
            # now`) and unapprovable/unrejectable (both check the same
            # way via `_actionable`), yet still durably stored with
            # status "pending" forever: a configuration mistake would
            # silently deny every request through this queue while
            # accumulating misleading pending-looking records nothing can
            # ever act on. Found by Codex review before this ever
            # shipped.
            raise ValueError(f"ttl must be positive, got {ttl}")
        now = datetime.now(UTC)
        ticket = ApprovalTicket(
            approval_id=str(uuid.uuid4()),
            task_id=task_id,
            prompt=prompt,
            prompt_hash=_hash_prompt(prompt),
            status="pending",
            kind=kind,
            operator_only=operator_only,
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
        """Unexpired tickets still awaiting a decision, oldest first (FIFO queue).

        Known accepted gap: this loads and deserializes EVERY record under
        ``prefix`` -- resolved and expired tickets included -- before
        filtering to pending ones and applying ``limit``, so cost is
        proportional to the queue's entire history, not to how many are
        actually pending; an encrypted store additionally decrypts every
        historical value on every call. Fine for a queue with a bounded or
        modest lifetime; for one meant to run indefinitely with many
        tickets ever filed, pair this with an external pruning/archival
        job (there's no built-in one) or scope ``prefix`` narrowly enough
        to keep each queue's own history small. Found by Codex review
        before this ever shipped.
        """
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

    def expire_ticket(self, approval_id: str) -> bool:
        """Record that nobody answered in time.

        Meant to be called by a waiter that is giving up (see
        :class:`StoreApprovalChannel`'s ``wait_budget``), not by a
        background sweep -- the read-time expiry filter in ``_actionable``/
        ``list_pending_tickets`` still has to stand on its own, because a
        waiter can die with its process and leave a record nobody will
        ever come back to transition. This only closes the ones a caller
        actually watched expire.

        Without it a ticket sits at ``"pending"`` forever even once it is
        truly dead, so any count of "waiting on a human" mixes live
        questions with corpses, and nothing distinguishes "the ticket was
        answered no" from "nobody ever saw it" -- two different situations
        that deserve two different responses, not one shared status.

        CAS: only transitions a ticket that is still ``"pending"`` at the
        moment this runs -- a concurrent approve/reject that lands first
        wins, and this call simply reports ``False``, the same "lost the
        race" contract every other mutator in this class already has.
        """
        key = self._key(approval_id)
        with self._anchored() as anchored:
            raw = anchored.read(key)
            if not isinstance(raw, dict) or raw.get("status") != "pending":
                return False
            ticket = ApprovalTicket.model_validate(raw)
            updated = ticket.model_copy(update={"status": "expired"})
            return anchored.compare_and_swap(key, raw, updated.model_dump(mode="json"))

    def claim_earlier_approval(
        self, *, task_id: str, prompt: str, operator_only: bool = False, now: datetime | None = None
    ) -> bool:
        """Spend an approval a human gave after a previous waiter had
        already stopped waiting for it.

        Without this, splitting "how long a ticket stays answerable"
        (``ttl``) from "how long one waiter blocks for it" (``wait_budget``)
        is a false promise: the ticket stays live, a human answers it, and
        nothing anywhere ever consumes that answer -- the queue ends up
        recording a "yes" for work that never ran, which is worse than
        having asked nothing at all. A caller that retries the identical
        request (same ``task_id``, same exact rendered ``prompt``) after an
        earlier wait gave up should check this FIRST, before filing a new
        ticket for the same question.

        Deliberately narrow:

        - the prompt must match EXACTLY, by hash -- not the tool, not the
          directory, the WHOLE rendered text a human was shown, arguments
          included. A different command is a different question.
        - same ``task_id``.
        - same ``operator_only`` scope. A ticket approved while
          ``operator_only=False`` (any resolver could have answered it,
          including a non-human one layered on top of this queue) must
          never satisfy a LATER request made with ``operator_only=True``
          -- that flag exists so a caller's own resolver can refuse to
          answer certain tickets itself and insist a specific human
          channel does; silently reusing a looser-scoped approval would
          let the stricter request through without ever being seen by
          that channel. The reverse (reusing an ``operator_only=True``
          approval for an ``operator_only=False`` request) is refused
          too, for simplicity and because "matching scope" is the whole
          point -- an exact match, not a one-way escalation rule. Found
          by Codex review before this ever shipped.
        - the candidate was created under THIS consumption protocol in
          the first place -- see the ``consumed_at`` key-presence check
          below.
        - still inside the ticket's own ``expires_at`` window.
        - once -- ``consumed_at`` is set by compare-and-swap, so two
          attempts racing for the same approval cannot both win.

        For comparison, ``lazybridge.ext.approval.tiered.TieredGate``'s own
        ``session`` tier already grants reuse for a whole process on a far
        coarser key (provider + kind + tool + cwd + rule fingerprint), with
        no expiry at all. An exact-text, single-use, task-scoped,
        scope-matched reuse is strictly narrower than that.

        Known accepted gap, inherited from ``prompt_hash``'s own documented
        nature (module docstring): this matches on the HASH of whatever
        rendered text ``ask()`` was given, which may itself be a lossy
        rendering of the real request -- ``TieredGate``'s own renderer
        elides long arguments past a fixed budget and redacts
        secret-shaped substrings, and a caller's own confirmation template
        (e.g. ``make_codex_writer``'s) elides its objective preview too.
        Two DIFFERENT real requests that happen to render down to
        IDENTICAL visible text (both truncated to the same elided tail, or
        both containing a secret masked the same way) hash identically,
        so an approval intended for one could, in principle, be claimed by
        a later, different request that collides with it under elision.
        This is the same non-binding nature ``prompt_hash`` already has as
        a pure AUDIT field, now also load-bearing for reuse -- closing it
        would need either a wider/lossless render budget (a separate,
        broader change than this queue owns) or binding on something
        beyond rendered text, which the ``Channel`` protocol does not hand
        this queue. Narrow in practice (needs a late approval, a
        ``wait_budget`` shorter than ``ttl``, AND a real elision
        collision), not closed here. Found by Codex review.
        """
        moment = now or datetime.now(UTC)
        wanted = _hash_prompt(prompt)
        with self._anchored() as anchored:
            items = anchored.items(prefix=self._prefix)
            for key, raw in items:
                if not isinstance(raw, dict) or raw.get("status") != "approved":
                    continue
                # A record written before `consumed_at` existed (any
                # ticket created by a pre-1.7.0 ApprovalQueue) has no
                # "consumed_at" KEY at all -- `raw.get("consumed_at")`
                # alone cannot tell that apart from a ticket deliberately
                # left unconsumed under THIS protocol (which always
                # serializes the key, as `null`, at creation). Without
                # this presence check, an old, already-acted-on approval
                # from before this queue ever tracked consumption would
                # be silently claimable and re-executed a second time, on
                # no new human decision at all. A legacy ticket is simply
                # never eligible for reuse -- it was never meant to be.
                # Found by Codex review before this ever shipped.
                if "consumed_at" not in raw:
                    continue
                # Scope must match EXACTLY, not just the ticket matching
                # or being looser: a ticket approved with
                # operator_only=False (any resolver could have answered
                # it) must never satisfy a request now asking with
                # operator_only=True, which exists precisely so a
                # caller's own resolver can refuse to answer certain
                # tickets itself and insist a specific human channel
                # does -- reusing a looser approval would let the
                # stricter request through unseen by that channel. The
                # reverse is refused too, for the same "matching scope,
                # not escalation" reason. ``.get(..., False)`` mirrors
                # ApprovalTicket.operator_only's own default, for a
                # record that predates this field but somehow already
                # passed the consumed_at check above (never, in
                # practice, since both fields were added together, but
                # this stays correct even if that ever changes). Found
                # by Codex review before this ever shipped.
                if raw.get("operator_only", False) != operator_only:
                    continue
                if raw.get("consumed_at") or raw.get("prompt_hash") != wanted or raw.get("task_id") != task_id:
                    continue
                ticket = ApprovalTicket.model_validate(raw)
                if ticket.expires_at <= moment:
                    continue
                updated = ticket.model_copy(update={"consumed_at": moment})
                if anchored.compare_and_swap(key, raw, updated.model_dump(mode="json")):
                    return True
        return False

    def mark_approval_consumed(self, approval_id: str, *, now: datetime | None = None) -> bool:
        """Spend an approved ticket's OWN approval, so a later
        :meth:`claim_earlier_approval` for an identical ``(task_id, prompt)``
        can never find it still "approved" and unconsumed.

        Without this, a ticket resolved through the ORDINARY polling path
        in :meth:`StoreApprovalChannel.ask` (a human answers while that
        same call is still waiting, not via a later retry) stayed
        "approved" with ``consumed_at`` left ``None`` forever -- the exact
        shape :meth:`claim_earlier_approval` looks for, so a LATER call
        with the identical prompt could silently re-claim and re-run that
        same approval a second time, off one human decision. Found by
        Codex review before this ever shipped.

        Best-effort: returns ``False`` (not an error) for a ticket that
        isn't ``"approved"``, or is already consumed -- a caller that
        already observed "approved" through its own read proceeds either
        way; this call only prevents a FUTURE reuse, it doesn't change
        what already happened.
        """
        moment = now or datetime.now(UTC)
        key = self._key(approval_id)
        with self._anchored() as anchored:
            raw = anchored.read(key)
            if not isinstance(raw, dict) or raw.get("status") != "approved" or raw.get("consumed_at"):
                return False
            ticket = ApprovalTicket.model_validate(raw)
            updated = ticket.model_copy(update={"consumed_at": moment})
            return anchored.compare_and_swap(key, raw, updated.model_dump(mode="json"))

    def record_denial(self, record: Any) -> None:
        """Persist one refusal, with the reason the caller actually had.

        Generic enough to wire directly as
        :class:`~lazybridge.ext.approval.tiered.TieredGate`'s own
        ``on_record`` callback (``on_record=queue.record_denial``): that
        callback fires for every decision the gate makes, allow-tier reads
        included, so only ``action in {"deny", "denied"}`` records
        anything here -- putting a Store write in front of every "allowed,
        as always" call would cost more than the trail is worth. Reads
        ``record`` via ``getattr`` rather than requiring a specific type,
        so both a :class:`~lazybridge.ext.approval.tiered.AuditRecord` and
        a plain ``SimpleNamespace`` built by a caller's own refusal path
        (see :class:`StoreApprovalChannel`'s ``ask()``) work identically.

        Best-effort and never raising: a failure to record WHY something
        was refused must not become a second fault stacked on top of the
        refusal itself.
        """
        action = str(getattr(record, "action", "") or "")
        if action not in ("deny", "denied"):
            return
        try:
            with self._anchored() as anchored:
                anchored.write(
                    f"{self._denial_prefix}{uuid.uuid4()}",
                    {
                        "denied_at": datetime.now(UTC).isoformat(),
                        "tool_name": str(getattr(record, "tool_name", "") or ""),
                        "kind": str(getattr(record, "kind", "") or ""),
                        "tier": str(getattr(record, "tier", "") or ""),
                        # The distinguishing field. "no rule matched" (the
                        # default deny), "a human said no", and "nobody
                        # answered in time" are three different repairs --
                        # this is what tells them apart later.
                        "message": str(getattr(record, "message", "") or ""),
                        "responder": str(getattr(record, "responder", "") or ""),
                        "cwd": str(getattr(record, "cwd", "") or ""),
                        "arguments_preview": str(getattr(record, "arguments_preview", "") or ""),
                    },
                )
        except Exception:
            logging.getLogger(__name__).exception("failed to record a denial")

    def record_notify_failure(self, *, source: str, text: str, error: str) -> None:
        """Durable, queryable audit record of a swallowed ``notify()``
        failure.

        Deliberately additive-only: never read back by anything that
        changes behaviour, never retried from here, never allowed to raise
        (a failure recording a failure must not itself become a second
        failure -- caught and logged, not propagated, same posture as the
        notify call it is describing). ``source`` identifies the call site
        in human terms (e.g. ``"StoreApprovalChannel:approval"``) so a
        reader doesn't have to guess which of several notify paths dropped
        a message. Without this, the only record of a failed notification
        was a single ``logging.exception`` call, visible only by tailing
        that one process's raw log file -- nobody could later ask "did a
        notification ever silently fail?" without doing exactly that.
        """
        try:
            with self._anchored() as anchored:
                anchored.write(
                    f"{self._notify_failure_prefix}{uuid.uuid4()}",
                    {
                        "source": source,
                        "text": text,
                        "error": error,
                        "failed_at": datetime.now(UTC).isoformat(),
                    },
                )
        except Exception:
            logging.getLogger(__name__).exception("failed to record a notify failure for source %r", source)


#: Kept alive here so nothing else has to hold a reference: asyncio only
#: keeps a WEAK reference to a Task once nothing else does, so an
#: un-awaited watcher task (see `StoreApprovalChannel.ask()` below) could
#: otherwise be garbage-collected mid-flight. Each task discards itself via
#: its own done-callback.
_background_tasks: set[asyncio.Task[Any]] = set()


def _is_async_callable(func: Callable[..., Any]) -> bool:
    """True for a plain ``async def`` function, AND for a callable
    OBJECT whose own ``__call__`` is ``async def`` -- ``inspect.
    iscoroutinefunction(obj)`` alone only recognizes the former:
    ``obj()`` for such an object still returns a coroutine without
    running any of its body (exactly the property this whole check
    exists to confirm), but ``iscoroutinefunction`` inspects the object
    itself, never its ``__call__``, and reports ``False`` for it
    regardless. Found by Codex review before this ever shipped."""
    if inspect.iscoroutinefunction(func):
        return True
    call = getattr(func, "__call__", None)  # noqa: B004 -- need the __call__ object itself, not a bool
    return call is not None and inspect.iscoroutinefunction(call)


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

    ``notify`` MUST be an ``async def`` -- calling one only constructs a
    coroutine object, running none of its body yet, which is what makes
    calling it directly here always cheap regardless of what the
    coroutine goes on to do. A plain synchronous factory that performs
    real, blocking work before ever returning its awaitable runs that
    work directly on the event loop's own thread the first time this
    channel notifies, freezing every other coroutine sharing it, with
    neither ``notify_timeout`` nor cancellation able to help. A
    non-coroutine-function ``notify`` triggers a one-time warning at
    construction for exactly this reason.

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
        wait_budget: timedelta | None = None,
        notify: Callable[[ApprovalTicket, str], Awaitable[None]] | None = None,
        notify_timeout: float = 10.0,
        renotify_interval: timedelta | None = DEFAULT_RENOTIFY_INTERVAL,
        kind: TicketKind = "approval",
        operator_only: bool = False,
    ):
        if not poll_seconds > 0:
            # Written as `not x > 0`, not `x <= 0`: NaN compares False
            # against BOTH, so `poll_seconds <= 0` lets float("nan")
            # straight through, unlike this phrasing (`not (nan > 0)` is
            # `not False` -- True, correctly rejected). A poll interval
            # that isn't verifiably positive means
            # min(poll_seconds, remaining_ttl) resolves to
            # asyncio.sleep(0-or-negative-or-NaN) every iteration -- a
            # NaN sleep may never wake at all, and a nonpositive one is a
            # hot loop of Store reads (and, for a file-backed queue, a
            # fresh SQLite connection opened and closed every single
            # iteration) hammering the database until the ticket's TTL
            # (up to hours) finally expires. Found by Codex review before
            # this ever shipped.
            raise ValueError(f"poll_seconds must be positive, got {poll_seconds}")
        if ttl <= timedelta(0):
            # ApprovalQueue.create_ticket() rejects this too, but only
            # once ask() actually calls it -- validating here as well
            # gives construction-time feedback, matching every other
            # timing parameter this class already checks up front. A
            # nonpositive ttl writes a ticket that's immediately invisible
            # to list_pending_tickets() and unapprovable/unrejectable, yet
            # still durably stored as "pending" forever. Found by Codex
            # review before this ever shipped.
            raise ValueError(f"ttl must be positive, got {ttl}")
        if wait_budget is not None and wait_budget <= timedelta(0):
            # Same reasoning as ttl's own check above: a nonpositive
            # wait_budget would make ask() give up on its very first poll,
            # every time, for a channel that otherwise looks correctly
            # configured.
            raise ValueError(f"wait_budget must be positive, got {wait_budget}")
        if renotify_interval is not None and renotify_interval <= timedelta(0):
            # `None` is already the documented way to disable reminders --
            # zero/negative is not a smaller version of that, it's
            # "always overdue": now - last_notified >= renotify_interval
            # is true on EVERY poll, so a reminder (a full _send call,
            # notify included) fires at the full polling rate --
            # ~100/s at poll_seconds=0.01 -- for up to the ticket's whole
            # TTL. Found by Codex review before this ever shipped.
            raise ValueError(f"renotify_interval must be positive or None, got {renotify_interval}")
        if not notify_timeout > 0:
            # `not x > 0`, not `x <= 0` -- see poll_seconds' own check
            # above for why (NaN slips past `<= 0` but not this). A
            # nonpositive timeout hands `_send()` an immediate deadline:
            # the notifier gets cancelled at its very first suspension
            # point (typically a network call), before it can ever
            # deliver the one message that tells a human this ticket
            # exists. ask() then just keeps polling, unnoticed, for up to
            # the ticket's full TTL -- hours by default. Found by Codex
            # review before this ever shipped.
            raise ValueError(f"notify_timeout must be positive, got {notify_timeout}")
        if notify is not None and not _is_async_callable(notify):
            # A loud, one-time warning at construction -- where a human
            # actually reads logs -- rather than silently guessing at
            # call time (`_send()` calls `notify` directly, trusting this
            # contract). `notify` must be an `async def`: calling one
            # only constructs a coroutine object, running none of its
            # body yet, which is what keeps evaluating it here always
            # cheap regardless of what the coroutine goes on to do. A
            # plain synchronous factory that performs real, blocking work
            # before returning an awaitable would run that work directly
            # on the event loop's own thread the first time `ask()`
            # notifies, freezing every other coroutine sharing it for as
            # long as it takes -- with neither notify_timeout nor
            # cancellation able to help, since nothing has reached an
            # await yet. (Offloading such a call to a worker thread was
            # tried and reverted: it broke an equally valid pattern --a
            # synchronous factory that legitimately needs the RUNNING
            # loop to build its result, e.g.
            # `asyncio.get_event_loop().create_future()` -- with
            # `RuntimeError: no running event loop`, confirmed by a
            # standalone repro. There is no way to tell the two apart via
            # introspection, so this stays a documented caller
            # responsibility instead of an auto-fix.) Found by Codex
            # review before this ever shipped.
            logging.getLogger(__name__).warning(
                "StoreApprovalChannel's notify=%r is not an `async def` -- if it performs blocking work "
                "before returning its awaitable, that work will run directly on the event loop and freeze "
                "every other coroutine sharing it. Prefer an async def notify(ticket, message): ...",
                notify,
            )
        self._queue = queue
        self._task_id = task_id
        self._poll_seconds = poll_seconds
        self._ttl = ttl
        # Never longer than the ticket is alive -- waiting past expiry would
        # just be waiting for a corpse. `None` (the default) reproduces this
        # module's own pre-split behaviour exactly: wait the full ttl.
        self._wait_budget = ttl if wait_budget is None else min(wait_budget, ttl)
        self._notify = notify
        self._notify_timeout = notify_timeout
        self._renotify_interval = renotify_interval
        #: Stamped on every ticket this instance creates -- "escalation" for
        #: an agent's own channel, "approval" for a default TieredGate "ask"
        #: gate. See TicketKind's docstring.
        self._kind = kind
        #: Stamped on every ticket this instance creates -- see
        #: ApprovalTicket.operator_only's own docstring. Pure metadata as
        #: far as this class is concerned; it enforces nothing about it
        #: itself.
        self._operator_only = operator_only
        #: The most recent detached notify task `_send()` created for each
        #: approval_id, if any is still running -- lets `_send()` refuse
        #: to start a SECOND one for the SAME ticket while a permanently
        #: uncooperative notifier is still stuck in the first. Keyed by
        #: ticket, not shared across the whole channel: one channel
        #: instance can file more than one ticket over its lifetime (the
        #: class docstring's "every ticket this channel files" is
        #: deliberately plural -- TieredGate can call ask() repeatedly, or
        #: concurrently, on the same channel), and a stuck notify for ONE
        #: of them must not silently swallow every notification for an
        #: unrelated other. See `_send()`'s own comment for why the guard
        #: itself matters. Found by Codex review before this ever shipped.
        #:
        #: Known accepted gap: this bounds a stuck notifier to ONE task
        #: per ticket, not per channel -- a notifier that never
        #: cooperates with cancellation, applied to every ticket this
        #: channel EVER files over an unboundedly long lifetime, still
        #: accumulates one permanently-stuck task per ticket, without an
        #: upper bound on how many tickets that can be. A circuit breaker
        #: (stop calling a notifier that's proven itself permanently
        #: uncooperative after some threshold, rather than retrying it
        #: fresh for every new ticket) would close this properly but is a
        #: real feature, not a narrow fix -- left as a caller
        #: responsibility (fix or replace a notifier discovered to behave
        #: this way) rather than built here. Found by Codex review before
        #: this ever shipped.
        self._notify_tasks: dict[str, asyncio.Task[None]] = {}

    async def _call(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run one of ``self._queue``'s plain synchronous methods without
        blocking THIS event loop -- offloaded to a worker thread, unless
        the queue itself says that's unsafe (``Store(db=":memory:")``,
        which relies on reusing one specific thread's own thread-local
        SQLite connection; ``asyncio.to_thread`` would instead hand each
        call to a different pool thread, each opening its OWN unrelated
        empty in-memory database). Found by Codex review before this ever
        shipped."""
        if self._queue.safe_to_call_from_any_thread:
            return await asyncio.to_thread(func, *args, **kwargs)
        return func(*args, **kwargs)

    async def _send(self, ticket: ApprovalTicket, message: str, *, deadline: datetime | None = None) -> None:
        """``deadline``, when given, REPLACES ``ticket.expires_at`` as what
        this send's timeout is measured against (``self._notify_timeout``
        itself still always applies on top, via the ``min()`` below) --
        used for the FIRST notification and for a renotify reminder, both
        sent while a wait is still genuinely in progress, to also respect
        ``StoreApprovalChannel``'s own ``wait_budget`` (not just
        ``ticket.expires_at``/``ttl``): without it, a short ``wait_budget``
        paired with a much longer ``ttl`` and a stalled notifier let the
        very first send alone block for up to ``notify_timeout`` (10s by
        default) before ``ask()`` ever reached its own polling loop,
        overshooting a ``wait_budget`` that could be milliseconds. Found
        by Codex review before this ever shipped.

        Both terminal notices -- "stopped waiting" (``wait_budget``
        elapsed, ticket still live) and "expired" (``ttl`` elapsed) --
        pass their own explicit ``deadline`` (``now + notify_timeout``)
        rather than relying on the default ``ticket.expires_at``-based
        fallback below. Two different failure modes motivate this for the
        two calls: for "expired", ``ticket.expires_at`` has ALREADY
        passed, so the default would compute a NEGATIVE remaining time,
        floor straight to a ``0.0`` timeout, and cancel the notifier
        before it could ever deliver the message explaining the expiry.
        For "stopped waiting", ``ticket.expires_at`` is usually still far
        in the FUTURE (``ttl`` is typically much longer than
        ``wait_budget``), so the default would instead let a stalled
        notifier cost this already-giving-up call up to a full extra
        ``notify_timeout`` on top of a ``wait_budget`` that may be
        milliseconds -- exactly the inflation ``wait_budget`` exists to
        cap. Found by Codex review before this ever shipped (two rounds:
        the first fix only caught the "expired" case).
        """
        if self._notify is None:
            return
        existing = self._notify_tasks.get(ticket.approval_id)
        if existing is not None and not existing.done():
            # A previous notify call FOR THIS TICKET (the original one at
            # creation, or an earlier reminder) is STILL running -- a
            # permanently uncooperative notifier that never responds to
            # cancellation, not just a slow one (a slow-but-eventually-
            # cancellable one is already `.done()` -- cancelled -- by the
            # time any later call gets here). Starting ANOTHER detached
            # notify task on top of it would mean every renotify interval
            # piles up its own forever-running task (and whatever
            # resources the notifier itself holds -- a socket, a thread,
            # anything) without bound over a long-lived ticket. Scoped per
            # ticket (approval_id), not to the whole channel: one channel
            # instance can file more than one ticket over its life (see
            # this class' own docstring), and a stuck notify for one
            # ticket must not silently swallow notifications for an
            # unrelated other one this same channel is also handling.
            # Found by Codex review before this ever shipped.
            logging.getLogger(__name__).warning(
                "skipping notify for ticket %s -- a previous notify call is still stuck", ticket.approval_id
            )
            return
        # Bounded by whichever is SHORTER: notify_timeout, or the ticket's
        # own remaining lifetime. notify_timeout alone (10s default) can
        # still make ask() overshoot a short ttl by nearly that whole
        # amount whenever a notify call stalls -- e.g. ttl=50ms with a
        # hanging notifier would return ~10s late instead of ~50ms late,
        # contradicting the timeout this channel documents. Found by
        # Codex review before this ever shipped.
        now = datetime.now(UTC)
        # `deadline`, when given, REPLACES the ticket.expires_at-based cap
        # entirely rather than being combined with it via min(): the one
        # caller that needs this (an already-past-expiry terminal notice)
        # needs that past expires_at taken OUT of the computation, not
        # min'd against (min() with an already-past timestamp would still
        # floor straight to 0.0 regardless of how far in the future
        # `deadline` itself is). `StoreApprovalChannel.ask()`'s own
        # `wait_budget` is always <= `ttl` by construction (see __init__),
        # so a `deadline=stop_waiting_at` passed for a send that's still
        # genuinely in progress is already the tighter of the two without
        # needing to also compare it against ticket.expires_at here. Found
        # by Codex review before this ever shipped.
        cap_at = deadline if deadline is not None else ticket.expires_at
        remaining = (cap_at - now).total_seconds()
        timeout = min(self._notify_timeout, max(remaining, 0.0))

        # A bounded wait, not just a try/except: a notify callback that
        # hangs (a stalled network transport, e.g.) would otherwise block
        # `ask()` from EVER reaching its own status-polling loop below --
        # so even a ticket approved through another surface a second later
        # would sit unnoticed for as long as the notify call never
        # returns. A timeout turns that into an ordinary swallowed notify
        # failure instead. Found by Codex review before this ever shipped.
        #
        # Shielded, with the cancel requested but never awaited: plain
        # `asyncio.wait_for(self._notify(...), timeout=timeout)` cancels
        # the notify coroutine on timeout and then WAITS for it to finish
        # cancelling before raising -- a notifier that catches
        # `CancelledError` to run its own cleanup or retry logic can make
        # that wait, and therefore this supposedly-bounded timeout, take
        # arbitrarily long. Shielding decouples the two: `wait_for` only
        # ever waits on the (plain, instantly-cancellable) shield wrapper,
        # so IT returns right on schedule regardless of how the real
        # notify task behaves; requesting its cancellation separately,
        # without awaiting it, is what keeps a slow-to-cancel notifier
        # from blocking `_send()` itself. Found by Codex review before
        # this ever shipped.
        def _detach(notify_task: asyncio.Task[None]) -> None:
            # Requests cancellation and walks away -- never awaited, so a
            # notifier that ignores/absorbs it can't block whoever called
            # this. Tracked in `_background_tasks` (module-level, shared
            # with `ask()`'s own detached cleanup tasks) purely so nothing
            # garbage-collects it mid-flight; retiring it is `_log_if_failed`
            # below's job, not this function's.
            notify_task.cancel()
            _background_tasks.add(notify_task)

            def _log_if_failed(t: asyncio.Task[None]) -> None:
                _background_tasks.discard(t)
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    # A notifier that ignores this cancellation and later
                    # raises something else entirely (not the
                    # CancelledError it was asked for) would otherwise
                    # surface as asyncio's own "Task exception was never
                    # retrieved" -- noisy, and NOT the deliberate,
                    # swallowed-and-logged treatment every other notify
                    # failure in this method gets. Retrieving it here
                    # (`t.exception()`) and logging it the same way closes
                    # that gap. Found by Codex review before this ever
                    # shipped.
                    logging.getLogger(__name__).error(
                        "notify failed for ticket %s -- it still exists and is still answerable",
                        ticket.approval_id,
                        exc_info=exc,
                    )
                    self._queue.record_notify_failure(
                        source=f"StoreApprovalChannel:{ticket.kind}:detached",
                        text=message,
                        error=repr(exc),
                    )

            notify_task.add_done_callback(_log_if_failed)

        current = asyncio.current_task()
        # Snapshotted before ANYTHING else in this method, `self._notify`
        # not even called yet -- not checked as a bare
        # `current.cancelling() == 0` after catching -- `cancelling()` is
        # a cumulative, never-auto-reset count of every `.cancel()` this
        # task has EVER received that hasn't been matched by an
        # `.uncancel()`. A task that swallowed some earlier, unrelated
        # cancellation elsewhere (its own cleanup path, e.g.) keeps a
        # nonzero count for the rest of its life without ever having
        # called `uncancel()` -- checking the raw count would then treat
        # every later notifier self-cancellation on that SAME task as if
        # it were a fresh external cancellation, rejecting a perfectly
        # fine ticket. Comparing the count from right here to the count
        # right after only flags a NEW request that landed somewhere in
        # this whole method -- `.cancel()` increments the count
        # synchronously, before the resulting CancelledError is ever
        # delivered, so a genuine cancellation is guaranteed to show up as
        # an increase regardless of WHERE in this method it landed.
        # Snapshotting any later than this (after constructing the notify
        # awaitable, say) would miss a `notify` callable unusual enough to
        # cancel this very task as a side effect of merely being called,
        # before ever returning its own coroutine. Found by Codex review
        # before this ever shipped (three times over: the first fix
        # checked `notify_task.cancelled()` alone, missing that OUR OWN
        # task's cancellation can race notify_task's; the second checked
        # the raw count instead of the delta, missing the stale-count case
        # above; the third snapshotted too late, after `self._notify` had
        # already run).
        cancelling_before = current.cancelling() if current is not None else 0

        def _is_a_fresh_cancellation_of_this_task() -> bool:
            # True only if a NEW `.cancel()` landed on THIS task somewhere
            # between the snapshot above and right now -- see that
            # snapshot's own comment for why a delta, not a raw
            # `cancelling() == 0` check, is what's needed here.
            cancelling_after = current.cancelling() if current is not None else 0
            return cancelling_after != cancelling_before

        def _swallow_notifier_cancellation() -> None:
            logging.getLogger(__name__).warning(
                "notify was cancelled (from within the notifier itself) for ticket %s -- "
                "it still exists and is still answerable",
                ticket.approval_id,
            )

        try:
            try:
                # Constructing the awaitable is inside its own try, not
                # just awaiting it: a `notify` that raises SYNCHRONOUSLY
                # when called (before ever returning a coroutine) must be
                # swallowed and logged the same as any other notify
                # failure, not left to propagate out of `_send()` (and,
                # from there, out of ask() itself, rejecting a ticket over
                # a notify problem rather than a real approval one). Found
                # by Codex review before this ever shipped.
                #
                # `notify` is documented (see the class docstring) as
                # required to be an `async def` for exactly this reason:
                # calling one only constructs a coroutine object, running
                # none of its body yet, so evaluating it here is always
                # cheap and safe regardless of what the coroutine goes on
                # to do. A plain (non-async-def) factory that performs
                # real, blocking work before ever returning an awaitable
                # would run that work on THIS event loop's own thread,
                # freezing every other coroutine sharing it -- but
                # offloading the call to a worker thread to guard against
                # that (an earlier version of this fix did exactly that)
                # is NOT safe in general either: a factory that
                # legitimately needs the running loop to construct its
                # result (`asyncio.get_event_loop().create_future()`, or
                # `asyncio.create_task(...)`, e.g.) would then fail with
                # `RuntimeError: no running event loop` in the worker
                # thread, confirmed with a standalone repro -- silently
                # turning a WORKING synchronous factory into a broken one.
                # There is no way to distinguish "safe, loop-bound, fast"
                # from "unsafe, blocking" via introspection alone, so
                # `__init__` warns instead (once, at construction, where a
                # human will actually see it) rather than guessing wrong
                # here on every call. Found by Codex review before this
                # ever shipped (twice: once for the original missing
                # protection, and again for this first fix attempt's own
                # regression).
                notify_awaitable = self._notify(ticket, message)
                notify_task = asyncio.ensure_future(notify_awaitable)
            except asyncio.CancelledError:
                # No `notify_task` exists yet here -- calling `self._notify`
                # is plain, synchronous Python with no `await` in it, so
                # asyncio's OWN cancellation-delivery machinery (which only
                # ever injects a CancelledError at an actual suspension
                # point) cannot be the source of one raised from evaluating
                # it. The delta check still guards this rather than
                # assuming that invariant always holds -- e.g. a `notify`
                # factory that itself awaits something internally before
                # this expression finishes evaluating. Found by Codex
                # review before this ever shipped: the previous fix only
                # covered a self-cancelling notify TASK, missing that the
                # factory call constructing it can raise the exact same
                # way before one ever exists to inspect.
                if _is_a_fresh_cancellation_of_this_task():
                    raise
                _swallow_notifier_cancellation()
                return
            self._notify_tasks[ticket.approval_id] = notify_task

            def _forget_if_still_current(_: asyncio.Task[None], approval_id: str = ticket.approval_id) -> None:
                # Only when THIS task is still the one on record for this
                # approval_id -- otherwise a later call for the same
                # ticket may already have replaced it, and this stale
                # completion must not evict that newer entry.
                if self._notify_tasks.get(approval_id) is notify_task:
                    self._notify_tasks.pop(approval_id, None)

            notify_task.add_done_callback(_forget_if_still_current)
            try:
                await asyncio.wait_for(asyncio.shield(notify_task), timeout=timeout)
            except asyncio.CancelledError:
                if not _is_a_fresh_cancellation_of_this_task() and notify_task.cancelled():
                    # No NEW cancellation landed on THIS task during this
                    # specific await -- so this CancelledError can only be
                    # notify_task's own doing (something inside the
                    # notifier's own implementation: an inner
                    # asyncio.wait_for, a transport task it manages and
                    # cancels itself, etc.), not `_send()`/ask() being
                    # torn down. Treat it like any other notify failure:
                    # logged and swallowed, not propagated to abort ask()
                    # and retire an otherwise-fine ticket over a
                    # notifier-internal hiccup.
                    _swallow_notifier_cancellation()
                    return
                # A NEW cancellation landed on `_send()` itself during this
                # await (ask() is being torn down) -- shield() keeps
                # `notify_task` running through that regardless of
                # whatever ELSE notify_task may be doing, so without
                # detaching it here it would run forever with nothing left
                # to ever cancel or retire it (the timeout machinery this
                # whole method exists for disappears along with `_send()`
                # unwinding).
                _detach(notify_task)
                raise
            except TimeoutError:
                _detach(notify_task)
                logging.getLogger(__name__).warning(
                    "notify timed out after %.1fs for ticket %s -- it still exists and is still answerable",
                    timeout,
                    ticket.approval_id,
                )
                self._queue.record_notify_failure(
                    source=f"StoreApprovalChannel:{ticket.kind}",
                    text=message,
                    error=f"timed out after {timeout:.1f}s",
                )
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "notify failed for ticket %s -- it still exists and is still answerable", ticket.approval_id
            )
            self._queue.record_notify_failure(
                source=f"StoreApprovalChannel:{ticket.kind}", text=message, error=repr(exc)
            )

    def _record_refusal(self, ticket: ApprovalTicket, *, reason: str, responder: str) -> None:
        """A refusal THIS CHANNEL knows the true reason for -- rejected by
        an actual human, or expired with nobody ever answering.

        The gate-level recorder (``queue.record_denial`` wired as
        :class:`~lazybridge.ext.approval.tiered.TieredGate`'s own
        ``on_record``) only ever sees TieredGate's generic ``"'...' denied
        by the human approver"`` sentence -- the same text whether a human
        actually declined or a deadline simply passed with nobody looking.
        This records the version where that distinction is still known,
        through the SAME :meth:`ApprovalQueue.record_denial` sink so both
        audit paths land in one place. Synchronous and direct (not through
        ``self._call``'s offload): this runs once, on ``ask()``'s own
        terminal path, not in the hot polling loop, the same trade-off the
        final cleanup at the bottom of ``ask()`` already makes.
        """
        self._queue.record_denial(
            SimpleNamespace(
                action="deny",
                tool_name=_tool_name_from_prompt(ticket.prompt) or ticket.kind,
                kind="approval-channel",
                tier="ask",
                message=reason,
                responder=responder,
                cwd="",
                arguments_preview=ticket_gist(ticket.prompt),
            )
        )

    async def ask(self, prompt: str) -> bool:
        # An answer a human already gave after an EARLIER waiter had
        # already given up on it (see wait_budget below, and
        # ApprovalQueue.claim_earlier_approval's own docstring) -- checked
        # BEFORE filing a new ticket, or the same question gets asked again
        # while the answer to it already sits unread in the queue.
        #
        # Shielded, like the ticket-creation call below: claim_earlier_approval
        # does a CAS that SPENDS an existing approval (sets consumed_at) --
        # the underlying thread-pool work can't actually be stopped once
        # started, so an unshielded await that unwinds on cancellation would
        # let that CAS land moments later with nobody ever told whether it
        # won, silently spending a human's "yes" for a call that never
        # itself returned True to anyone. Waiting out the cancellation here
        # (rather than detaching, the way a fresh ticket's own orphaned
        # creation is retired) is deliberate: a claim either wins or
        # doesn't, there is no record to roll back the way an orphaned
        # ticket is rejected -- the outcome is simply resolved before this
        # cancellation becomes observable, not undone. Found by Codex
        # review before this ever shipped.
        claim = asyncio.ensure_future(
            self._call(
                self._queue.claim_earlier_approval,
                task_id=self._task_id,
                prompt=prompt,
                operator_only=self._operator_only,
            )
        )
        try:
            claimed = await asyncio.shield(claim)
        except asyncio.CancelledError:
            while not claim.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(claim)
            raise
        if claimed:
            return True

        # ApprovalQueue's own methods are plain synchronous Store I/O (by
        # design -- usable from a sync caller too, e.g. a CLI or a plain
        # webhook handler), including, for a file-backed queue, opening a
        # fresh SQLite connection per call. Called directly from this
        # ASYNC method, a slow write (Store's busy_timeout is 5s under
        # write contention -- a shared Store with many agents filing
        # tickets/heartbeats is exactly what produces that) would block
        # THIS WHOLE asyncio event loop, freezing every other coroutine
        # sharing it (unrelated agents, other approval channels, anything
        # else) for as long as the call takes -- not just this one
        # ticket's own progress. Offloaded via self._call() throughout
        # this method instead. Found by Codex review before this ever
        # shipped.
        #
        # Shielded: the offloaded call's underlying thread-pool work can't
        # actually be stopped once started (Python threads aren't forcibly
        # killable) -- if ask() is cancelled while THIS specific call is
        # pending, an unshielded await would unwind immediately with no
        # ticket reference to clean up, while the write still lands moments
        # later in the Store, orphaned, with nothing ever retiring it.
        # Shielding keeps the creation itself running regardless of the
        # outer cancellation. Found by Codex review before this ever
        # shipped.
        create = asyncio.ensure_future(
            self._call(
                self._queue.create_ticket,
                task_id=self._task_id,
                prompt=prompt,
                kind=self._kind,
                ttl=self._ttl,
                operator_only=self._operator_only,
            )
        )

        async def _retire_orphaned_ticket() -> None:
            # Scheduled from the `except` block below as an entirely
            # separate task -- ask() DOES end up waiting for this one
            # (unlike the post-creation cleanup, which no longer needs to
            # wait for anything), but shielded and in a retry loop rather
            # than a single inline `await`, so no number of FURTHER
            # cancellations delivered to ask() while it waits can
            # interrupt this task's OWN execution -- only each individual
            # wait for it. Only ever created on the cancellation path:
            # ask()'s own normal, non-cancelled flow never spawns this, so
            # it can never race a legitimate resolution for a ticket that
            # was never orphaned in the first place.
            #
            # Two earlier versions got this wrong in opposite directions.
            # The first re-awaited `create` INLINE, in ask()'s own
            # `except asyncio.CancelledError:` handler, nesting a second
            # `asyncio.shield()` around that wait -- shield() only
            # protects the awaited FUTURE from being cancelled, never the
            # coroutine doing the awaiting from being cancelled AGAIN, so
            # a second `.cancel()` landing while that inline cleanup was
            # itself suspended on the second shield blew straight through
            # it (confirmed with a standalone repro). The second detached
            # this task correctly, for that reason, but then re-raised
            # ask()'s own cancellation IMMEDIATELY, before this task was
            # necessarily done -- letting another surface approve the
            # ticket the instant `create`'s write landed, winning the CAS
            # before this task got to reject it, for a decision nobody was
            # listening for anymore (confirmed as a real gap by Codex
            # review). ask()'s own `except` block now loops, shielded,
            # re-waiting on this task after each further cancellation
            # instead of giving up after one -- getting immunity to
            # repeated cancellation AND a guarantee that retirement
            # finishes before ask()'s cancellation becomes observable, at
            # once.
            #
            # Still not immune to the event loop itself shutting down
            # (which cancels this task too, before `create`'s underlying
            # thread is guaranteed done) -- see this module's docstring for
            # why that narrower, compound case is an accepted gap rather
            # than something this task can be made to survive.
            try:
                ticket = await asyncio.shield(create)
            except Exception:
                return
            with contextlib.suppress(Exception):
                await self._call(
                    self._queue.reject_ticket,
                    ticket.approval_id,
                    actor="system",
                    channel=self.name,
                    reason="ask() was cancelled during ticket creation",
                )

        try:
            ticket = await asyncio.shield(create)
        except asyncio.CancelledError:
            # Retirement runs on an independent task -- not awaited
            # inline, which is what an earlier, insufficient fix did --
            # so it's immune to however many more times ask() gets
            # cancelled from here on: cancelling ask()'s OWN task never
            # touches a separate task's execution, only whatever ask() is
            # currently suspended on.
            watcher = asyncio.ensure_future(_retire_orphaned_ticket())
            _background_tasks.add(watcher)
            watcher.add_done_callback(_background_tasks.discard)
            # But immunity to repeated cancellation isn't the only
            # requirement: re-raising immediately here (an earlier version
            # did exactly that) makes THIS cancellation observable to
            # ask()'s own caller before `watcher` has necessarily finished
            # retiring the ticket -- once `create`'s write actually lands,
            # another surface can approve the newly-visible ticket in that
            # gap, winning the CAS before `watcher` gets to reject it, for
            # a decision nobody is listening for anymore (confirmed as a
            # real gap by Codex review; the post-creation cleanup below
            # had the identical bug before it was made synchronous).
            # Waiting for `watcher` here -- shielded, in a loop that keeps
            # re-waiting after each further cancellation rather than
            # giving up after one -- gets BOTH properties at once: no
            # number of repeated cancellations can make this loop exit
            # before `watcher` is actually done (shield() only lets a
            # cancellation of ASK's own task interrupt each individual
            # wait, never `watcher` itself, which nothing here ever
            # cancels), and this cancellation only becomes observable to
            # ask()'s caller once retirement has been ATTEMPTED and fully
            # resolved -- not once it has necessarily WON. `watcher`'s own
            # reject_ticket call is still an ordinary CAS that can lose to
            # a genuinely concurrent external approve reaching the Store
            # first (the module docstring's own "Known accepted gap"
            # documents this residual, irreducible race: it takes an
            # external actor already polling and racing to approve this
            # SAME ticket the instant it becomes visible, which no amount
            # of code inside ask() itself can prevent -- only shrink the
            # window for, which this already does about as tightly as
            # possible by attempting retirement as the very next thing
            # once creation completes, with nothing else interposed).
            # Structurally the same technique asyncio.wait_for() itself
            # uses to survive a cancellation racing its own timeout.
            while not watcher.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(watcher)
            raise
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
        # How long THIS call waits before giving up -- never past the
        # ticket's own expires_at (self._wait_budget is already clamped to
        # self._ttl at construction, but a ticket's actual expires_at is
        # ticket.created_at + self._ttl computed independently by
        # create_ticket, so this is derived from the ticket itself rather
        # than re-deriving it from self._ttl a second time).
        stop_waiting_at = ticket.created_at + self._wait_budget
        try:
            await self._send(ticket, message, deadline=stop_waiting_at)
            last_notified = datetime.now(UTC)
            while True:
                current = await self._call(self._queue.get_ticket, ticket.approval_id)
                if current is None or current.status in ("rejected", "expired"):
                    # "expired" reaches here when some OTHER caller (an
                    # operator tool, a second concurrent poller) calls
                    # expire_ticket() directly -- expire_ticket itself does
                    # not check elapsed time, only CAS's pending -> expired,
                    # so this can happen before THIS loop's own
                    # current.expires_at <= now check below would ever fire
                    # on its own. Without this branch, an externally-expired
                    # ticket fell through to that later check, which stayed
                    # false (not actually time-expired yet), and this loop
                    # kept polling -- and even re-notifying -- an already
                    # terminal ticket indefinitely. Found by Codex review
                    # before this ever shipped.
                    #
                    # Recorded HERE, where the difference is still known --
                    # by the time a plain "denied by the human approver" (if
                    # this channel sits behind TieredGate) reaches the
                    # gate-level recorder it is the same sentence for a
                    # rejection and for a timeout.
                    if current is not None and current.status == "expired":
                        reason = "expired" + (f": {current.reason}" if current.reason else "")
                        responder = "nobody"
                    else:
                        actor = (current.actor if current is not None else None) or "unknown"
                        reason = f"rejected by {actor}" + (
                            f": {current.reason}" if current is not None and current.reason else ""
                        )
                        responder = actor
                    self._record_refusal(current or ticket, reason=reason, responder=responder)
                    return False
                if current.status == "approved":
                    # Spent now, not left for a later claim_earlier_approval
                    # to find still "approved" and unconsumed -- otherwise a
                    # later retry with the IDENTICAL prompt (same task_id)
                    # could silently re-claim and re-run THIS SAME approval
                    # a second time, off one human decision.
                    #
                    # The return value matters, not just the side effect:
                    # TWO CONCURRENT ask() calls for the identical
                    # (task_id, prompt) each create their OWN ticket (tickets
                    # are never deduplicated by content), and either one's
                    # ticket getting approved makes claim_earlier_approval
                    # claimable by the OTHER call too. If this call's own
                    # mark_approval_consumed loses that race (the other call's
                    # claim_earlier_approval won it first), returning True
                    # here regardless would let BOTH calls report success for
                    # the SAME single human decision -- two executions from
                    # one "yes". Only the caller that actually wins the
                    # consumption may report True; losing it is treated the
                    # same as never having been approved at all, for THIS
                    # call. Found by Codex review before this ever shipped
                    # (two rounds: the first fix recorded consumption but
                    # ignored whether it actually won).
                    return await self._call(self._queue.mark_approval_consumed, current.approval_id)
                now = datetime.now(UTC)
                if current.expires_at > now >= stop_waiting_at:
                    # THIS CALLER stops waiting; the REQUEST does not die.
                    # Nothing is expired and nothing is recorded as refused
                    # here, because nobody refused anything -- the ticket
                    # stays pending, listed, and answerable, and a LATER
                    # call to ask() with the identical prompt picks up an
                    # answer given after this point via
                    # claim_earlier_approval, instead of asking the same
                    # question again into a void.
                    waited = int((now - ticket.created_at).total_seconds() // 60)
                    await self._send(
                        current,
                        f"⏸️ Stopped waiting after {waited} min, so this did NOT run. The request is still "
                        f"open and still answerable -- an answer given now still counts and can be picked up "
                        f"on a later attempt.\n\n{ticket_gist(ticket.prompt)}",
                        # An explicit, bounded deadline -- NOT the default
                        # fallback to ticket.expires_at, which can still be
                        # arbitrarily far in the future (ttl is usually much
                        # longer than wait_budget). Falling back to it here
                        # would let a stalled notifier cost this already-
                        # giving-up call up to a full extra notify_timeout on
                        # top of a wait_budget that may be milliseconds --
                        # exactly the inflation wait_budget exists to put a
                        # ceiling on. Found by Codex review before this ever
                        # shipped.
                        deadline=now + timedelta(seconds=self._notify_timeout),
                    )
                    return False
                if current.expires_at <= now:
                    # Giving up silently would make a timeout
                    # indistinguishable, from the human's side, from a
                    # message they simply never opened, and indistinguishable,
                    # from this channel's side, from a human saying no.
                    if not await self._call(self._queue.expire_ticket, current.approval_id):
                        # Lost a race on the deadline: someone answered
                        # between the read above and this write. Their
                        # answer is the real one -- ignoring the CAS result
                        # here would tell a human who just approved that
                        # their action "did NOT run", and leave the audit
                        # trail saying approved while nothing happened.
                        #
                        # Spent here too, exactly like the ordinary approved
                        # path above -- returning True directly, without
                        # consuming, left this ticket "approved" and
                        # unconsumed, the precise shape claim_earlier_approval
                        # looks for, so a LATER identical ask() could
                        # silently re-claim and re-run this SAME approval a
                        # second time. The return value is the CAS outcome,
                        # not a bare "was it approved": a lost consumption
                        # race here (a concurrent caller's own
                        # claim_earlier_approval winning it first) means
                        # someone else already claimed it, so THIS call must
                        # not also report success -- same reasoning as the
                        # ordinary approved path. Found by Codex review
                        # before this ever shipped.
                        settled = await self._call(self._queue.get_ticket, current.approval_id)
                        if settled is not None and settled.status == "approved":
                            return await self._call(self._queue.mark_approval_consumed, settled.approval_id)
                        return False
                    waited = int((now - ticket.created_at).total_seconds() // 60)
                    self._record_refusal(
                        current, reason=f"expired: nobody answered in {waited} min", responder="nobody"
                    )
                    await self._send(
                        current,
                        f"⌛ No answer in {waited} min, so this did NOT run. Nothing is waiting on you now; "
                        f"ask again if it should still happen.\n\n{ticket_gist(ticket.prompt)}",
                        # An explicit FUTURE deadline, not the default
                        # fallback to ticket.expires_at: this send runs
                        # AFTER the ticket's own expires_at has already
                        # passed (that's why we're here), so the default
                        # cap would compute a NEGATIVE remaining time,
                        # floor to a 0.0 timeout, and cancel the notifier
                        # before it could ever deliver the one message
                        # explaining the expiry. Found by Codex review
                        # before this ever shipped.
                        deadline=now + timedelta(seconds=self._notify_timeout),
                    )
                    return False
                if self._renotify_interval is not None and now - last_notified >= self._renotify_interval:
                    waited_minutes = int((now - ticket.created_at).total_seconds() // 60)
                    reminder = f"⏰ Still waiting on this one ({waited_minutes} min) -- {message}"
                    await self._send(current, reminder, deadline=stop_waiting_at)
                    # A FRESH timestamp, not the pre-send `now` above: a
                    # reminder notify can itself take real time (bounded
                    # by notify_timeout, which can be a meaningful
                    # fraction of a short renotify_interval), and stamping
                    # the pre-send time here would let that delivery time
                    # eat into the NEXT interval too -- a 50ms interval
                    # with a 40ms notifier would collapse to ~10ms between
                    # completed reminders instead of the configured
                    # spacing. Found by Codex review before this ever
                    # shipped.
                    last_notified = datetime.now(UTC)
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
                # Also capped at stop_waiting_at, not just expires_at: without
                # this a long poll_seconds (or a short wait_budget well inside
                # a much longer ttl) could sleep straight past the point this
                # call is meant to give up, overshooting it by up to a whole
                # poll interval before the check above ever runs again.
                seconds_until_deadline = (min(current.expires_at, stop_waiting_at) - now).total_seconds()
                await asyncio.sleep(min(self._poll_seconds, max(seconds_until_deadline, 0.0)))
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
            # Called directly here, NOT through self._call()'s offload,
            # and NOT detached onto its own task the way the creation-path
            # cleanup above has to be -- ticket already exists by this
            # point, so unlike that one, there's nothing left to wait on
            # asynchronously before this can run. A synchronous call has
            # no `await` for a repeated `.cancel()` to land in, so it's
            # immune to however many more times ask() gets cancelled from
            # here on for the same reason an async detached task is --
            # but WITHOUT that approach's own race: a detached task lets
            # ask()'s own CancelledError reach its caller BEFORE the
            # reject_ticket CAS actually runs, so another surface can
            # approve the still-"pending" ticket in that window --
            # succeeding, silently, for a decision nobody is listening for
            # anymore (confirmed as a real gap by Codex review; the
            # earlier detached version of this cleanup had it). Calling it
            # synchronously guarantees the CAS has already resolved one
            # way or the other by the time `raise` below makes this
            # cancellation observable to anyone. The trade-off is a rare,
            # bounded block of the event loop for the duration of one
            # Store write (this is the exceptional ask()-is-terminating
            # path, not the hot polling loop) -- see `_call`'s own
            # docstring for why that offload exists at all, and why it
            # isn't needed for a call this infrequent. Found by Codex
            # review before this ever shipped.
            with contextlib.suppress(Exception):
                self._queue.reject_ticket(ticket.approval_id, actor="system", channel=self.name, reason=reason)
            raise
