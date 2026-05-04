"""Session — event bus, SQLite-backed EventLog, and observability container."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal

from lazybridge.graph import GraphSchema

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    run_id TEXT,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    ts REAL NOT NULL
)
"""


# Sentinel pushed through the batched writer's queue by ``flush()`` to
# force the writer to commit its current accumulated batch immediately,
# regardless of ``batch_size`` / ``batch_interval``.  Identity check
# only — never persisted.
_FLUSH_SENTINEL = object()


class EventType(StrEnum):
    AGENT_START = "agent_start"
    AGENT_FINISH = "agent_finish"
    LOOP_STEP = "loop_step"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    # Human-in-the-loop audit trail: one event per decision a human
    # takes inside ``HumanEngine`` / ``SupervisorEngine``.  Payload:
    # ``{"agent_name": ..., "kind": "continue"|"retry"|"store"|"tool"|"input",
    #    "command": "<raw repl input>", "result": "<brief>"}``.
    HIL_DECISION = "hil_decision"
    STORE_WRITE = "store_write"
    STORE_READ = "store_read"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"


#: Events the ``"hybrid"`` back-pressure policy considers critical.  An
#: ``EventLog`` configured with ``on_full="hybrid"`` blocks the producer
#: when the writer queue is saturated *only* for these event types, and
#: drops the cheap high-volume ones (``LOOP_STEP``, ``MODEL_REQUEST``,
#: ``MODEL_RESPONSE``).  This matches the operator expectation that an
#: audit-relevant trace (an agent completing, a tool failing) must
#: never silently disappear, while transient telemetry can.
#:
#: Override per-Session via ``Session(critical_events=[...])`` if the
#: defaults don't fit the deployment.
DEFAULT_CRITICAL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EventType.AGENT_START.value,
        EventType.AGENT_FINISH.value,
        EventType.TOOL_ERROR.value,
        EventType.TOOL_CALL.value,
        EventType.TOOL_RESULT.value,
        EventType.HIL_DECISION.value,
    }
)


class EventLog:
    """SQLite-backed event log. Thread-safe via thread-local connections.

    By default ``record()`` performs an ``INSERT + COMMIT`` per event
    on the calling thread.  That is fine for low event rates but
    becomes a bottleneck under sustained load.  Pass
    ``batched=True`` to delegate persistence to a background daemon
    thread that drains a bounded queue and commits in batches; the
    hot path becomes a non-blocking ``queue.put_nowait``.
    """

    def __init__(
        self,
        session_id: str,
        db: str | None = None,
        *,
        batched: bool = False,
        batch_size: int = 100,
        batch_interval: float = 1.0,
        max_queue_size: int = 10_000,
        on_full: Literal["drop", "block", "hybrid"] = "hybrid",
        critical_events: frozenset[str] | set[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self._db = db
        if batched:
            if batch_size < 1:
                raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
            if batch_interval <= 0:
                raise ValueError(f"batch_interval must be > 0, got {batch_interval!r}")
            if max_queue_size < 1:
                raise ValueError(f"max_queue_size must be >= 1, got {max_queue_size!r}")
            if on_full not in ("drop", "block", "hybrid"):
                raise ValueError(f"on_full must be 'drop', 'block', or 'hybrid', got {on_full!r}")
        # In-memory SQLite needs a SHARED cache otherwise every thread
        # gets its own isolated DB and events emitted from worker
        # threads (e.g. ``SupervisorEngine`` via ``asyncio.to_thread``)
        # land in a DB the main thread can never see.  Using the
        # ``file::memory:?cache=shared`` URI uniquely named per
        # ``session_id`` gives us one shared in-memory DB per Session.
        self._uri: str | None
        if db is None:
            self._uri = f"file:memdb_{session_id}?mode=memory&cache=shared"
        else:
            self._uri = None
        self._local = threading.local()
        self._lock = threading.Lock()
        # Registry of every thread-local connection we've handed out.
        # ``close()`` walks it to release file descriptors deterministically;
        # without this, worker-thread connections leak until GC runs.
        self._all_conns: list[sqlite3.Connection] = []
        self._closed = False
        # Keep one anchor connection alive for in-memory DBs — SQLite
        # drops a ``file::memory:?cache=shared`` DB as soon as the last
        # connection closes, so without the anchor the first cleanup
        # of a thread-local conn would wipe the table.
        if self._uri is not None:
            self._anchor: sqlite3.Connection | None = sqlite3.connect(
                self._uri,
                uri=True,
                check_same_thread=False,
            )
        else:
            self._anchor = None
        self._init_schema()

        # Batched writer state — set up after schema so the background
        # thread can rely on the events table existing on first flush.
        self._batched = batched
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._max_queue_size = max_queue_size
        self._on_full = on_full
        # Per-event-type back-pressure — the ``"hybrid"`` policy uses
        # this set to decide which events block under saturation.  A
        # caller-supplied set wins over the defaults (frozen for safety
        # against post-construction mutation).
        self._critical_events: frozenset[str] = (
            frozenset(critical_events) if critical_events is not None else DEFAULT_CRITICAL_EVENT_TYPES
        )
        self._dropped_count = 0
        self._dropped_critical_count = 0
        self._writer_thread: threading.Thread | None = None
        self._writer_queue: queue.Queue[tuple] | None = None
        self._writer_stop = threading.Event()
        if batched:
            self._writer_queue = queue.Queue(maxsize=max_queue_size)
            self._writer_thread = threading.Thread(
                target=self._writer_run,
                name=f"lazybridge-eventlog-{session_id[:8]}",
                daemon=True,
            )
            self._writer_thread.start()

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local connection, initialising schema on first use.

        Schema init is idempotent (``CREATE TABLE IF NOT EXISTS``) so
        calling it once per new thread is safe and necessary — worker
        threads spawned by ``asyncio.to_thread`` otherwise end up with
        a connection that has no ``events`` table.
        """
        if self._closed:
            raise RuntimeError("EventLog is closed")
        if not hasattr(self._local, "conn"):
            if self._db:
                conn = sqlite3.connect(self._db, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
            else:
                # Shared in-memory DB — see __init__ for the URI rationale.
                conn = sqlite3.connect(self._uri or "", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Ensure the schema exists on every thread-local connection.
            # ``CREATE TABLE IF NOT EXISTS`` makes this cheap + idempotent;
            # persistent DBs skip creation after the first hit.
            conn.execute(_SCHEMA_DDL)
            conn.commit()
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return self._local.conn

    def close(self) -> None:
        """Close every thread-local connection and the anchor (if any).

        Idempotent.  After ``close()`` further ``record`` / ``query``
        calls raise ``RuntimeError``.  Required for deterministic FD
        cleanup in long-running services that spawn Sessions per
        request.

        The lock is held across the entire shutdown so a concurrent
        ``record`` that already obtained a connection via ``_conn``
        can't race ahead and commit against a connection we're about
        to close — SQLite would otherwise raise ``ProgrammingError:
        Cannot operate on a closed database``.

        When batching is enabled the background writer is signalled to
        drain its queue and exit before connections are released.
        """
        # Idempotent: a second close() (typically from ``__del__`` at
        # GC time) must not re-trigger flush() — that would push a
        # sentinel into a queue whose writer thread has already
        # exited, causing flush() to block for the full timeout
        # waiting for an ack that never comes.
        if self._closed:
            return
        # Drain + stop the writer first.  Done outside the lock so the
        # writer thread can finish using the EventLog's connections;
        # ``record_many`` checks ``self._closed`` itself.  Order:
        # (1) flush so any pending events land, (2) set stop and push a
        # sentinel so the writer wakes from its long ``queue.get``
        # timeout immediately, (3) join.
        if self._batched and self._writer_thread is not None:
            self.flush(timeout=5.0)
            self._writer_stop.set()
            assert self._writer_queue is not None
            try:
                self._writer_queue.put_nowait(_FLUSH_SENTINEL)
            except queue.Full:
                # Queue is saturated — the writer is busy and will see
                # the stop flag on its next iteration anyway.
                pass
            self._writer_thread.join(timeout=5.0)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            conns = list(self._all_conns)
            self._all_conns.clear()
            anchor = self._anchor
            self._anchor = None
            for c in conns:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            if anchor is not None:
                try:
                    anchor.close()
                except sqlite3.Error:
                    pass

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        self._conn().execute(_SCHEMA_DDL)
        self._conn().commit()

    def record(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> None:
        # Fast-path check: if ``close()`` has fired we fail fast instead
        # of executing against a connection that's about to disappear.
        # The narrow race where close fires after this check is bounded
        # to a single ``INSERT + COMMIT`` — SQLite will either succeed
        # or raise ``ProgrammingError``, which the caller can ignore.
        if self._closed:
            raise RuntimeError("EventLog is closed")
        row = (
            self.session_id,
            run_id,
            str(event_type),
            json.dumps(payload),
            time.time(),
        )
        if self._batched:
            self._submit_to_writer(row, event_type=str(event_type))
            return
        conn = self._conn()
        conn.execute(
            "INSERT INTO events (session_id, run_id, event_type, payload, ts) VALUES (?,?,?,?,?)",
            row,
        )
        conn.commit()

    def record_many(self, rows: list[tuple]) -> None:
        """Insert a batch of pre-serialised rows in a single transaction.

        Each ``row`` is the 5-tuple
        ``(session_id, run_id, event_type, payload_json, ts)`` —
        the on-disk shape, not a dict.  Used by the background batched
        writer; callers should use :meth:`record` instead.
        """
        if not rows:
            return
        if self._closed:
            raise RuntimeError("EventLog is closed")
        conn = self._conn()
        conn.executemany(
            "INSERT INTO events (session_id, run_id, event_type, payload, ts) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Batched writer — opt-in, ``batched=True`` at construction.
    # ------------------------------------------------------------------

    def _submit_to_writer(self, row: tuple, *, event_type: str) -> None:
        """Push a row to the writer queue.

        ``on_full`` policy:

        * ``"block"``  — back-pressure unconditionally; the producer
          waits until the writer drains.
        * ``"drop"``   — back-pressure never; on saturation the row is
          dropped silently (with a doubling-interval warning).  Useful
          for "telemetry must not block production traffic".
        * ``"hybrid"`` — block for *critical* event types
          (``AGENT_*``, ``TOOL_*``, ``HIL_DECISION`` by default; see
          ``critical_events``) and drop the cheap high-volume ones
          (``LOOP_STEP`` / ``MODEL_REQUEST`` / ``MODEL_RESPONSE``).
          Default; balances audit completeness against producer
          latency.
        """
        assert self._writer_queue is not None
        if self._on_full == "block":
            self._writer_queue.put(row)
            return
        if self._on_full == "hybrid" and event_type in self._critical_events:
            # Block on the audit-critical path; drop only for cheap
            # telemetry.  Producer latency takes precedence over
            # losing an AGENT_FINISH or TOOL_ERROR.
            self._writer_queue.put(row)
            return
        try:
            self._writer_queue.put_nowait(row)
        except queue.Full:
            self._dropped_count += 1
            if event_type in self._critical_events:
                # Reachable only when ``on_full="drop"``: the operator
                # opted out of back-pressure entirely, so we still drop
                # but track separately so it's distinguishable from
                # benign telemetry loss in the warning text.
                self._dropped_critical_count += 1
            # Warn on first drop and at every doubling thereafter so
            # operators see saturation without flooding logs.
            n = self._dropped_count
            if n == 1 or (n & (n - 1)) == 0:
                critical_note = (
                    f" ({self._dropped_critical_count} of which were critical event types)"
                    if self._dropped_critical_count
                    else ""
                )
                warnings.warn(
                    f"EventLog batched queue is full; dropped {n} event(s)"
                    f"{critical_note} so far on session {self.session_id[:8]}. "
                    f"Raise max_queue_size=, switch to on_full='hybrid' "
                    f"(default — blocks only on critical events), or "
                    f"on_full='block' if drops aren't acceptable.",
                    UserWarning,
                    stacklevel=4,
                )

    def _writer_run(self) -> None:
        """Background thread: drain queue, INSERT in batches, COMMIT.

        Uses ``Queue.task_done()`` after each successful flush so
        callers can join on the queue to confirm all submitted events
        have been persisted (used by :meth:`flush`).
        """
        assert self._writer_queue is not None
        batch: list[tuple] = []
        deadline = time.monotonic() + self._batch_interval
        while True:
            timeout = max(0.0, deadline - time.monotonic())
            try:
                row = self._writer_queue.get(timeout=timeout)
                if row is _FLUSH_SENTINEL:
                    # ``flush()`` (or ``close()``) is asking us to commit
                    # everything pending right now.  Drain the in-progress
                    # batch then ack the sentinel itself so
                    # ``unfinished_tasks`` decrements.
                    if batch:
                        self._flush_batch(batch)
                        batch = []
                    self._writer_queue.task_done()
                    # ``close()`` wakes us with a sentinel after setting
                    # the stop flag — exit immediately rather than going
                    # back to a (possibly long) batch_interval get().
                    if self._writer_stop.is_set():
                        return
                    deadline = time.monotonic() + self._batch_interval
                    continue
                batch.append(row)
                if len(batch) >= self._batch_size:
                    self._flush_batch(batch)
                    batch = []
                    deadline = time.monotonic() + self._batch_interval
            except queue.Empty:
                if batch:
                    self._flush_batch(batch)
                    batch = []
                deadline = time.monotonic() + self._batch_interval
                if self._writer_stop.is_set():
                    return

    def _flush_batch(self, batch: list[tuple]) -> None:
        assert self._writer_queue is not None
        try:
            self.record_many(batch)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"EventLog batched flush failed: {type(exc).__name__}: {exc}.  {len(batch)} event(s) lost.",
                UserWarning,
                stacklevel=2,
            )
        finally:
            # Mark each item as done regardless of flush outcome so a
            # ``flush()`` waiter never deadlocks on a write failure.
            for _ in batch:
                self._writer_queue.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """Block until every event submitted before the call is persisted.

        No-op when ``batched=False``.  Pushes a flush sentinel so the
        writer commits its current batch immediately rather than waiting
        for ``batch_size`` or ``batch_interval``, then waits on the
        queue's ``task_done`` accounting.  Returns early if ``timeout``
        elapses; the queue may still have items in that case.
        """
        if not self._batched or self._writer_queue is None:
            return
        try:
            self._writer_queue.put_nowait(_FLUSH_SENTINEL)
        except queue.Full:
            # If the queue is saturated, fall back to a blocking put so
            # flush still has well-defined semantics — accepting the
            # backpressure cost is the right trade-off here since the
            # caller asked for a synchronous barrier.
            self._writer_queue.put(_FLUSH_SENTINEL)
        # ``Queue.join`` has no timeout in stdlib; use ``unfinished_tasks``
        # plus ``all_tasks_done`` polling under the queue's own lock.
        deadline = time.monotonic() + timeout
        with self._writer_queue.all_tasks_done:
            while self._writer_queue.unfinished_tasks > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._writer_queue.all_tasks_done.wait(timeout=remaining)

    def query(self, *, run_id: str | None = None, event_type: EventType | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE session_id=?"
        params: list[Any] = [self.session_id]
        if run_id:
            sql += " AND run_id=?"
            params.append(run_id)
        if event_type:
            sql += " AND event_type=?"
            params.append(str(event_type))
        sql += " ORDER BY id ASC"
        rows = self._conn().execute(sql, params).fetchall()
        # Include ``run_id`` so callers (e.g. usage_summary) don't have
        # to re-query the DB row-by-row just to resolve it.
        return [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "run_id": r["run_id"],
                "payload": json.loads(r["payload"]),
                "ts": r["ts"],
            }
            for r in rows
        ]


class Session:
    """Container for observability config: exporters, redaction, EventLog."""

    def __init__(
        self,
        *,
        db: str | None = None,
        exporters: list[Any] | None = None,
        redact: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        redact_on_error: Literal["fallback", "strict"] = "strict",
        console: bool = False,
        batched: bool = False,
        batch_size: int = 100,
        batch_interval: float = 1.0,
        max_queue_size: int = 10_000,
        on_full: Literal["drop", "block", "hybrid"] = "hybrid",
        critical_events: frozenset[str] | set[str] | None = None,
    ) -> None:
        """Construct a Session.

        Back-pressure policy
        --------------------
        ``on_full`` selects what happens when the batched-writer queue
        is saturated (``batched=True``):

        * ``"hybrid"`` (default) — block for audit-critical event types
          (``AGENT_*``, ``TOOL_*``, ``HIL_DECISION``; override via
          ``critical_events=``) and silently drop the cheap high-volume
          ones (``LOOP_STEP`` / ``MODEL_REQUEST`` / ``MODEL_RESPONSE``).
          A buggy slow exporter no longer makes ``AGENT_FINISH`` or
          ``TOOL_ERROR`` disappear, while a steady-state telemetry
          firehose still doesn't add latency to the producer.
        * ``"block"`` — back-pressure unconditionally.  Pick this when
          every event must persist (compliance) and producer latency
          can absorb the wait.
        * ``"drop"``  — never back-pressure.  Saturation drops events
          silently (with a doubling-interval warning).  Pick this when
          telemetry must not block production traffic and lossy traces
          are acceptable.

        Redactor failure modes
        ----------------------
        ``redact_on_error`` governs what happens when a ``redact``
        callable either raises or returns a non-dict:

        * ``"strict"`` (default) — warn once, then **drop the event
          entirely**.  No record in the EventLog, no export to
          exporters.  Fail-closed: unredacted data can never leak via
          this session.  This is the safe default for compliance
          workloads where a broken redactor is a bug to be fixed, not
          a reason to keep leaking.
        * ``"fallback"`` — warn once, then record and export the
          **original, unredacted** payload.  Observability is preserved
          at the cost of potentially leaking unredacted data through
          the event bus.  Useful for development; opt in explicitly
          when you want it.

        Default is ``"strict"``: a redactor that fails closes the
        event rather than persisting it unredacted.  Pass
        ``redact_on_error="fallback"`` to keep the unredacted event
        flowing when redaction fails (lower fidelity, but lossless).
        """
        if redact_on_error not in ("fallback", "strict"):
            raise ValueError(
                f"Session(redact_on_error={redact_on_error!r}): must be "
                f"'fallback' (warn + pass through) or 'strict' (warn + "
                f"drop event)."
            )
        self.session_id = str(uuid.uuid4())
        self.events = EventLog(
            self.session_id,
            db=db,
            batched=batched,
            batch_size=batch_size,
            batch_interval=batch_interval,
            max_queue_size=max_queue_size,
            on_full=on_full,
            critical_events=critical_events,
        )
        self._exporters: list[Any] = list(exporters or [])
        self._redact = redact
        self._redact_on_error = redact_on_error
        self._lock = threading.Lock()
        self.graph = GraphSchema(session_id=self.session_id)
        if console:
            # Late import to avoid circular dependency with exporters
            from lazybridge.exporters import ConsoleExporter

            self._exporters.append(ConsoleExporter())

    # ------------------------------------------------------------------
    # Graph registration — called by Agent.__init__ when session= is set.
    # ------------------------------------------------------------------

    def register_agent(self, agent: Any) -> None:
        """Register an agent with this session's graph."""
        self.graph.add_agent(agent)

    def register_tool_edge(self, from_agent: Any, to_agent: Any, *, label: str = "") -> None:
        """Record a tool-call edge between two registered agents."""
        from_id = str(getattr(from_agent, "name", "agent"))
        to_id = str(getattr(to_agent, "name", "agent"))
        from lazybridge.graph import EdgeType

        self.graph.add_edge(from_id, to_id, label=label, kind=EdgeType.TOOL)

    def add_exporter(self, exporter: Any) -> None:
        with self._lock:
            self._exporters = [*self._exporters, exporter]

    def remove_exporter(self, exporter: Any) -> None:
        with self._lock:
            self._exporters = [e for e in self._exporters if e is not exporter]

    def _warn_once_redact(self, attr: str, msg: str) -> None:
        """Emit a UserWarning at most once per (Session, reason) pair.

        Reason is distinguished by ``attr`` so a redactor that both raises
        on some events and returns a non-dict on others emits one warning
        per failure mode rather than staying silent after the first.

        State lives on ``self`` (the Session), not on the redactor
        callable.  A redactor function shared across many short-lived
        Session instances therefore warns once *per session*; stamping
        the callable would suppress every session after the first.
        """
        if self._redact is None:
            return
        flag_name = f"_warn_once_redact_{attr}"
        if getattr(self, flag_name, False):
            return
        import warnings

        warnings.warn(
            f"Session redact callable: {msg}.  redact_on_error={self._redact_on_error!r}.",
            stacklevel=3,
        )
        setattr(self, flag_name, True)

    def emit(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> None:
        if self._redact:
            # Validate the redactor — two failure modes:
            #   (a) it raises
            #   (b) it returns something that isn't a dict
            # ``redact_on_error="fallback"`` warns once and persists
            # the original payload; ``"strict"`` (the default) drops
            # the event so no unredacted data can leak via this session.
            try:
                result: Any = self._redact(payload)
            except Exception as exc:
                self._warn_once_redact(
                    "_lazybridge_raise_warned",
                    f"redact callable raised {type(exc).__name__}: {exc}",
                )
                if self._redact_on_error == "strict":
                    return  # drop event — unredacted data stays off the bus
                # fallback: ``payload`` unchanged, continue to record.
            else:
                if isinstance(result, dict):
                    payload = result
                else:
                    self._warn_once_redact(
                        "_lazybridge_type_warned",
                        f"redact returned {type(result).__name__!s}; expected dict",
                    )
                    if self._redact_on_error == "strict":
                        return
                    # fallback: ``payload`` unchanged.
        self.events.record(event_type, payload, run_id=run_id)
        exporters = self._exporters  # snapshot for thread safety
        event_dict = {"event_type": str(event_type), "session_id": self.session_id, "run_id": run_id, **payload}
        for exp in exporters:
            try:
                exp.export(event_dict)
            except Exception as exc:
                # Warn once per exporter instance so a buggy exporter
                # is visible in logs instead of silently eating events.
                if not getattr(exp, "_lazybridge_export_warned", False):
                    import warnings

                    warnings.warn(
                        f"Exporter {exp.__class__.__name__} raised "
                        f"{type(exc).__name__}: {exc}. Further failures "
                        f"from this exporter will be suppressed.",
                        stacklevel=2,
                    )
                    try:
                        exp._lazybridge_export_warned = True  # type: ignore[attr-defined]
                    except AttributeError:
                        pass

    def flush(self, timeout: float = 5.0) -> None:
        """Drain the EventLog's batched-writer queue.

        No-op when ``batched=False``.  Useful before a checkpoint or
        a clean shutdown so recently-emitted events are persisted
        before the caller proceeds.
        """
        self.events.flush(timeout=timeout)

    def close(self) -> None:
        """Release the underlying EventLog's SQLite connections.

        Idempotent.  Call this when a Session's lifetime ends (e.g.
        end of an HTTP request) so file descriptors don't linger until
        the owning thread exits.  Using Session as a context manager is
        equivalent.  Exporters that expose ``close()`` are flushed too
        — useful for OTelExporter's orphaned-span cleanup.
        """
        self.events.close()
        for exp in list(self._exporters):
            close = getattr(exp, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Exporter shutdown must never mask the real reason
                    # Session.close() is being called.
                    pass

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def usage_summary(self) -> dict[str, Any]:
        """Aggregate token usage and cost across all agent runs in this session.

        Returns a dict with:
          - "total": {input_tokens, output_tokens, cost_usd}
          - "by_agent": {agent_name: {input_tokens, output_tokens, cost_usd}}
          - "by_run":   {run_id:    {agent_name, input_tokens, output_tokens, cost_usd}}

        O(events) with TWO queries total (AGENT_START +
        MODEL_RESPONSE).  ``EventLog.query`` exposes ``run_id``
        directly in the result dict.
        """
        # Two bulk queries.  No per-row DB trip.
        agent_starts = self.events.query(event_type=EventType.AGENT_START)
        model_responses = self.events.query(event_type=EventType.MODEL_RESPONSE)

        # Build run_id → agent_name map.
        run_agent: dict[str, str] = {
            row["run_id"]: row["payload"].get("agent_name", "unknown") for row in agent_starts if row.get("run_id")
        }

        total = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_agent: dict[str, dict[str, Any]] = {}
        by_run: dict[str, dict[str, Any]] = {}

        for row in model_responses:
            p = row["payload"]
            run_id = row.get("run_id")
            agent_name = run_agent.get(run_id or "", "unknown") if run_id else "unknown"

            in_tok = p.get("input_tokens", 0) or 0
            out_tok = p.get("output_tokens", 0) or 0
            cost = p.get("cost_usd") or 0.0

            total["input_tokens"] += in_tok
            total["output_tokens"] += out_tok
            total["cost_usd"] += cost

            ag = by_agent.setdefault(agent_name, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            ag["input_tokens"] += in_tok
            ag["output_tokens"] += out_tok
            ag["cost_usd"] += cost

            if run_id:
                rn = by_run.setdefault(
                    run_id, {"agent_name": agent_name, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                )
                rn["input_tokens"] += in_tok
                rn["output_tokens"] += out_tok
                rn["cost_usd"] += cost

        return {"total": total, "by_agent": by_agent, "by_run": by_run}
