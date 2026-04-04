"""LazySession — shared container for multi-agent pipelines.

A session holds:
  - LazyStore:  shared blackboard (read/write from any agent)
  - EventLog:   built-in event tracking (always on, level configurable)
  - GraphSchema: auto-built graph of registered agents/tools (for GUI)
  - gather():   concurrent agent execution helper

Usage::

    # Simple session (in-memory, basic tracking)
    sess = LazySession()

    # Persistent session
    sess = LazySession(db="pipeline.db", tracking="verbose")

    researcher = LazyAgent("anthropic", name="researcher", session=sess)
    writer     = LazyAgent("openai",    name="writer",     session=sess)

    # Concurrent execution
    await sess.gather(
        researcher.aloop("find news about X"),
        writer.aloop("find news about Y"),
    )

    # Parallel: all agents run on the same task concurrently
    news_tool = sess.as_tool("news", "AI news by region", mode="parallel")

    # Chain: agents run sequentially, each gets previous output as context
    pipeline_tool = sess.as_tool("pipeline", "Research then analyse", mode="chain")

    orchestrator.loop("coordinate", tools=[news_tool, pipeline_tool])
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Awaitable
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lazybridge.graph.schema import GraphSchema
from lazybridge.lazy_store import LazyStore

if TYPE_CHECKING:
    from lazybridge.lazy_agent import LazyAgent
    from lazybridge.lazy_tool import LazyTool

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _ChainState — internal state propagated between chain steps
# ---------------------------------------------------------------------------

class _ChainState:
    """Immutable-per-step state object for the chain pipeline.

    Each step in a chain produces a new _ChainState that is passed to the
    next step.  Separating state from logic makes the handoff contract
    explicit and the loop body readable.

    Attributes
    ----------
    text : str
        Text representation of this step's output.  Always available.
        Used as the next agent's task when ``ctx`` is None (tool → agent),
        and by the final return when no typed object is available.
    typed : Any | None
        Typed Pydantic object produced by this step, or None.
        Only set when the step was an agent with ``output_schema`` active.
        The chain returns this directly if the *last* step set it.
    ctx : LazyContext | None
        If not None, the previous step was an agent and its output should
        be injected as context into the next step.  The next agent then
        receives the *original* task string as its message, with this
        context merged into its system prompt.

        If None, the previous step was a tool (or this is the first step),
        and ``text`` is passed directly as the next agent's task.

    Handoff semantics (decided by ``ctx``):
        ctx is not None  →  agent → agent  →  inject context, keep original task
        ctx is None      →  tool  → agent  →  use ``text`` as new task
    """

    __slots__ = ("text", "typed", "ctx")

    def __init__(self, text: str, typed: Any, ctx: Any) -> None:
        self.text = text
        self.typed = typed
        self.ctx = ctx


# ---------------------------------------------------------------------------
# TrackLevel
# ---------------------------------------------------------------------------

class TrackLevel(StrEnum):
    OFF     = "off"
    BASIC   = "basic"
    VERBOSE = "verbose"
    FULL    = "full"   # synonym for VERBOSE


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class Event(StrEnum):
    """All event types emitted by LazyAgent during execution."""
    MODEL_REQUEST  = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL      = "tool_call"
    TOOL_RESULT    = "tool_result"
    TOOL_ERROR     = "tool_error"
    AGENT_START    = "agent_start"
    AGENT_FINISH   = "agent_finish"
    LOOP_STEP      = "loop_step"
    # verbose-only events (high volume — only emitted when tracking="verbose")
    MESSAGES       = "messages"
    SYSTEM_CONTEXT = "system_context"
    STREAM_CHUNK   = "stream_chunk"


# These events flood the log at BASIC level; only emitted when tracking="verbose"/"full"
_VERBOSE_ONLY = {Event.MESSAGES, Event.SYSTEM_CONTEXT, Event.STREAM_CHUNK}
_VERBOSE_LEVELS = {TrackLevel.VERBOSE, TrackLevel.FULL}


# ---------------------------------------------------------------------------
# EventLog — SQLite-backed event tracking
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, default=str)
    except Exception as exc:
        _logger.debug("_safe_json: could not serialise %s: %s", type(data).__name__, exc)
        return json.dumps({"_error": "not serialisable", "_type": type(data).__name__})


class EventLog:
    """SQLite-backed event log for an entire session.

    One EventLog per session; agents receive a scoped view via ``agent_log()``.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        session_id  TEXT    NOT NULL,
        agent_id    TEXT,
        agent_name  TEXT,
        event_type  TEXT    NOT NULL,
        data_json   TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ev_session   ON events (session_id);
    CREATE INDEX IF NOT EXISTS idx_ev_agent     ON events (agent_id);
    CREATE INDEX IF NOT EXISTS idx_ev_type      ON events (event_type);
    """

    def __init__(
        self,
        session_id: str,
        db: str | None = None,
        level: TrackLevel | str = TrackLevel.BASIC,
        console: bool = False,
    ) -> None:
        self.session_id = session_id
        self.level = TrackLevel(level) if not isinstance(level, TrackLevel) else level
        self._console = console
        self._db = str(Path(db).resolve()) if db else None
        self._mem: list[dict] | None = [] if db is None else None  # in-memory fallback
        self._lock = threading.Lock()
        self._local = threading.local()  # thread-local connection cache
        if self._db:
            self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db, check_same_thread=False, timeout=30)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=10000")
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: str,
        *,
        agent_id: str | None = None,
        agent_name: str | None = None,
        **data: Any,
    ) -> None:
        """Log an event. No-op when level is OFF.  Verbose-only events are
        skipped unless level is VERBOSE or FULL."""
        if self.level == TrackLevel.OFF:
            return
        if self.level not in _VERBOSE_LEVELS and event_type in _VERBOSE_ONLY:
            return
        row = {
            "timestamp":  _now(),
            "session_id": self.session_id,
            "agent_id":   agent_id,
            "agent_name": agent_name,
            "event_type": event_type,
            "data":       data,
        }
        if self._db:
            try:
                with self._conn() as conn:
                    conn.execute(
                        "INSERT INTO events(timestamp, session_id, agent_id, agent_name,"
                        " event_type, data_json) VALUES (?,?,?,?,?,?)",
                        (
                            row["timestamp"], self.session_id, agent_id,
                            agent_name, event_type, _safe_json(data),
                        ),
                    )
            except Exception as exc:
                _logger.warning("EventLog write failed (event dropped): %s", exc)
        else:
            with self._lock:
                self._mem.append(row)  # type: ignore[union-attr]

        if self._console:
            self._print_event(agent_name, event_type, data)

    def _print_event(self, agent_name: str | None, event_type: str, data: dict) -> None:
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%H:%M:%S")
        label = f"[{agent_name or '?':12s}]"

        if event_type == Event.MODEL_REQUEST:
            model = data.get("model", "")
            msgs  = data.get("n_messages", "")
            line  = f">> model_request   model={model}  msgs={msgs}"
        elif event_type == Event.MODEL_RESPONSE:
            model   = data.get("model", "")
            stop    = data.get("stop_reason", "")
            in_t    = data.get("input_tokens", "")
            out_t   = data.get("output_tokens", "")
            content = data.get("content") or ""
            preview = content.replace("\n", " ").strip()
            if len(preview) > 200:
                preview = preview[:200] + "…"
            line = (
                f"<< model_response  model={model}  stop={stop}  in={in_t} out={out_t}\n"
                f"{'':26s}{label} {preview!r}"
            )
        elif event_type == Event.TOOL_CALL:
            name = data.get("name", "")
            args = str(data.get("arguments", ""))[:120]
            line = f">> tool_call       {name}({args})"
        elif event_type == Event.TOOL_RESULT:
            name   = data.get("name", "")
            result = str(data.get("result", ""))[:120]
            line   = f"<< tool_result     {name} -> {result}"
        elif event_type == Event.TOOL_ERROR:
            name  = data.get("name", "")
            error = str(data.get("error", ""))[:120]
            line  = f"!! tool_error      {name}: {error}"
        else:
            line = f"   {event_type}"

        print(f"{ts} {label} {line}", flush=True)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(
        self,
        *,
        agent_id: str | None = None,
        event_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Return events, optionally filtered by agent and/or event type."""
        if self._db:
            return self._get_db(agent_id=agent_id, event_type=event_type, limit=limit)
        with self._lock:
            events: list[dict] = list(self._mem or [])
        if agent_id:
            events = [e for e in events if e.get("agent_id") == agent_id]
        if event_type:
            events = [e for e in events if e.get("event_type") == event_type]
        return events[-limit:]

    def _get_db(self, *, agent_id, event_type, limit) -> list[dict]:
        clauses = ["session_id = ?"]
        params: list[Any] = [self.session_id]
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT timestamp, agent_id, agent_name, event_type, data_json"
                f" FROM events WHERE {' AND '.join(clauses)}"
                f" ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        # Reverse so result is chronological (oldest→newest), matching in-memory behaviour.
        return [
            {
                "timestamp":  r[0],
                "agent_id":   r[1],
                "agent_name": r[2],
                "event_type": r[3],
                "data":       json.loads(r[4]),
            }
            for r in reversed(rows)
        ]

    def agent_log(self, agent_id: str, agent_name: str | None = None) -> _AgentLog:
        """Return a scoped view for a single agent."""
        return _AgentLog(self, agent_id=agent_id, agent_name=agent_name)


class _AgentLog:
    """Scoped EventLog view for a single LazyAgent.  Used internally."""

    __slots__ = ("_log", "agent_id", "agent_name")

    def __init__(self, log: EventLog, *, agent_id: str, agent_name: str | None) -> None:
        self._log = log
        self.agent_id = agent_id
        self.agent_name = agent_name

    def log(self, event_type: str, **data: Any) -> None:
        self._log.log(event_type, agent_id=self.agent_id, agent_name=self.agent_name, **data)

    def get(self, *, event_type: str | None = None, limit: int = 100) -> list[dict]:
        return self._log.get(agent_id=self.agent_id, event_type=event_type, limit=limit)


# ---------------------------------------------------------------------------
# LazySession
# ---------------------------------------------------------------------------

class LazySession:
    """Shared context for a multi-agent pipeline.

    Holds the LazyStore (blackboard), EventLog (tracking), GraphSchema (graph)
    and provides ``gather()`` for concurrent agent execution.

    Usage::

        sess = LazySession()
        sess = LazySession(db="pipeline.db", tracking="verbose")

        a1 = LazyAgent("anthropic", session=sess)
        a2 = LazyAgent("openai",    session=sess)
        await sess.gather(a1.aloop("task 1"), a2.aloop("task 2"))

        print(sess.store.read_all())
        print(sess.events.get())
        print(sess.graph.to_json())
    """

    def __init__(
        self,
        *,
        db: str | None = None,
        tracking: TrackLevel | str = TrackLevel.BASIC,
        console: bool = False,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.store = LazyStore(db=db)
        self.events = EventLog(self.id, db=db, level=tracking, console=console)
        self.graph = GraphSchema(self.id)
        self._agents: list[Any] = []  # ordered list of registered agents

    # ------------------------------------------------------------------
    # Agent registration (called by LazyAgent.__init__)
    # ------------------------------------------------------------------

    def _register_agent(self, agent: LazyAgent) -> None:
        self.graph.add_agent(agent)
        self._agents.append(agent)

    # ------------------------------------------------------------------
    # Concurrent execution
    # ------------------------------------------------------------------

    async def gather(self, *coros: Awaitable) -> list[Any]:
        """Run multiple coroutines concurrently and return their results.

        Usage::

            results = await sess.gather(
                researcher.aloop("find X"),
                analyst.aloop("find Y"),
            )
        """
        return list(await asyncio.gather(*coros, return_exceptions=True))

    # ------------------------------------------------------------------
    # Pipeline as tool
    # ------------------------------------------------------------------

    def as_tool(
        self,
        name: str,
        description: str,
        *,
        mode: str | None = None,
        participants: list[Any] | None = None,
        combiner: str = "concat",
        native_tools: list[Any] | None = None,
        entry_agent: Any | None = None,
        guidance: str | None = None,
    ) -> LazyTool:
        """Expose agents (or nested pipeline tools) as a single LazyTool.

        Two modes are available — no wrapper function needed:

        ``mode="parallel"``
            All participants receive the same task and run concurrently.
            Their outputs are combined (default: concatenated with agent name
            headers). Participants can be LazyAgent or LazyTool instances.

            ::

                sess = LazySession()
                a = LazyAgent("anthropic", name="us",     session=sess)
                b = LazyAgent("openai",    name="europe", session=sess)
                c = LazyAgent("google",    name="asia",   session=sess)

                news_tool = sess.as_tool("world_news", "AI news by region",
                                         mode="parallel")
                master.loop("Report AI news", tools=[news_tool])

        ``mode="chain"``
            Participants run sequentially in registration order (or the order
            of ``participants=``).  Each receives the previous participant's
            output as context. Participants can be LazyAgent or LazyTool.

            ::

                researcher = LazyAgent("anthropic", name="researcher", session=sess)
                analyst    = LazyAgent("openai",    name="analyst",    session=sess)

                pipeline_tool = sess.as_tool("pipeline", "Research then analyse",
                                              mode="chain")
                master.loop("Analyse fusion energy", tools=[pipeline_tool])

        Legacy (backward-compatible): pass ``entry_agent=`` without ``mode``
        to delegate to a single agent as before.

        Parameters
        ----------
        name:
            Tool name exposed to the orchestrator LLM.
        description:
            Tool description exposed to the orchestrator LLM.
        mode:
            ``"parallel"`` or ``"chain"``.  Required unless using legacy
            ``entry_agent=`` path.
        participants:
            Explicit ordered list of LazyAgent / LazyTool participants.
            Defaults to all agents registered in this session in order.
        combiner:
            Parallel mode only. ``"concat"`` (default) joins outputs with
            agent-name headers; ``"last"`` returns only the last result.
        entry_agent:
            Legacy single-agent delegation (kept for backward compatibility).
        guidance:
            Optional hint injected into the tool description for the LLM.
        """
        from lazybridge.lazy_tool import LazyTool

        # ── Legacy path ────────────────────────────────────────────────────
        if mode is None:
            if entry_agent is None:
                raise ValueError(
                    "Provide mode='parallel'|'chain', or entry_agent= for single-agent delegation."
                )
            return LazyTool.from_agent(
                entry_agent, name=name, description=description, guidance=guidance
            )

        # ── Resolve participant list ────────────────────────────────────────
        _parts: list[Any] = participants if participants is not None else list(self._agents)
        if not _parts:
            raise ValueError(
                "No participants found. Pass participants= or register agents in the session first."
            )

        _native = native_tools or []

        # ── Parallel mode ──────────────────────────────────────────────────
        if mode == "parallel":
            if combiner not in ("concat", "last"):
                raise ValueError(
                    f"Invalid combiner {combiner!r} for parallel mode. Use 'concat' or 'last'."
                )

            def _run_parallel(task: str) -> str:
                from lazybridge.lazy_run import run_async

                async def _gather() -> str:
                    coros = []
                    for p in _parts:
                        if hasattr(p, "achat"):          # LazyAgent
                            kw = {"native_tools": _native} if _native else {}
                            schema = getattr(p, "output_schema", None)
                            if schema is not None:
                                coros.append(p.ajson(task, schema, **kw))
                            else:
                                coros.append(p.achat(task, **kw))
                        else:                            # LazyTool
                            coros.append(p.arun({"task": task}))
                    results = await asyncio.gather(*coros, return_exceptions=True)
                    if not results:
                        return ""

                    def _to_text(r: Any) -> str:
                        if isinstance(r, BaseException):
                            return f"[ERROR: {type(r).__name__}: {r}]"
                        if hasattr(r, "model_dump_json"):  # Pydantic model
                            return r.model_dump_json(indent=2)
                        if hasattr(r, "content"):          # CompletionResponse
                            return r.content
                        return str(r)

                    if combiner == "last":
                        return _to_text(results[-1])
                    # combiner == "concat" (validated at creation time)
                    parts_out = []
                    for p, r in zip(_parts, results):
                        pname = getattr(p, "name", "?")
                        parts_out.append(f"[{pname}]\n{_to_text(r)}")
                    return "\n\n".join(parts_out)

                return run_async(_gather())

            return LazyTool.from_function(_run_parallel, name=name, description=description)

        # ── Chain mode ─────────────────────────────────────────────────────
        if mode == "chain":
            def _run_chain(task: str) -> Any:
                from lazybridge.lazy_context import LazyContext

                # _ChainState carries the result of each step to the next.
                # See _ChainState docstring for the full handoff contract.
                #
                # Short version:
                #   state.ctx is not None  →  previous step was an agent
                #                              inject ctx into next agent's system prompt
                #                              keep original task as the message
                #   state.ctx is None      →  previous step was a tool (or first step)
                #                              use state.text directly as next agent's message
                #
                # Call dispatch (per agent step, in priority order):
                #   output_schema set  → json()  — structured output + JSON suffix enforcement
                #   tools/native_tools → loop()  — tool-calling loop required
                #   else               → chat()  — single turn, no tools
                #
                # Final return:
                #   state.typed  →  Pydantic object if last step had output_schema
                #   state.text   →  plain string otherwise

                state = _ChainState(text=task, typed=None, ctx=None)

                for p in _parts:
                    if hasattr(p, "chat"):                      # LazyAgent
                        kw: dict[str, Any] = {}
                        if _native:
                            kw["native_tools"] = _native

                        if state.ctx is not None:
                            # Previous step was an agent: inject its context so the
                            # current agent sees prior results in its system prompt,
                            # while still receiving the *original* task as its message.
                            # Keeping the original task preserves the pipeline's goal
                            # across all agent steps — the context is additive, not
                            # a replacement.
                            kw["context"] = state.ctx
                            current_task = task
                        else:
                            # Previous step was a tool (or first step): the tool's
                            # output is the raw material to process, so it becomes
                            # the agent's message directly.
                            current_task = state.text

                        schema = getattr(p, "output_schema", None)
                        _has_tools = bool(
                            getattr(p, "tools", None) or
                            getattr(p, "native_tools", None) or
                            kw.get("native_tools")
                        )

                        if schema is not None:
                            result = p.json(current_task, schema, **kw)
                            state = _ChainState(
                                text=(
                                    result.model_dump_json()
                                    if hasattr(result, "model_dump_json")
                                    else str(result)
                                ),
                                typed=result,
                                ctx=LazyContext.from_agent(p),
                            )
                        elif _has_tools:
                            resp = p.loop(current_task, **kw)
                            state = _ChainState(
                                text=resp.content if hasattr(resp, "content") else str(resp),
                                typed=None,
                                ctx=LazyContext.from_agent(p),
                            )
                        else:
                            resp = p.chat(current_task, **kw)
                            state = _ChainState(
                                text=resp.content if hasattr(resp, "content") else str(resp),
                                typed=None,
                                ctx=LazyContext.from_agent(p),
                            )

                    elif hasattr(p, "run"):                     # LazyTool (nested pipeline)
                        result = p.run({"task": state.text})
                        # ctx=None signals to the next agent: use text as task,
                        # not as context — tool output is data, not interpretation.
                        state = _ChainState(text=str(result), typed=None, ctx=None)

                    else:
                        raise TypeError(
                            f"Chain participant {p!r} is not a LazyAgent (needs .chat()) "
                            f"or a LazyTool (needs .run()). "
                            f"Participants must be LazyAgent instances or LazyTool objects."
                        )

                # Return the typed Pydantic object if the last step produced one,
                # otherwise the plain text output.
                # When this chain is used as a tool inside loop(), the executor
                # serialises Pydantic objects via model_dump_json() automatically.
                return state.typed if state.typed is not None else state.text

            return LazyTool.from_function(_run_chain, name=name, description=description)

        raise ValueError(f"Unknown mode {mode!r}. Use 'parallel' or 'chain'.")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise session graph to JSON (for GUI)."""
        return self.graph.to_json()

    @classmethod
    def from_json(cls, text: str, **kwargs) -> LazySession:
        """Load a session from a previously serialised graph JSON."""
        data = json.loads(text)
        sess = cls(**kwargs)
        sess.graph = GraphSchema.from_dict(data)
        sess.id = data.get("session_id", sess.id)
        # Rebind EventLog so events logged after restore use the correct session_id.
        sess.events.session_id = sess.id
        return sess

    def __repr__(self) -> str:
        return f"LazySession(id={self.id[:8]}..., store={len(self.store.keys())} keys)"
