"""Replay a previously-recorded session by reading the SQLite event log.

Live mode pushes into the hub via :class:`HubExporter`. Replay mode
needs the same interface, but driven by a thread that reads
historical events and re-publishes them with a configurable speed
factor. The controller exposes pause/resume/step/speed so the UI
can scrub through a finished run.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any

from lazybridge.ext.viz._normalizer import normalise_event
from lazybridge.ext.viz.exporter import EventHub


def _open_readonly(db_path: str) -> sqlite3.Connection:
    # Open the SQLite file read-only so a concurrent live producer
    # never sees lock contention from a tab we opened just to look.
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def list_sessions(db_path: str) -> list[dict[str, Any]]:
    """Return ``[{session_id, count, first_ts, last_ts}]`` ordered newest first."""
    conn = _open_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT session_id, COUNT(*) AS c, MIN(ts) AS first_ts, MAX(ts) AS last_ts "
            "FROM events GROUP BY session_id ORDER BY last_ts DESC"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"session_id": r["session_id"], "count": r["c"], "first_ts": r["first_ts"], "last_ts": r["last_ts"]}
        for r in rows
    ]


def load_session_events(db_path: str, session_id: str) -> list[dict[str, Any]]:
    """Read every event for ``session_id`` in chronological order."""
    import json

    conn = _open_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT id, run_id, event_type, payload, ts FROM events WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        payload = json.loads(r["payload"])
        out.append(
            {
                "event_type": r["event_type"],
                "session_id": session_id,
                "run_id": r["run_id"],
                "ts": r["ts"],
                **(payload if isinstance(payload, dict) else {"payload": payload}),
            }
        )
    return out


def reconstruct_graph(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Best-effort graph from a flat event list (used when the live
    GraphSchema is no longer available — typical for replay).

    Adds an agent node for every distinct ``agent_name`` seen in
    ``agent_start`` (or any other event). Tool nodes appear lazily on
    the frontend as ``tool_call`` events stream in, so we don't need
    to add them here.
    """
    nodes: dict[str, dict[str, Any]] = {}
    for ev in events:
        agent = ev.get("agent_name")
        if not agent or agent in nodes:
            continue
        nodes[agent] = {"id": agent, "name": agent, "type": "agent"}
    return {"nodes": list(nodes.values()), "edges": []}


class ReplayController:
    """Pumps recorded events into a live :class:`EventHub`.

    Behaviour:
      - Plays in real time scaled by ``speed`` (1.0 = real time).
      - Pause stops the pump; resume continues from the same offset.
      - Step advances exactly one event regardless of speed/pause.
    """

    def __init__(self, hub: EventHub, events: list[dict[str, Any]], *, speed: float = 1.0) -> None:
        self._hub = hub
        self._events = events
        self._idx = 0
        self._speed = max(0.1, float(speed))
        self._paused = True  # start paused so the UI controls drive playback
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        t = threading.Thread(target=self._run, name="lazybridge-viz-replay", daemon=True)
        t.start()
        self._thread = t

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def play(self) -> None:
        with self._lock:
            self._paused = False
        self._wake.set()

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def set_speed(self, speed: float) -> None:
        with self._lock:
            self._speed = max(0.1, float(speed))
        self._wake.set()

    def step(self) -> None:
        """Emit exactly one event regardless of pause state."""
        with self._lock:
            if self._idx >= len(self._events):
                return
            ev = self._events[self._idx]
            self._idx += 1
        self._hub.publish(normalise_event(ev))

    @property
    def progress(self) -> tuple[int, int]:
        return self._idx, len(self._events)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                paused = self._paused
                idx = self._idx
                speed = self._speed
            if paused or idx >= len(self._events):
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            now_ev = self._events[idx]
            next_ev = self._events[idx + 1] if idx + 1 < len(self._events) else None
            self._hub.publish(normalise_event(now_ev))
            with self._lock:
                self._idx = idx + 1
            if next_ev is not None:
                gap = max(0.0, (next_ev.get("ts", 0.0) - now_ev.get("ts", 0.0)) / speed)
                gap = min(gap, 2.0)  # cap so a long idle in the recorded run doesn't stall us
                if gap > 0:
                    self._wake.wait(timeout=gap)
                    self._wake.clear()
