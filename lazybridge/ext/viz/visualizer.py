"""Public entry point: ``Visualizer`` ties the hub, server, and
optional replay controller together.

Two construction paths:

* ``Visualizer(session)`` — live mode. Installs a :class:`HubExporter`
  on the session, serves the live graph + store, optionally opens a
  browser tab.
* ``Visualizer.replay(db=..., session_id=...)`` — replay mode. Reads
  events from SQLite, reconstructs a minimal graph, and exposes
  pause/play/speed/step controls to the UI.
"""

from __future__ import annotations

import time
import webbrowser
from typing import TYPE_CHECKING, Any

from lazybridge.ext.viz.exporter import EventHub, HubExporter
from lazybridge.ext.viz.replay import (
    ReplayController,
    list_sessions,
    load_session_events,
    reconstruct_graph,
)
from lazybridge.ext.viz.server import VizServer

if TYPE_CHECKING:
    from lazybridge.session import Session
    from lazybridge.store import Store


class Visualizer:
    """Live or replay visualizer.

    Use as a context manager so the HTTP server is shut down cleanly
    when the with-block exits. The browser stays open across that
    boundary; the user is expected to close the tab themselves.
    """

    # Class-level attribute hints so mypy sees fields populated only by
    # the alternate ``replay()`` constructor (set via ``cls.__new__(cls)``).
    _fixed_graph: dict[str, Any] | None = None
    _fixed_session_id: str | None = None

    def __init__(
        self,
        session: Session,
        *,
        store: Store | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        auto_open: bool = True,
    ) -> None:
        self._session = session
        self._store = store
        self._hub = EventHub()
        self._exporter = HubExporter(self._hub)
        self._mode = "live"
        self._replay: ReplayController | None = None

        # Wire the hub into the live session
        session.add_exporter(self._exporter)

        self._server = VizServer(
            self._hub,
            graph_provider=self._graph_payload,
            store_provider=self._store_payload,
            meta_provider=self._meta_payload,
            host=host,
            port=port,
        )
        self._auto_open = auto_open
        self._opened = False

    # ------------------------------------------------------------------
    # Replay constructor
    # ------------------------------------------------------------------

    @classmethod
    def replay(
        cls,
        db: str,
        *,
        session_id: str | None = None,
        speed: float = 1.0,
        host: str = "127.0.0.1",
        port: int = 0,
        auto_open: bool = True,
    ) -> Visualizer:
        sessions = list_sessions(db)
        if not sessions:
            raise ValueError(f"No sessions found in {db!r}")
        sid = session_id or sessions[0]["session_id"]
        events = load_session_events(db, sid)
        graph = reconstruct_graph(events)

        # Build a free-standing instance without a live Session
        self = cls.__new__(cls)
        self._session = None  # type: ignore[assignment]
        self._store = None
        self._hub = EventHub()
        self._exporter = HubExporter(self._hub)
        self._mode = "replay"
        self._replay = ReplayController(self._hub, events, speed=speed)
        self._fixed_graph = graph
        self._fixed_session_id = sid

        self._server = VizServer(
            self._hub,
            graph_provider=lambda: graph,
            store_provider=lambda: {},
            meta_provider=lambda: {
                "mode": "replay",
                "session_id": sid,
                "events_total": len(events),
            },
            control_handler=self._on_control,
            host=host,
            port=port,
        )
        self._auto_open = auto_open
        self._opened = False
        return self

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._server.start()
        if self._replay is not None:
            self._replay.start()
        if self._auto_open and not self._opened:
            self._opened = True
            try:
                webbrowser.open(self._server.url, new=2)
            except Exception:
                pass

    def stop(self) -> None:
        if self._replay is not None:
            self._replay.stop()
        if self._session is not None:
            try:
                self._session.remove_exporter(self._exporter)
            except Exception:
                pass
        self._server.stop()

    @property
    def url(self) -> str:
        return self._server.url

    def open(self) -> None:
        """Block the caller until Ctrl+C, useful for replay scripts."""
        self.start()
        print(f"[viz] open → {self.url}")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            self.stop()

    def __enter__(self) -> Visualizer:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Provider callbacks (server-bound)
    # ------------------------------------------------------------------

    def _graph_payload(self) -> dict[str, Any]:
        if self._mode == "replay":
            return getattr(self, "_fixed_graph", {"nodes": [], "edges": []})
        if self._session is None:
            return {"nodes": [], "edges": []}
        return self._session.graph.to_dict()

    def _store_payload(self) -> dict[str, Any]:
        if self._store is None:
            return {}
        try:
            return self._store.read_all()
        except Exception:
            return {}

    def _meta_payload(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "session_id": (
                getattr(self, "_fixed_session_id", None)
                if self._mode == "replay"
                else (self._session.session_id if self._session else "")
            ),
        }

    def _on_control(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._replay is None:
            return {"error": "controls only available in replay mode"}
        if action == "play":
            self._replay.play()
        elif action == "pause":
            self._replay.pause()
        elif action == "step":
            self._replay.step()
        elif action == "speed":
            self._replay.set_speed(float(body.get("speed", 1.0)))
        else:
            return {"error": f"unknown action: {action}"}
        idx, total = self._replay.progress
        return {"ok": True, "idx": idx, "total": total}
