"""Local HTTP server: HTML shell + JSON endpoints + SSE event stream.

Stdlib only — ``http.server.ThreadingHTTPServer`` so each long-lived
SSE connection runs on its own thread without starving short reads of
``/api/graph`` or ``/api/store``. The server is bound to
``127.0.0.1`` and gated by a URL token; it is not meant to be exposed
to a network.
"""

from __future__ import annotations

import hmac
import json
import logging
import queue as _queue
import secrets
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from lazybridge.ext.viz.exporter import EventHub

_log = logging.getLogger("lazybridge.viz.server")


def _is_client_disconnect(exc: BaseException) -> bool:
    """True for the family of exceptions raised when the browser closes
    the SSE socket — a normal control flow condition, not an error.

    On POSIX this is ``BrokenPipeError`` / ``ConnectionResetError``; on
    Windows the same scenario surfaces as ``ConnectionAbortedError`` or
    a generic ``OSError`` with ``WinError 10053`` (software-caused
    connection abort) or ``10054`` (forcibly closed by remote host).
    """
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
        return True
    if isinstance(exc, OSError):
        winerror = getattr(exc, "winerror", None)
        if winerror in (10053, 10054, 10058):
            return True
    return False


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that recognises browser-disconnect exceptions
    as a normal control flow signal rather than an error.

    Stdlib's default ``handle_error`` prints every exception, including
    expected disconnects, to stderr — fine for a generic web server,
    fatally noisy for an SSE-heavy app where every closed tab triggers
    one. We log disconnects at DEBUG so they remain observable, and
    delegate any other exception to the stdlib handler unchanged.
    """

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if exc is not None and _is_client_disconnect(exc):
            _log.debug("client %s disconnected: %s", client_address, exc)
            return
        super().handle_error(request, client_address)


_STATIC_DIR = Path(__file__).parent / "static"

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
}

# Server-level keep-alive cadence for SSE. Browsers close idle
# connections aggressively (some proxies at 30s) so a heartbeat
# ensures the EventSource stays open during quiet pipeline phases.
_SSE_HEARTBEAT_S = 12.0


class _Handler(BaseHTTPRequestHandler):
    # Filled in by VizServer before the server starts
    hub: EventHub
    token: str
    graph_provider: Callable[[], dict[str, Any]]
    store_provider: Callable[[], dict[str, Any]]
    meta_provider: Callable[[], dict[str, Any]]
    control_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None

    server_version = "LazyBridgeViz/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Default BaseHTTPRequestHandler logging spams stderr with a
        # line per request which fights with the server's own console
        # output. The Visualizer prints a single banner; per-request
        # noise belongs in a dev-only debug switch we don't have yet.
        return

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_ok(self, params: dict[str, list[str]]) -> bool:
        supplied = (params.get("t") or [""])[0]
        if not supplied:
            supplied = self.headers.get("X-Token", "")
        return bool(supplied) and hmac.compare_digest(supplied, self.token)

    def _deny(self, code: int = 401, msg: str = "unauthorised") -> None:
        body = msg.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _send_json(self, payload: Any, *, code: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, name: str) -> None:
        # Resolve under _STATIC_DIR and reject any path that escapes it.
        # ``Path('/static/../secrets').resolve()`` would otherwise let a
        # crafted URL read files outside the package directory.
        try:
            target = (_STATIC_DIR / name).resolve(strict=True)
            target.relative_to(_STATIC_DIR.resolve())
        except (FileNotFoundError, ValueError, OSError):
            self._send_json({"error": "not found"}, code=404)
            return
        body = target.read_bytes()
        ctype = _MIME.get(target.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # GET routes
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        url = urlparse(self.path)
        params = parse_qs(url.query)
        path = url.path

        if path == "/healthz":
            self._send_json({"ok": True})
            return

        # Index + static assets do not enforce auth: the token travels
        # in the URL hash and is consumed by app.js for subsequent
        # fetch() / EventSource calls. The static files contain no
        # session data, so this is intentional.
        if path == "/":
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
            return

        if not self._auth_ok(params):
            self._deny()
            return

        if path == "/api/meta":
            self._send_json(self.meta_provider())
            return
        if path == "/api/graph":
            self._send_json(self.graph_provider())
            return
        if path == "/api/store":
            self._send_json(self.store_provider())
            return
        if path == "/api/snapshot":
            self._send_json({"events": self.hub.snapshot(), "seq": self.hub.seq})
            return
        if path == "/api/events":
            self._stream_events()
            return

        self._send_json({"error": "not found", "path": path}, code=404)

    # ------------------------------------------------------------------
    # POST routes — replay controls
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        url = urlparse(self.path)
        params = parse_qs(url.query)
        if not self._auth_ok(params):
            self._deny()
            return
        if not url.path.startswith("/api/control/"):
            self._send_json({"error": "not found"}, code=404)
            return
        action = url.path[len("/api/control/") :]
        if not self.control_handler:
            self._send_json({"error": "controls not enabled"}, code=400)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, code=400)
            return
        try:
            result = self.control_handler(action, body if isinstance(body, dict) else {})
        except Exception as exc:  # surface controller errors to the UI
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, code=500)
            return
        self._send_json(result or {"ok": True})

    # ------------------------------------------------------------------
    # SSE
    # ------------------------------------------------------------------

    def _stream_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.end_headers()

        sub = self.hub.subscribe(replay_recent=False)
        try:
            self._sse_write_comment("connected")
            last_beat = time.monotonic()
            while True:
                timeout = max(0.5, _SSE_HEARTBEAT_S - (time.monotonic() - last_beat))
                try:
                    event = sub.get(timeout=timeout)
                except _queue.Empty:
                    self._sse_write_comment("heartbeat")
                    last_beat = time.monotonic()
                    continue
                if event.get("event_type") == "_closed":
                    break
                self._sse_write_event(event)
                if (time.monotonic() - last_beat) > _SSE_HEARTBEAT_S:
                    self._sse_write_comment("heartbeat")
                    last_beat = time.monotonic()
        except BaseException as exc:
            if not _is_client_disconnect(exc):
                raise
            # Browser tab closed — normal control flow.
        finally:
            self.hub.unsubscribe(sub)

    def _sse_write_event(self, event: dict[str, Any]) -> None:
        try:
            payload = json.dumps(event, default=str)
        except (TypeError, ValueError):
            payload = json.dumps({"event_type": "encode_error", "_seq": event.get("_seq")})
        self.wfile.write(b"event: lb\n")
        self.wfile.write(b"data: ")
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.write(b"\n\n")
        self.wfile.flush()

    def _sse_write_comment(self, comment: str) -> None:
        self.wfile.write(f": {comment}\n\n".encode())
        self.wfile.flush()


class VizServer:
    """ThreadingHTTPServer wrapper with provider-injection wiring."""

    def __init__(
        self,
        hub: EventHub,
        *,
        graph_provider: Callable[[], dict[str, Any]],
        store_provider: Callable[[], dict[str, Any]],
        meta_provider: Callable[[], dict[str, Any]],
        control_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
    ) -> None:
        self.hub = hub
        self.token = token or secrets.token_urlsafe(24)
        # Build a per-server Handler subclass so multiple Visualizers
        # in one process don't clobber each other's class attributes.
        handler_cls = type(
            "_BoundHandler",
            (_Handler,),
            {
                "hub": hub,
                "token": self.token,
                "graph_provider": staticmethod(graph_provider),
                "store_provider": staticmethod(store_provider),
                "meta_provider": staticmethod(meta_provider),
                "control_handler": staticmethod(control_handler) if control_handler else None,
            },
        )
        self._server = _QuietThreadingHTTPServer((host, port), handler_cls)
        self._server.daemon_threads = True
        self.host, self.port = self._server.server_address[:2]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/#t={self.token}"

    def start(self) -> None:
        if self._thread is not None:
            return
        t = threading.Thread(target=self._server.serve_forever, name="lazybridge-viz", daemon=True)
        t.start()
        self._thread = t

    def stop(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.hub.close()
