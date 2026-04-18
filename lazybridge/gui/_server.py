"""GuiServer — shared HTTP server for lazybridge.gui panels.

One ``ThreadingHTTPServer`` hosts the whole GUI. Each registered
:class:`~lazybridge.gui._panel.Panel` appears as a sidebar entry; clicking it
fetches the panel's JSON state and renders it using the matching JS renderer
in :mod:`lazybridge.gui._templates`. ``POST /api/panel/<id>/action`` routes
edits and live test runs back to ``panel.handle_action()``.

Threading model: ``ThreadingHTTPServer`` spawns a new thread per request,
so a long-running test call (e.g. ``agent.chat(...)`` hitting the Anthropic
API) does not block other requests such as sidebar polling.
"""

from __future__ import annotations

import json
import logging
import queue
import secrets
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from lazybridge.gui._panel import Panel
from lazybridge.gui._templates import PAGE_TEMPLATE

_logger = logging.getLogger(__name__)

_STATIC_JS_PATH = (
    __import__("pathlib").Path(__file__).resolve().parent / "_static" / "app.js"
)
_STATIC_JS_CACHE: str | None = None


def _load_static_js() -> str:
    """Lazily read ``lazybridge/gui/_static/app.js`` into memory once.

    The file is static for the lifetime of the Python process — we read
    it on first request and cache the contents so each GUI session pays
    a single disk hit.
    """
    global _STATIC_JS_CACHE
    if _STATIC_JS_CACHE is None:
        try:
            _STATIC_JS_CACHE = _STATIC_JS_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            _logger.error("Failed to read %s: %s", _STATIC_JS_PATH, exc)
            _STATIC_JS_CACHE = f"// error loading app.js: {exc}"
    return _STATIC_JS_CACHE


class _Subscriber:
    """One SSE client.  Messages flow through a bounded queue; dropped
    notifications become a ``{"type": "refresh"}`` hint so the client
    re-syncs the next time it polls or reconnects."""

    __slots__ = ("q", "alive")

    def __init__(self) -> None:
        self.q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        self.alive = threading.Event()
        self.alive.set()

    def push(self, event: dict[str, Any]) -> None:
        if not self.alive.is_set():
            return
        try:
            self.q.put_nowait(event)
        except queue.Full:
            # Drop the event silently; the client's full-refetch on
            # reconnection will catch it up.
            pass


class GuiServer:
    """Shared HTTP server hosting every registered panel.

    Parameters
    ----------
    host:
        Bind address. Default ``127.0.0.1`` — do not change without
        understanding the security implications.
    port:
        TCP port, ``0`` for an OS-assigned ephemeral port.
    open_browser:
        If ``True``, opens :attr:`url` on first start via :mod:`webbrowser`.
    title:
        Browser tab title.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        title: str = "LazyBridge GUI",
    ) -> None:
        self._host = host
        self._title = title
        self._token = secrets.token_urlsafe(24)
        self._panels: dict[str, Panel] = {}
        self._panels_lock = threading.RLock()
        self._subscribers: set[_Subscriber] = set()
        self._subscribers_lock = threading.Lock()
        self._browser_opened = False
        self._closed = threading.Event()

        handler_cls = _make_handler(self)
        self._httpd = ThreadingHTTPServer((host, port), handler_cls)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="lazybridge-gui-server",
            daemon=True,
        )
        self._thread.start()

        if open_browser:
            self._maybe_open_browser()

        # SECURITY: never log the full URL — it contains the session token.
        # Log the bind address only; users that need the tokenised URL can
        # read `server.url` directly or receive it from `.gui()` return values.
        _logger.info("LazyBridge GUI server listening at http://%s:%d/", self._host, self.port)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}/?t={self._token}"

    @property
    def token(self) -> str:
        return self._token

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def register(self, panel: Panel) -> str:
        """Add or replace ``panel`` and return its panel URL."""
        with self._panels_lock:
            self._panels[panel.id] = panel
        panel._notifier = self._panel_notify  # noqa: SLF001 — friend access
        self._broadcast({"type": "list"})
        return self.url_for(panel.id)

    def unregister(self, panel_id: str) -> None:
        with self._panels_lock:
            removed = self._panels.pop(panel_id, None)
        if removed is not None:
            removed._notifier = None  # noqa: SLF001
            self._broadcast({"type": "list"})

    def get(self, panel_id: str) -> Panel | None:
        with self._panels_lock:
            return self._panels.get(panel_id)

    def panels(self) -> list[Panel]:
        with self._panels_lock:
            return list(self._panels.values())

    def url_for(self, panel_id: str) -> str:
        return f"{self.url}#panel={panel_id}"

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # Wake every SSE handler so their queues return and the
        # connections shut down cleanly.
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for sub in subs:
            sub.alive.clear()
            try:
                sub.q.put_nowait({"type": "closed"})
            except queue.Full:
                pass
        try:
            self._httpd.shutdown()
        except Exception:  # pragma: no cover
            pass
        self._httpd.server_close()

    # ------------------------------------------------------------------
    # SSE subscriber plumbing (internal, used by the request handler)
    # ------------------------------------------------------------------

    def _add_subscriber(self, sub: _Subscriber) -> None:
        with self._subscribers_lock:
            self._subscribers.add(sub)

    def _remove_subscriber(self, sub: _Subscriber) -> None:
        with self._subscribers_lock:
            self._subscribers.discard(sub)

    def _broadcast(self, event: dict[str, Any]) -> None:
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for sub in subs:
            sub.push(event)

    def _panel_notify(self, kind: str, panel_id: str | None) -> None:
        """Called by panels via ``Panel.notify()``."""
        payload: dict[str, Any] = {"type": "state"}
        if panel_id is not None:
            payload["panel_id"] = panel_id
        self._broadcast(payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_open_browser(self) -> None:
        if self._browser_opened:
            return
        self._browser_opened = True
        try:
            webbrowser.open(self.url)
        except Exception as exc:  # pragma: no cover - headless envs
            _logger.info("Could not open browser: %s", exc)


def _make_handler(server: GuiServer) -> type[BaseHTTPRequestHandler]:
    token = server.token
    title = server._title  # noqa: SLF001 — friend access by design

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _logger.debug("gui " + format, *args)

        # ------------------------------------------------------------------
        def _check_token(self) -> bool:
            query_token = None
            if "?" in self.path:
                _, _, qs = self.path.partition("?")
                for pair in qs.split("&"):
                    if pair.startswith("t="):
                        query_token = pair[2:]
                        break
            header_token = self.headers.get("X-Token")
            return secrets.compare_digest(query_token or header_token or "", token)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, body: str, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # ------------------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("/", "/index.html"):
                page = PAGE_TEMPLATE.format(
                    title=title,
                    token_json=json.dumps(token),
                    token_raw=token,  # for the <script src="..?t=TOKEN"> query
                )
                self._send_text(page)
                return
            if path == "/static/app.js":
                # External client script (audit L6).  Token-gated via the
                # same ?t= query as every other /api/* call so the URL
                # can't be fetched cross-origin from an unrelated tab.
                if not self._check_token():
                    self._send_json({"error": "unauthorized"}, status=401)
                    return
                self._send_text(
                    _load_static_js(),
                    content_type="application/javascript; charset=utf-8",
                )
                return
            if path == "/healthz":
                self._send_json({"ok": True, "panels": len(server.panels()), "closed": server.closed})
                return
            if not self._check_token():
                self._send_json({"error": "unauthorized"}, status=401)
                return
            if path == "/api/panels":
                panels = [
                    {"id": p.id, "kind": p.kind, "label": p.label, "group": p.group}
                    for p in server.panels()
                ]
                self._send_json({"panels": panels})
                return
            if path == "/api/events":
                self._serve_sse(server)
                return
            if path.startswith("/api/panel/"):
                panel_id = path[len("/api/panel/"):]
                panel = server.get(panel_id)
                if panel is None:
                    self._send_json({"error": "unknown panel"}, status=404)
                    return
                try:
                    state = panel.render_state()
                except Exception as exc:
                    _logger.exception("render_state failed for %s", panel_id)
                    self._send_json({"error": "render_state failed", "detail": str(exc)}, status=500)
                    return
                payload = {"id": panel.id, "kind": panel.kind, "label": panel.label,
                           "group": panel.group, **state}
                self._send_json(payload)
                return
            self._send_text("Not found", status=404, content_type="text/plain")

        # ------------------------------------------------------------------
        def _serve_sse(self, srv: GuiServer) -> None:
            sub = _Subscriber()
            srv._add_subscriber(sub)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-transform")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                while sub.alive.is_set():
                    try:
                        evt = sub.q.get(timeout=15)
                    except queue.Empty:
                        try:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            return
                        continue
                    if evt.get("type") == "closed":
                        return
                    body = json.dumps(evt).encode("utf-8")
                    payload = b"event: refresh\ndata: " + body + b"\n\n"
                    try:
                        self.wfile.write(payload)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
            finally:
                srv._remove_subscriber(sub)

        # ------------------------------------------------------------------
        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if not self._check_token():
                self._send_json({"error": "unauthorized"}, status=401)
                return
            if not path.startswith("/api/panel/") or not path.endswith("/action"):
                self._send_text("Not found", status=404, content_type="text/plain")
                return
            panel_id = path[len("/api/panel/"):-len("/action")]
            panel = server.get(panel_id)
            if panel is None:
                self._send_json({"error": "unknown panel"}, status=404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 10_000_000:
                self._send_json({"error": "invalid length"}, status=400)
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
                action_name = str(payload["action"])
                args = payload.get("args") or {}
                if not isinstance(args, dict):
                    raise ValueError("'args' must be an object")
            except (ValueError, KeyError, TypeError) as exc:
                self._send_json({"error": "bad payload", "detail": str(exc)}, status=400)
                return
            try:
                result = panel.handle_action(action_name, args)
            except ValueError as exc:
                self._send_json({"error": "bad action", "detail": str(exc)}, status=400)
                return
            except Exception as exc:
                _logger.exception("handle_action failed for %s/%s", panel_id, action_name)
                self._send_json(
                    {"error": "action failed", "detail": str(exc), "trace": traceback.format_exc()},
                    status=500,
                )
                return
            self._send_json(result if isinstance(result, dict) else {"result": result})

    return _Handler
