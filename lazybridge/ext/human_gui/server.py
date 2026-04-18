"""WebInputServer — local HTTP-based ``input_fn`` for HumanAgent / SupervisorAgent.

Minimal stdlib-only implementation: one page, three endpoints (``/``,
``/prompt``, ``/submit``), a ``queue.Queue`` per direction, and a single
random token required on every non-root request.  The server runs on a
daemon thread so the process exits cleanly even if ``close()`` is not
called.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import secrets
import threading
import time
import webbrowser
from collections.abc import Awaitable, Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from lazybridge.ext.human_gui.templates import PAGE_TEMPLATE

_logger = logging.getLogger(__name__)


class _SentinelClosed:
    """Marker pushed onto response queue to unblock ask() on close()."""


_CLOSED = _SentinelClosed()


class WebInputServer:
    """Serves a single HTML page and routes prompts ↔ responses.

    Parameters
    ----------
    host:
        Bind address. Defaults to ``127.0.0.1`` — do not change unless you
        understand the security implications of exposing the page.
    port:
        TCP port, or ``0`` for an OS-assigned ephemeral port (default).
    open_browser:
        If ``True`` (default), opens the page via :mod:`webbrowser`.  Set
        to ``False`` for headless environments; print ``server.url`` to
        share manually.
    title:
        Browser tab title.
    poll_interval:
        Seconds between status updates logged while the input is pending.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        title: str = "LazyBridge — Human Input",
        poll_interval: float = 30.0,
    ) -> None:
        self._host = host
        self._title = title
        self._poll_interval = poll_interval
        self._token = secrets.token_urlsafe(24)

        self._prompt_lock = threading.Lock()
        self._current_prompt: dict[str, Any] | None = None  # {"seq", "prompt", "quick_commands"}
        self._current_seq = 0
        self._response_q: "queue.Queue[str | _SentinelClosed]" = queue.Queue(maxsize=1)
        self._closed = threading.Event()

        handler_cls = _make_handler(self)
        self._httpd = ThreadingHTTPServer((host, port), handler_cls)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="lazybridge-human-gui",
            daemon=True,
        )
        self._thread.start()

        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception as exc:  # pragma: no cover - browser absence
                _logger.info("Could not open browser automatically: %s", exc)

        _logger.info("Human-input web server listening at %s", self.url)

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

    def ask(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        quick_commands: list[str] | None = None,
    ) -> str:
        """Publish ``prompt`` to the page and block until a response is posted.

        Raises :class:`TimeoutError` if ``timeout`` is set and elapses, or
        :class:`RuntimeError` if the server was closed while waiting.
        """
        if self._closed.is_set():
            raise RuntimeError("WebInputServer is closed")

        with self._prompt_lock:
            self._current_seq += 1
            self._current_prompt = {
                "seq": self._current_seq,
                "prompt": prompt,
                "quick_commands": quick_commands or [],
            }
            # Drain any stale response enqueued after a previous timeout.
            try:
                while True:
                    self._response_q.get_nowait()
            except queue.Empty:
                pass

        start = time.monotonic()
        try:
            while True:
                remaining: float | None
                if timeout is None:
                    remaining = self._poll_interval
                else:
                    elapsed = time.monotonic() - start
                    if elapsed >= timeout:
                        raise TimeoutError(f"Web input timed out after {timeout}s")
                    remaining = min(self._poll_interval, timeout - elapsed)
                try:
                    resp = self._response_q.get(timeout=remaining)
                except queue.Empty:
                    _logger.debug("Still waiting for browser response at %s", self.url)
                    continue
                if isinstance(resp, _SentinelClosed):
                    raise RuntimeError("WebInputServer closed while waiting for input")
                return resp
        finally:
            with self._prompt_lock:
                self._current_prompt = None

    async def aask(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        quick_commands: list[str] | None = None,
    ) -> str:
        """Awaitable wrapper — delegates to a thread so the loop is free."""
        return await asyncio.to_thread(
            self.ask, prompt, timeout=timeout, quick_commands=quick_commands
        )

    @property
    def input_fn(self) -> Callable[[str], str]:
        """Return a callable suitable for ``HumanAgent(input_fn=...)``."""
        return lambda prompt: self.ask(prompt)

    @property
    def ainput_fn(self) -> Callable[[str], Awaitable[str]]:
        """Return an async callable suitable for ``HumanAgent(ainput_fn=...)``."""

        async def _ainput(prompt: str) -> str:
            return await self.aask(prompt)

        return _ainput

    def close(self) -> None:
        """Shut the server down and unblock any in-flight :meth:`ask` call."""
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._response_q.put_nowait(_CLOSED)
        except queue.Full:
            pass
        try:
            self._httpd.shutdown()
        except Exception:  # pragma: no cover - shutdown race
            pass
        self._httpd.server_close()

    def __enter__(self) -> "WebInputServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal — called by the HTTP handler
    # ------------------------------------------------------------------

    def _snapshot_prompt(self) -> dict[str, Any]:
        with self._prompt_lock:
            if self._current_prompt is None:
                return {"prompt": None, "seq": self._current_seq, "closed": self.closed}
            data = dict(self._current_prompt)
            data["closed"] = self.closed
            return data

    def _deliver_response(self, seq: int, response: str) -> bool:
        with self._prompt_lock:
            if self._current_prompt is None or self._current_prompt["seq"] != seq:
                return False
        try:
            self._response_q.put_nowait(response)
        except queue.Full:
            return False
        return True


def _make_handler(server: WebInputServer) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to ``server``."""

    token = server.token
    title = server._title  # noqa: SLF001 — intentional friend access

    class _Handler(BaseHTTPRequestHandler):
        # Suppress default stderr access log; route to our logger instead.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _logger.debug("human_gui " + format, *args)

        def _check_token(self) -> bool:
            # Token can be in query string (?t=...) or X-Token header.
            query_token = None
            if "?" in self.path:
                _, _, qs = self.path.partition("?")
                for pair in qs.split("&"):
                    if pair.startswith("t="):
                        query_token = pair[2:]
                        break
            header_token = self.headers.get("X-Token")
            return secrets.compare_digest(query_token or header_token or "", token)

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _text(self, body: str, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # ---- GET ------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                page = PAGE_TEMPLATE.format(title=title, token_json=json.dumps(token))
                self._text(page)
                return
            if path == "/healthz":
                self._json({"ok": True, "closed": server.closed})
                return
            if path == "/prompt":
                if not self._check_token():
                    self._json({"error": "unauthorized"}, status=401)
                    return
                self._json(server._snapshot_prompt())  # noqa: SLF001
                return
            self._text("Not found", status=404, content_type="text/plain")

        # ---- POST -----------------------------------------------------
        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path != "/submit":
                self._text("Not found", status=404, content_type="text/plain")
                return
            if not self._check_token():
                self._json({"error": "unauthorized"}, status=401)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 1_000_000:
                self._json({"error": "invalid length"}, status=400)
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
                seq = int(payload["seq"])
                response = str(payload.get("response", ""))
            except (ValueError, KeyError, TypeError):
                self._json({"error": "bad payload"}, status=400)
                return
            accepted = server._deliver_response(seq, response)  # noqa: SLF001
            self._json({"accepted": accepted})

    return _Handler


def web_input_fn(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    title: str = "LazyBridge — Human Input",
) -> Callable[[str], str]:
    """One-call factory: start a :class:`WebInputServer` and return its ``input_fn``.

    The returned callable carries a ``.server`` attribute so callers can
    shut the server down explicitly::

        fn = web_input_fn()
        agent = HumanAgent(name="reviewer", input_fn=fn)
        # ... use agent ...
        fn.server.close()
    """
    server = WebInputServer(host=host, port=port, open_browser=open_browser, title=title)

    def _fn(prompt: str) -> str:
        return server.ask(prompt)

    _fn.server = server  # type: ignore[attr-defined]
    return _fn
