"""Process-wide singleton accessor for :class:`GuiServer`.

The user-facing ``.gui()`` methods on LazyBridge core classes call
:func:`get_server` lazily. First call spins the server up (and opens a
browser tab); subsequent calls return the same instance so every panel
ends up in the same tab.
"""

from __future__ import annotations

import threading
from typing import Any

from lazybridge.gui._server import GuiServer

_lock = threading.Lock()
_server: GuiServer | None = None


def get_server(
    *,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    title: str = "LazyBridge GUI",
) -> GuiServer:
    """Return the shared :class:`GuiServer`, creating it on first call.

    ``open_browser`` / ``host`` / ``port`` / ``title`` are honoured only
    on the first call in the process. Subsequent calls ignore them — the
    server stays the same so every panel ends up in one tab.
    """
    global _server
    with _lock:
        if _server is None or _server.closed:
            _server = GuiServer(
                host=host, port=port, open_browser=open_browser, title=title
            )
    return _server


def close_server() -> None:
    """Shut down the shared server if one is running."""
    global _server
    with _lock:
        if _server is not None:
            _server.close()
            _server = None


def is_running() -> bool:
    with _lock:
        return _server is not None and not _server.closed


def _reset_for_tests() -> None:
    """Internal: drop the singleton without shutting the old server down.

    Used by unit tests that instantiate their own :class:`GuiServer`
    instance directly and want to re-enter ``get_server`` on the next call.
    """
    global _server
    with _lock:
        if _server is not None:
            _server.close()
        _server = None


__all__: list[str] = ["GuiServer", "get_server", "close_server", "is_running"]


def __getattr__(name: str) -> Any:  # pragma: no cover - import shim
    if name == "_server":
        return _server
    raise AttributeError(name)
