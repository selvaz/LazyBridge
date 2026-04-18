"""Install ``.gui()`` methods on LazyBridge core classes.

Called from :mod:`lazybridge.gui` at import time.  Re-installing is
idempotent.  Each installed method is a thin wrapper around
:func:`lazybridge.gui._dispatch.open_gui`, which is the single source of
truth for type → panel mapping.
"""

from __future__ import annotations

from typing import Any

_INSTALLED = False


def install_gui_methods() -> None:
    """Attach ``.gui()`` to every GUI-enabled LazyBridge core class."""
    global _INSTALLED
    if _INSTALLED:
        return

    from lazybridge.gui._dispatch import open_gui
    from lazybridge.lazy_agent import LazyAgent
    from lazybridge.lazy_router import LazyRouter
    from lazybridge.lazy_session import LazySession
    from lazybridge.lazy_store import LazyStore
    from lazybridge.lazy_tool import LazyTool

    def _agent_gui(
        self: Any,
        *,
        open_browser: bool = True,
        available_tools: list[Any] | None = None,
    ) -> str:
        """Open the agent's GUI panel on the shared server.  See :func:`open_gui`."""
        return open_gui(self, open_browser=open_browser, available_tools=available_tools)

    def _simple_gui(self: Any, *, open_browser: bool = True) -> str:
        """Open the object's GUI panel on the shared server.  See :func:`open_gui`."""
        return open_gui(self, open_browser=open_browser)

    LazyAgent.gui = _agent_gui  # type: ignore[attr-defined]
    LazyTool.gui = _simple_gui  # type: ignore[attr-defined]
    LazySession.gui = _simple_gui  # type: ignore[attr-defined]
    LazyRouter.gui = _simple_gui  # type: ignore[attr-defined]
    LazyStore.gui = _simple_gui  # type: ignore[attr-defined]

    _INSTALLED = True


def uninstall_gui_methods() -> None:
    """Remove ``.gui()`` methods — used by tests for isolation."""
    global _INSTALLED
    if not _INSTALLED:
        return
    from lazybridge.lazy_agent import LazyAgent
    from lazybridge.lazy_router import LazyRouter
    from lazybridge.lazy_session import LazySession
    from lazybridge.lazy_store import LazyStore
    from lazybridge.lazy_tool import LazyTool

    for cls in (LazyAgent, LazyTool, LazySession, LazyRouter, LazyStore):
        if hasattr(cls, "gui"):
            delattr(cls, "gui")
    _INSTALLED = False
