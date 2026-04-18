"""Panel abstraction for lazybridge.gui.

Every GUI-enabled object (LazyAgent, LazyTool, LazySession, ...) registers
itself on the shared :class:`GuiServer` through a ``Panel`` subclass that
exposes:

- an ``id`` (routable in the URL),
- a ``kind`` (used to group panels in the sidebar and pick an HTML template),
- ``render_state()`` (read-only payload used to populate the inspect tab),
- ``handle_action()`` (RPC endpoint for edits and live test runs).

The panel does NOT render HTML itself — the shared page template knows how
to render each ``kind`` based on the JSON state. This keeps the server-side
code small and lets us grow the UI without touching Python.

Panels can push change notifications to SSE subscribers by calling
``self.notify()``.  The server wires up the notifier at registration time;
panels created outside of a registered server see a no-op.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

_logger = logging.getLogger(__name__)


class Panel(ABC):
    """Base class for GUI panels."""

    #: Panel kind — one of ``"agent"``, ``"tool"``, ``"session"``,
    #: ``"router"``, ``"store"``, ``"memory"``, ``"pipeline"``, ``"human"``.
    kind: str = "generic"

    #: Set by :class:`GuiServer.register` — call ``self.notify()`` to push
    #: a ``refresh`` event to every SSE subscriber.  No-op when unset.
    _notifier: Callable[[str, str | None], None] | None = None

    def notify(self, panel_id: str | None = None) -> None:
        """Tell SSE subscribers this panel's state changed.

        ``panel_id`` defaults to ``self.id`` but can be overridden to
        notify the sidebar list as a whole (pass ``None``) — useful when
        a panel's ``label`` changes, because the list is cached in the
        client until a refresh.
        """
        if self._notifier is not None:
            try:
                self._notifier("state", panel_id if panel_id is not None else self.id)
            except Exception:
                # Don't propagate — notifications are fire-and-forget — but do
                # log so genuine bugs in the downstream notifier surface in
                # debug output instead of vanishing.
                _logger.debug("Panel.notify() raised", exc_info=True)

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable identifier (used in URLs). Must be unique per server."""

    @property
    def label(self) -> str:
        """Short text shown in the sidebar."""
        return self.id

    @property
    def group(self) -> str:
        """Sidebar group name. Defaults to the capitalised ``kind``."""
        return self.kind.capitalize() + "s"

    @abstractmethod
    def render_state(self) -> dict[str, Any]:
        """Return JSON-serialisable state for the inspect/edit tab.

        Called on every ``GET /api/panel/<id>``.  Should be fast —
        expensive work belongs in :meth:`handle_action`.
        """

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``POST /api/panel/<id>/action``.

        Returns a JSON-serialisable dict.  Raise ``ValueError`` for bad
        input (becomes HTTP 400) or any other exception for unexpected
        failures (becomes HTTP 500 with the exception text).

        The base implementation rejects every action — concrete panels
        override.
        """
        raise ValueError(f"Panel {self.kind!r} does not support action {action!r}")
