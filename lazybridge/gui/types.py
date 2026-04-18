"""Typing helpers for :mod:`lazybridge.gui`.

Monkey-patched methods are invisible to type checkers.  Two ways to get a
typed call site are provided:

1. :func:`lazybridge.gui.open_gui` — fully-typed, returns ``str``, no
   monkey-patching required.
2. The :class:`GuiEnabled` protocol below — ``cast``-able wrapper for when
   you want to keep writing ``obj.gui()`` with type-checker support.

Example::

    from typing import cast
    from lazybridge.gui.types import GuiEnabled

    url = cast(GuiEnabled, agent).gui()          # mypy sees .gui() now
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GuiEnabled(Protocol):
    """Structural type for any object whose ``.gui()`` has been installed."""

    def gui(self, *, open_browser: bool = True) -> str: ...
