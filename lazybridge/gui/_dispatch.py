"""Type-dispatched ``open_gui(obj)`` helper.

An explicit alternative to the monkey-patched ``.gui()`` method.  Picks
the correct :class:`Panel` subclass based on the argument's type and
registers it on the shared :class:`GuiServer`.  Useful when the caller
wants mypy-visible call sites or does not want to rely on import-time
monkey-patching.
"""

from __future__ import annotations

from typing import Any

from lazybridge.gui._global import get_server


def open_gui(obj: Any, *, open_browser: bool = True, **kwargs: Any) -> str:
    """Open a GUI panel for any supported LazyBridge object.

    Parameters
    ----------
    obj:
        A :class:`LazyAgent`, :class:`LazyTool`, :class:`LazySession`,
        :class:`LazyRouter`, or :class:`LazyStore`.
    open_browser:
        Open a browser tab on first call.  Ignored if the shared server
        is already running.
    **kwargs:
        Forwarded to the underlying panel constructor when relevant
        (currently only ``available_tools=`` for ``LazyAgent``).

    Returns
    -------
    str
        The URL of the panel inside the shared tab.
    """
    from lazybridge.gui.agent import AgentPanel
    from lazybridge.gui.pipeline import PipelinePanel, is_pipeline_tool
    from lazybridge.gui.router import RouterPanel
    from lazybridge.gui.session import SessionPanel
    from lazybridge.gui.store import StorePanel
    from lazybridge.gui.tool import ToolPanel
    from lazybridge.lazy_agent import LazyAgent
    from lazybridge.lazy_router import LazyRouter
    from lazybridge.lazy_session import LazySession
    from lazybridge.lazy_store import LazyStore
    from lazybridge.lazy_tool import LazyTool

    server = get_server(open_browser=open_browser)

    if isinstance(obj, LazyAgent):
        return server.register(AgentPanel(obj, **kwargs))
    if isinstance(obj, LazyTool):
        panel = PipelinePanel(obj) if is_pipeline_tool(obj) else ToolPanel(obj)
        return server.register(panel)
    if isinstance(obj, LazySession):
        url = server.register(SessionPanel(obj))
        for agent in list(getattr(obj, "_agents", []) or []):
            server.register(AgentPanel(agent))
            for tool in list(getattr(agent, "tools", None) or []):
                sub = PipelinePanel(tool) if is_pipeline_tool(tool) else ToolPanel(tool)
                server.register(sub)
        store = getattr(obj, "store", None)
        if store is not None:
            server.register(StorePanel(store, label=f"session store · {obj.id[:8]}"))
        return url
    if isinstance(obj, LazyRouter):
        return server.register(RouterPanel(obj))
    if isinstance(obj, LazyStore):
        return server.register(StorePanel(obj))

    raise TypeError(
        f"open_gui() does not know how to panelize {type(obj).__name__!r}. "
        "Supported: LazyAgent, LazyTool, LazySession, LazyRouter, LazyStore."
    )
