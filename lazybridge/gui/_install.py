"""Install ``.gui()`` methods on LazyBridge core classes.

Called from :mod:`lazybridge.gui` at import time.  Re-installing is
idempotent — repeated imports don't stack methods.
"""

from __future__ import annotations

from typing import Any

_INSTALLED = False


def install_gui_methods() -> None:
    """Attach ``.gui()`` to ``LazyAgent`` / ``LazyTool`` / ``LazySession``."""
    global _INSTALLED
    if _INSTALLED:
        return

    from lazybridge.gui._global import get_server
    from lazybridge.gui.agent import AgentPanel
    from lazybridge.gui.pipeline import PipelinePanel, is_pipeline_tool
    from lazybridge.gui.session import SessionPanel
    from lazybridge.gui.tool import ToolPanel
    from lazybridge.lazy_agent import LazyAgent
    from lazybridge.lazy_session import LazySession
    from lazybridge.lazy_tool import LazyTool

    def _agent_gui(
        self: Any,
        *,
        open_browser: bool = True,
        available_tools: list[Any] | None = None,
    ) -> str:
        """Open the agent's GUI panel and return its URL.

        Parameters
        ----------
        open_browser:
            Open a browser tab on first call (default).  Ignored if the
            shared server is already running.
        available_tools:
            Explicit tool-scope override.  When omitted, the panel shows
            every tool currently bound to any agent in the same session.
        """
        server = get_server(open_browser=open_browser)
        panel = AgentPanel(self, available_tools=available_tools)
        return server.register(panel)

    def _tool_gui(self: Any, *, open_browser: bool = True) -> str:
        """Open the tool's GUI panel and return its URL.

        Pipeline tools (``LazyTool.chain`` / ``LazyTool.parallel``) get a
        richer :class:`PipelinePanel` showing the topology; function-backed
        tools and ``from_agent`` tools get the standard :class:`ToolPanel`.
        """
        server = get_server(open_browser=open_browser)
        panel = PipelinePanel(self) if is_pipeline_tool(self) else ToolPanel(self)
        return server.register(panel)

    def _session_gui(self: Any, *, open_browser: bool = True) -> str:
        """Open the session's GUI panel and return its URL.

        Also registers a panel for every agent already in the session so
        they appear in the sidebar immediately.
        """
        server = get_server(open_browser=open_browser)
        session_url = server.register(SessionPanel(self))
        for agent in list(getattr(self, "_agents", []) or []):
            server.register(AgentPanel(agent))
            for tool in list(getattr(agent, "tools", None) or []):
                sub_panel = PipelinePanel(tool) if is_pipeline_tool(tool) else ToolPanel(tool)
                server.register(sub_panel)
        return session_url

    LazyAgent.gui = _agent_gui  # type: ignore[attr-defined]
    LazyTool.gui = _tool_gui  # type: ignore[attr-defined]
    LazySession.gui = _session_gui  # type: ignore[attr-defined]

    _INSTALLED = True


def uninstall_gui_methods() -> None:
    """Remove ``.gui()`` methods — used by tests for isolation."""
    global _INSTALLED
    if not _INSTALLED:
        return
    from lazybridge.lazy_agent import LazyAgent
    from lazybridge.lazy_session import LazySession
    from lazybridge.lazy_tool import LazyTool

    for cls in (LazyAgent, LazyTool, LazySession):
        if hasattr(cls, "gui"):
            delattr(cls, "gui")
    _INSTALLED = False
