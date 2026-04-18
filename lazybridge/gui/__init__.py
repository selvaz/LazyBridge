"""lazybridge.gui — Core, reusable GUI surfaces for LazyBridge objects.

First-class (not an ``ext``) because the primitives here are intended to
be reused by any future LazyBridge GUI — including the planned
whole-pipeline inspector.

Importing this package installs a ``.gui()`` method on the core classes::

    import lazybridge.gui                          # monkey-patches .gui()

    from lazybridge import LazyAgent, LazySession
    sess  = LazySession()
    agent = LazyAgent("anthropic", session=sess)
    print(agent.gui())                             # opens a browser tab, returns URL

The first ``.gui()`` call in the process starts a shared, localhost-bound,
token-gated HTTP server; subsequent calls register extra panels on the
same server so you end up with one tab listing every agent / tool /
session you have opened.

Sub-modules:

- ``human``  — browser UI that supplies an ``input_fn`` for ``HumanAgent``
  / ``SupervisorAgent`` (uses its own dedicated server; shipped).
- ``agent``  — :class:`AgentPanel` (inspect, edit, live test).
- ``tool``   — :class:`ToolPanel` (inspect, live invoke from a schema-generated form).
- ``session``— :class:`SessionPanel` (agents + store-key overview; dispatcher).

Lower-level primitives for building your own panels:

- :class:`lazybridge.gui.Panel` — base class.
- :class:`lazybridge.gui.GuiServer` — shared HTTP server.
- :func:`lazybridge.gui.get_server` — singleton accessor.
"""

from lazybridge.gui._global import GuiServer, close_server, get_server, is_running
from lazybridge.gui._install import install_gui_methods, uninstall_gui_methods
from lazybridge.gui._panel import Panel
from lazybridge.gui.agent import AgentPanel
from lazybridge.gui.human_panel import HumanInputPanel, panel_input_fn
from lazybridge.gui.pipeline import PipelinePanel
from lazybridge.gui.router import RouterPanel
from lazybridge.gui.session import SessionPanel
from lazybridge.gui.store import StorePanel
from lazybridge.gui.tool import ToolPanel

# Install .gui() methods on LazyAgent / LazyTool / LazySession as a side
# effect of importing this package. The install is idempotent.
install_gui_methods()

__all__ = [
    "GuiServer",
    "Panel",
    "AgentPanel",
    "ToolPanel",
    "PipelinePanel",
    "SessionPanel",
    "HumanInputPanel",
    "panel_input_fn",
    "RouterPanel",
    "StorePanel",
    "get_server",
    "close_server",
    "is_running",
    "install_gui_methods",
    "uninstall_gui_methods",
]
