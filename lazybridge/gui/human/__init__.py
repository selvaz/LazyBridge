"""lazybridge.gui.human — Legacy dedicated-port browser input for HumanAgent / SupervisorAgent.

.. deprecated::
    Prefer :func:`lazybridge.gui.panel_input_fn`, which registers the same
    human-input UI as a panel on the shared :class:`lazybridge.gui.GuiServer`
    so the prompt shows up in the single LazyBridge-GUI tab alongside your
    agents and tools.  This module runs its own dedicated port and its own
    browser tab; it stays available for users that want *only* human input
    without the rest of the GUI stack.

Quick start (legacy)::

    from lazybridge import SupervisorAgent
    from lazybridge.gui.human import web_input_fn          # emits DeprecationWarning

    fn = web_input_fn()
    supervisor = SupervisorAgent(name="sup", input_fn=fn)
    fn.server.close()

Preferred replacement::

    from lazybridge import SupervisorAgent
    from lazybridge.gui import panel_input_fn

    fn = panel_input_fn(name="sup")
    supervisor = SupervisorAgent(name="sup", input_fn=fn)
    fn.panel.close()
"""

import warnings

from lazybridge.gui.human.server import WebInputServer, web_input_fn

warnings.warn(
    "lazybridge.gui.human is deprecated in favour of "
    "lazybridge.gui.panel_input_fn (shared-server UI, same sidebar as "
    "agents/tools). This module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["WebInputServer", "web_input_fn"]
