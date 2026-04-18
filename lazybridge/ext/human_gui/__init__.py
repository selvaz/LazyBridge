"""lazybridge.ext.human_gui — Optional browser-based I/O for HumanAgent / SupervisorAgent.

Stdlib-only. Provides a drop-in ``input_fn`` that opens a local web page
instead of reading from stdin, so a human can review the previous output,
pick commands (``continue``, ``retry <agent>``, tool calls), and submit a
response from the browser.

Quick start::

    from lazybridge import SupervisorAgent, LazyTool
    from lazybridge.ext.human_gui import web_input_fn

    fn = web_input_fn()                              # opens a browser tab
    supervisor = SupervisorAgent(name="sup", input_fn=fn)
    # ... use supervisor normally; every prompt is answered in the browser.

    fn.server.close()                                # optional: shutdown when done

Works for both ``HumanAgent`` and ``SupervisorAgent`` because both accept
``input_fn=``.  No extra dependencies — everything is stdlib (http.server,
threading, queue, webbrowser, secrets).

Security: the server binds to ``127.0.0.1`` by default and requires a
random token on every request.  It is intended for local developer use,
not for production exposure.
"""

from lazybridge.ext.human_gui.server import WebInputServer, web_input_fn

__all__ = ["WebInputServer", "web_input_fn"]
