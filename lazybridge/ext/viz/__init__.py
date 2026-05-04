"""viz — live & replay pipeline visualizer (alpha).

A local-only web UI that shows what is happening inside a LazyBridge
``Session`` in real time. Pulses travel along graph edges as agents
call tools, the inspector lets you read every event payload, and the
store viewer highlights writes as they happen. The same UI replays a
finished session from the SQLite event log with speed control and
step-by-step navigation.

Two-line entry::

    from lazybridge import Session
    from lazybridge.ext.viz import Visualizer

    sess = Session(db="demo.db")
    with Visualizer(sess) as viz:
        # ...run your pipeline here; the browser is already open...
        ...

Replay a finished run::

    from lazybridge.ext.viz import Visualizer
    Visualizer.replay(db="demo.db").open()

Backend is stdlib-only (``http.server`` + Server-Sent Events + a token
in the URL); the frontend loads D3.js v7 from a CDN so there is no
build step. Local-only by design — bound to ``127.0.0.1`` with an
ephemeral port.
"""

__stability__ = "alpha"
__lazybridge_min__ = "1.0.0"

from lazybridge.ext.viz.visualizer import Visualizer

__all__ = ["Visualizer"]
