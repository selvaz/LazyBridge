"""`.gui()` demo — open a browser panel for each LazyBridge object.

Creates a small session (researcher + writer + a shared search tool),
imports ``lazybridge.gui`` to install the ``.gui()`` method on every core
class, and opens the session panel.  The session panel also auto-registers
a panel for every agent already in the session.

From the browser you can:

- Read and live-edit each agent's system prompt.
- Toggle which session-level tools each agent has access to.
- Run ``chat`` / ``loop`` / ``text`` against the real provider.
- Invoke the ``search`` tool manually with arbitrary parameters.

Run::

    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    python examples/gui_demo.py

The script keeps the server alive until you press Enter in the terminal.
"""

from __future__ import annotations

from lazybridge import LazyAgent, LazySession, LazyTool

# Importing this package installs .gui() on LazyAgent, LazyTool, LazySession.
import lazybridge.gui  # noqa: F401  — side-effect import
from lazybridge.gui import close_server


def search(query: str, limit: int = 5) -> str:
    """Search the web for a query (stubbed)."""
    return f"[search stub] top {limit} results for {query!r}"


def main() -> None:
    sess = LazySession()
    search_tool = LazyTool.from_function(search)

    researcher = LazyAgent(
        "anthropic",
        name="researcher",
        tools=[search_tool],
        session=sess,
        system="You are a terse research assistant.",
    )
    writer = LazyAgent(
        "openai",
        name="writer",
        session=sess,
        system="You are a precise technical writer.",
    )

    # Opens the session panel (which auto-registers researcher, writer, and
    # search_tool panels in the sidebar).
    url = sess.gui()
    print(f"Open: {url}")
    print("Use the sidebar to pick a panel. Tests run live against the real provider.")
    try:
        input("Press Enter to shut the GUI server down… ")
    finally:
        close_server()


if __name__ == "__main__":
    main()
