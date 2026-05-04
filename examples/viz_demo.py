"""Pipeline Visualizer demo — a small chain of three agents that calls
two tools, all wrapped in a live :class:`Visualizer` so you can watch
the data flow in your browser.

Run::

    python examples/viz_demo.py

The browser tab opens automatically on a local URL. To replay the
recorded run later::

    python -c "from lazybridge.ext.viz import Visualizer; \\
               Visualizer.replay('examples/viz_demo.db').open()"
"""

from __future__ import annotations

import time

from lazybridge import Agent, Session, Tool
from lazybridge.ext.viz import Visualizer

DB = "examples/viz_demo.db"


def search(query: str) -> str:
    """Stub web search tool — returns a fake result with a short delay."""
    time.sleep(0.4)
    return f"[search] results for '{query}': 5 hits, top is example.com"


def summarise(text: str) -> str:
    """Stub summariser tool that pretends to compress text."""
    time.sleep(0.3)
    return f"[summary] {text[:120]}…"


def main() -> None:
    sess = Session(db=DB, console=False)

    researcher = Agent(
        "claude-haiku-4-5",
        name="researcher",
        tools=[Tool.from_function(search)],
        session=sess,
        system="Find facts. Cite sources.",
    )
    analyst = Agent(
        "claude-haiku-4-5",
        name="analyst",
        tools=[Tool.from_function(summarise)],
        session=sess,
        system="Summarise findings.",
    )
    writer = Agent(
        "claude-haiku-4-5",
        name="writer",
        session=sess,
        system="Write a short brief.",
    )
    pipeline = Agent.chain(researcher, analyst, writer)

    with Visualizer(sess) as viz:
        print(f"[viz] open → {viz.url}")
        print("[viz] running pipeline…")
        envelope = pipeline("Brief me on the state of fusion energy in 2026.")
        print("[viz] pipeline done")
        print("--- result ---")
        print(envelope.text())
        print("---")
        print("[viz] press Ctrl+C to stop the server")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
