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

from lazybridge import Agent, LLMEngine, Session, Tool
from lazybridge.ext.viz import Visualizer

DB = "examples/viz_demo.db"


def search(query: str) -> str:
    """Stub web search tool — returns a fake result with a short delay."""
    time.sleep(0.4)
    return f"[search] results for '{query}': 5 hits, top is example.com"


def summarise(text: str) -> str:
    """Stub summariser tool that pretends to compress text."""
    time.sleep(0.3)
    return f"[summary] {text[:120]}..."


def main() -> None:
    sess = Session(db=DB, console=False)

    researcher = Agent(
        engine=LLMEngine("claude-haiku-4-5", system="Find facts. Cite sources."),
        tools=[Tool(search)],
        name="researcher",
        session=sess,
    )
    analyst = Agent(
        engine=LLMEngine("claude-haiku-4-5", system="Summarise findings."),
        tools=[Tool(summarise)],
        name="analyst",
        session=sess,
    )
    writer = Agent(
        engine=LLMEngine("claude-haiku-4-5", system="Write a short brief."),
        name="writer",
        session=sess,
    )
    pipeline = Agent.chain(researcher, analyst, writer)

    with Visualizer(sess) as viz:
        print(f"[viz] open -> {viz.url}")
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
