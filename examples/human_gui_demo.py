"""Human-in-the-loop with a browser UI instead of stdin.

Uses ``lazybridge.gui.panel_input_fn`` to wire a ``researcher →
SupervisorAgent → writer`` chain into the shared LazyBridge GUI: the
supervisor's REPL appears as a panel in the same browser tab as the
agents and tools.  No TTY required, no extra dependencies.

Run::

    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    python examples/human_gui_demo.py

The script will print a URL and open a browser tab; each REPL prompt appears
on the page with the researcher's output, quick-command chips, and a textarea.
Type (or click a chip) and hit Submit.  Typical session:

    > search("AI safety 2026")          # call a tool
    > retry researcher: add 2026 data   # re-run the researcher with feedback
    > continue                          # forward output to the writer

Close the tab or interrupt with Ctrl+C to stop.
"""

from __future__ import annotations

from lazybridge import LazyAgent, LazySession, LazyTool, SupervisorAgent
from lazybridge.gui import panel_input_fn


def search(query: str) -> str:
    """Search for recent papers on a topic (stubbed for the demo)."""
    return f"[search stub] top 3 papers for {query!r}: paper-a, paper-b, paper-c"


def main() -> None:
    sess = LazySession()
    search_tool = LazyTool.from_function(search)

    researcher = LazyAgent(
        "anthropic",
        name="researcher",
        tools=[search_tool],
        session=sess,
        system="You are a concise research assistant. Call `search` and summarise in 2 bullet points.",
    )
    writer = LazyAgent(
        "openai",
        name="writer",
        session=sess,
        system="You are a technical writer. Turn the supervisor's notes into a 3-sentence brief.",
    )

    fn = panel_input_fn(name="supervisor")
    # panel_input_fn registers on the shared GUI server. Its URL is the
    # server URL; print it so headless users can copy-paste.
    from lazybridge.gui import get_server
    print(f"Open this URL if the browser did not launch: {get_server().url}")

    supervisor = SupervisorAgent(
        name="supervisor",
        tools=[search_tool],
        agents=[researcher],
        session=sess,
        input_fn=fn,
    )

    pipeline = LazyTool.chain(
        researcher,
        supervisor,
        writer,
        name="supervised_pipeline",
        description="Research, human-supervise via browser, write final brief.",
    )

    try:
        result = pipeline.run({"task": "AI safety developments in 2026"})
        print("\n=== Writer output ===")
        print(result)
    finally:
        fn.panel.close()


if __name__ == "__main__":
    main()
