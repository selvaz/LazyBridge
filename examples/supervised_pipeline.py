"""Supervised pipeline — researcher → SupervisorAgent → writer.

Non-interactive demo: the SupervisorAgent is driven by a scripted ``input_fn``
that replays three REPL commands (search, retry, continue), so this file runs
end-to-end under CI without a real terminal.

For the interactive version, remove the ``input_fn=...`` argument — the
supervisor will fall back to :func:`input` and accept commands at the prompt::

    [supervisor] > search("query")
    [supervisor] > retry researcher: add 2026 data
    [supervisor] > continue

Full walkthrough of every REPL command: docs/course/13-human-in-the-loop.md
LLM-oriented reference:                  lazy_wiki/bot/13_supervisor.md

Run::

    python examples/supervised_pipeline.py
"""

from __future__ import annotations

from lazybridge import LazyAgent, LazySession, LazyTool, SupervisorAgent


def search(query: str) -> str:
    """Search for recent papers on a topic (stubbed for the demo)."""
    return f"[search stub] top 3 papers for {query!r}: paper-a, paper-b, paper-c"


def build_pipeline() -> LazyTool:
    sess = LazySession()
    search_tool = LazyTool.from_function(search)

    researcher = LazyAgent(
        "anthropic",
        name="researcher",
        tools=[search_tool],
        session=sess,
        system="You are a concise research assistant. Call `search` to find sources, then summarise in 2 bullet points.",
    )
    writer = LazyAgent(
        "openai",
        name="writer",
        session=sess,
        system="You are a technical writer. Turn the supervisor's notes into a 3-sentence brief.",
    )

    # Scripted REPL — three commands, consumed one per prompt.
    #   1) call the search tool interactively
    #   2) re-run the researcher with feedback (agent retry)
    #   3) forward the current output to the writer
    scripted = iter(
        [
            'search("AI safety 2026")',
            "retry researcher: include the 2026 EU AI Act",
            "continue",
        ]
    )

    supervisor = SupervisorAgent(
        name="supervisor",
        tools=[search_tool],
        agents=[researcher],
        session=sess,
        input_fn=lambda prompt: next(scripted),
    )

    return LazyTool.chain(
        researcher,
        supervisor,
        writer,
        name="supervised_pipeline",
        description="Research, human-supervise with retry, write final brief.",
    )


def main() -> None:
    pipeline = build_pipeline()
    result = pipeline.run({"task": "AI safety developments in 2026"})
    print("\n=== Writer output ===")
    print(result)


if __name__ == "__main__":
    main()
