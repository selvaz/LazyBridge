"""CrewAI quickstart (single researcher) — ported to LazyBridge.

Original (CrewAI quickstart):

    # config/agents.yaml
    researcher:
      role: "{topic} Senior Data Researcher"
      goal: "Uncover cutting-edge developments in {topic}"
      backstory: "You're a seasoned researcher..."

    # config/tasks.yaml
    research_task:
      description: "Conduct thorough research about {topic}..."
      expected_output: "A markdown report with clear sections..."
      agent: researcher
      output_file: output/report.md

    # content_crew.py
    @CrewBase
    class ResearchCrew:
        agents_config = "config/agents.yaml"
        tasks_config = "config/tasks.yaml"

        @agent
        def researcher(self) -> Agent:
            return Agent(config=self.agents_config["researcher"],
                         verbose=True, tools=[SerperDevTool()])

        @task
        def research_task(self) -> Task: ...

        @crew
        def crew(self) -> Crew:
            return Crew(agents=self.agents, tasks=self.tasks,
                        process=Process.sequential, verbose=True)

    # Plus a Flow class with @start / @listen decorators that calls
    # ResearchCrew().crew().kickoff(inputs={"topic": "AI Agents"}) and writes
    # report.md.

LazyBridge equivalent: no YAML, no decorators, no @CrewBase / @agent / @task
boilerplate, no separate Crew or Flow. The "agent" is an :class:`Agent`,
the "task" is the string passed to it, and the "tool" is just a Python
function. The output file is a one-liner at the end.
"""

from __future__ import annotations

from pathlib import Path

from lazybridge import Agent, LLMEngine

OUTPUT_FILE = Path("output/report.md")


def serper_search(query: str) -> str:
    """Search the web (stub — wire to Serper / Tavily / your search API).

    The CrewAI original uses ``SerperDevTool()``. Replace this body with the
    real call once you have an API key; the rest of the example is unchanged.
    """
    return (
        f"[stub results for {query!r}] "
        "Top sources: latest agent frameworks comparisons, multi-agent "
        "patterns, observability tooling, hosted runtimes."
    )


def build_researcher(topic: str) -> Agent:
    """Mirror of the YAML role/goal/backstory — just three Python strings."""
    role = f"{topic} Senior Data Researcher"
    goal = f"Uncover cutting-edge developments in {topic}"
    backstory = (
        "You're a seasoned researcher with a knack for uncovering the latest "
        f"developments in {topic}. You find the most relevant information and "
        "present it clearly."
    )
    system = f"# Role\n{role}\n\n# Goal\n{goal}\n\n# Backstory\n{backstory}"
    return Agent(
        engine=LLMEngine("claude-opus-4-7", system=system),
        tools=[serper_search],
        name="researcher",
        verbose=True,
    )


def kickoff(topic: str = "AI Agents") -> str:
    researcher = build_researcher(topic)
    task = (
        f"Conduct thorough research about {topic}. Use web search to find "
        "current, credible information. The current year is 2026.\n\n"
        "Expected output: a markdown report with clear sections — key trends, "
        "notable tools or companies, and implications. Aim for 800-1200 words. "
        "Do not wrap the document in fenced code blocks."
    )
    report = researcher(task).text()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(f"Report written to {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    kickoff()
