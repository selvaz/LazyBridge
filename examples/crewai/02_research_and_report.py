"""CrewAI canonical two-agent crew (researcher + reporting_analyst) — ported.

Original (CrewAI README):

    # agents.yaml
    researcher:
      role: "{topic} Senior Data Researcher"
      goal: "Uncover cutting-edge developments in {topic}"
      backstory: "You're a seasoned researcher..."
    reporting_analyst:
      role: "{topic} Reporting Analyst"
      goal: "Create detailed reports based on {topic} data analysis..."
      backstory: "You're a meticulous analyst..."

    # tasks.yaml
    research_task:
      description: "Conduct a thorough research about {topic}..."
      expected_output: "A list with 10 bullet points..."
      agent: researcher
    reporting_task:
      description: "Review the context you got and expand each topic..."
      expected_output: "A fully fledge reports... Formatted as markdown."
      agent: reporting_analyst
      output_file: report.md

    # crew.py
    @CrewBase
    class LatestAiDevelopmentCrew:
        @agent
        def researcher(self) -> Agent:
            return Agent(config=self.agents_config["researcher"],
                         verbose=True, tools=[SerperDevTool()])
        @agent
        def reporting_analyst(self) -> Agent:
            return Agent(config=self.agents_config["reporting_analyst"],
                         verbose=True)
        @task
        def research_task(self) -> Task: ...
        @task
        def reporting_task(self) -> Task: ...
        @crew
        def crew(self) -> Crew:
            return Crew(agents=self.agents, tasks=self.tasks,
                        process=Process.sequential, verbose=True)

    # main.py
    LatestAiDevelopmentCrew().crew().kickoff(inputs={"topic": "AI Agents"})

LazyBridge equivalent: ``Agent.chain(researcher, reporter)`` runs them
sequentially and feeds the researcher's output as the reporter's task —
exactly what CrewAI's ``Process.sequential`` does, minus the YAML, the
decorators, and the @CrewBase metaclass.
"""

from __future__ import annotations

from pathlib import Path

from lazybridge import Agent, LLMEngine

OUTPUT_FILE = Path("report.md")


def serper_search(query: str) -> str:
    """Stub web search; swap for Serper / Tavily in production."""
    return f"[stub results for {query!r}]"


def build_researcher(topic: str) -> Agent:
    system = (
        f"# Role\n{topic} Senior Data Researcher\n\n"
        f"# Goal\nUncover cutting-edge developments in {topic}\n\n"
        "# Backstory\nYou're a seasoned researcher with a knack for uncovering "
        f"the latest developments in {topic}. Known for your ability to find "
        "the most relevant information and present it in a clear and concise "
        "manner.\n\n"
        "# Task contract\n"
        f"Conduct a thorough research about {topic}. Make sure you find any "
        "interesting and relevant information given the current year is 2026.\n"
        "Expected output: a list with 10 bullet points of the most relevant "
        f"information about {topic}."
    )
    return Agent(
        engine=LLMEngine("claude-opus-4-7", system=system),
        tools=[serper_search],
        name="researcher",
        verbose=True,
    )


def build_reporting_analyst(topic: str) -> Agent:
    system = (
        f"# Role\n{topic} Reporting Analyst\n\n"
        f"# Goal\nCreate detailed reports based on {topic} data analysis "
        "and research findings.\n\n"
        "# Backstory\nYou're a meticulous analyst with a keen eye for detail. "
        "You're known for your ability to turn complex data into clear and "
        "concise reports, making it easy for others to understand and act on "
        "the information you provide.\n\n"
        "# Task contract\n"
        "Review the context you got and expand each topic into a full section "
        "for a report. Make sure the report is detailed and contains any and "
        "all relevant information.\n"
        "Expected output: a full report with the main topics, each with a full "
        "section of information. Formatted as markdown without ``` fences."
    )
    return Agent(
        engine=LLMEngine("claude-opus-4-7", system=system),
        name="reporting_analyst",
        verbose=True,
    )


def kickoff(topic: str = "AI Agents") -> str:
    crew = Agent.chain(build_researcher(topic), build_reporting_analyst(topic))
    report = crew(f"Topic: {topic}").text()

    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(f"Report written to {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    kickoff()
