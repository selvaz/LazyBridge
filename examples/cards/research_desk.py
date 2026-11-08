"""Card 1: a research desk in plain Python.

A lead agent (Claude Code, Opus) has three tools:
  - research_brief: a deterministic pipeline, two web researchers in parallel,
    then a review, then a summary written by DeepSeek over its API;
  - deep_research (Claude Code, Sonnet) and fact_check (Codex, Terra), the
    same specialists, callable directly to dig deeper.

    python research_desk.py "Should a small team start with SQLite or Postgres?"
"""

import sys

from lazybridge import (
    Agent,
    ClaudeCodeEngine,
    CodexEngine,
    CodingAgentConfig,
    LLMEngine,
    Plan,
    Step,
    from_parallel_all,
    from_start,
    from_step,
)

# Prompts
SUPPORT = "Find the strongest evidence FOR the answer you would give. Cite URLs."
CHALLENGE = "Find the strongest evidence AGAINST it and its limits. Cite URLs."
REVIEW = "Audit both research notes. Flag unsupported, weak or outdated claims."
SUMMARY = "Write a balanced brief under 200 words: verdict, reasons, caveats, URLs."
LEAD = (
    "Call research_brief once. Use deep_research or fact_check only to close "
    "a gap that changes the answer. Reply concisely, with sources."
)

# Specialists: read-only, with web search
web = CodingAgentConfig.reviewer(web=True)
sonnet = Agent(
    name="deep_research",
    description="Researches one narrow question on the web.",
    engine=ClaudeCodeEngine(model="sonnet", config=web),
)
terra = Agent(
    name="fact_check",
    description="Checks claims, dates and sources on the web.",
    engine=CodexEngine(model="terra", config=web),
)

# Writer: a token-based API model, a peer of the two coding agents
writer = Agent(name="writer", engine=LLMEngine("deepseek-flash"))

# Pipeline: research in parallel, review, summarise
research_brief = Agent(
    name="research_brief",
    description="Two-angle web research, reviewed and summarised.",
    engine=Plan(
        Step(sonnet, name="support", task=from_start, context=SUPPORT, parallel=True),
        Step(terra, name="challenge", task=from_start, context=CHALLENGE, parallel=True),
        Step(terra, name="review", task=from_start, context=[REVIEW, from_parallel_all("support")]),
        Step(
            writer,
            name="summary",
            task=from_start,
            context=[SUMMARY, from_parallel_all("support"), from_step("review")],
        ),
    ),
)

# Lead: the pipeline and the specialists are all tools
desk = Agent(
    name="research_desk",
    description="Answers a research question with a reviewed, cited brief.",
    engine=ClaudeCodeEngine(model="opus", system=LEAD, web=False),
    tools=[research_brief, sonnet, terra],
)

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Should a small team start with SQLite or Postgres?"
    result = desk(question)
    print(result.text() if result.ok else f"failed: {result.error.message}")
