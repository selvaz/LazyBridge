"""
Research Pipeline — LangGraph
==============================
Same pipeline as the raw SDK example, built with LangGraph.
create_react_agent handles the tool loop. with_structured_output handles parsing.
State must be declared explicitly as a TypedDict (LangGraph requirement).

Install: pip install langgraph langchain-anthropic langchain-openai pydantic
"""

from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

# ── Mock tool ──────────────────────────────────────────────────────────────────

@tool
def search_company(name: str) -> str:
    """Search company information by name."""
    db = {
        "Stripe": "Payments infra, founded 2010, valuation $65 B",
        "Plaid":  "Banking data network, founded 2013, valuation $13 B",
        "Brex":   "Corporate cards & spend management, founded 2017, valuation $12 B",
    }
    return db.get(name, f"No data for {name}")


class Report(BaseModel):
    title: str
    executive_summary: str
    companies: list[str]


# ── Graph state (mandatory TypedDict in LangGraph) ────────────────────────────

class PipelineState(TypedDict):
    task: str
    research: str
    report: Report | None


# ── Nodes ──────────────────────────────────────────────────────────────────────

_researcher = create_react_agent(
    ChatAnthropic(model="claude-opus-4-6"),
    tools=[search_company],
)


def researcher_node(state: PipelineState) -> dict:
    result = _researcher.invoke({"messages": [("user", state["task"])]})
    return {"research": result["messages"][-1].content}


def writer_node(state: PipelineState) -> dict:
    llm    = ChatOpenAI(model="gpt-4o").with_structured_output(Report)
    report = llm.invoke(f"Write a structured report based on:\n\n{state['research']}")
    return {"report": report}


# ── Graph ──────────────────────────────────────────────────────────────────────

graph = StateGraph(PipelineState)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)
graph.set_entry_point("researcher")
pipeline = graph.compile()


# ── Pipeline as tool for the orchestrator ─────────────────────────────────────

def run_pipeline(task: str) -> str:
    result = pipeline.invoke({"task": task, "research": "", "report": None})
    return result["report"].model_dump_json()


pipeline_tool = StructuredTool.from_function(
    run_pipeline,
    name="research_pipeline",
    description="Research companies and produce a structured report.",
)

# ── Orchestrator ───────────────────────────────────────────────────────────────

orchestrator = create_react_agent(
    ChatAnthropic(model="claude-opus-4-6"),
    tools=[pipeline_tool],
)

result = orchestrator.invoke({
    "messages": [("user", "Research Stripe, Plaid, and Brex. Call the pipeline for each company.")]
})
print(result["messages"][-1].content)
