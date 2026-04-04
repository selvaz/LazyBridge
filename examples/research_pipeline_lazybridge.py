"""
Research Pipeline — LazyBridge
========================================
Same pipeline as the raw SDK and LangGraph examples.

No manual loops. No explicit state. No separate clients per provider.
Provider swap = change the string "anthropic" / "openai".
Session tracking included out of the box.

Install: pip install lazybridge pydantic
"""

from pydantic import BaseModel

from lazybridge import LazyAgent, LazyContext, LazySession, LazyTool

# ── Mock tool ──────────────────────────────────────────────────────────────────

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


# ── Session and agents ─────────────────────────────────────────────────────────

sess       = LazySession(tracking="basic")
search     = LazyTool.from_function(search_company)

researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)


# ── Pipeline as tool ──────────────────────────────────────────────────────────

def run_pipeline(task: str) -> str:
    """Research companies and produce a structured report."""
    researcher.loop(task, tools=[search])
    return str(writer.json(
        "Write a structured report based on the research.",
        Report,
        context=LazyContext.from_agent(researcher),  # injects researcher's last output
    ))


pipeline = LazyTool.from_function(run_pipeline)


# ── Orchestrator ───────────────────────────────────────────────────────────────

orchestrator = LazyAgent("anthropic", name="orchestrator")
orchestrator.loop(
    "Research Stripe, Plaid, and Brex. Call the pipeline for each company.",
    tools=[pipeline],
)
