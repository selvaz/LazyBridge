"""
Investment Research Platform — Four Variants
=============================================
The same investment research workflow in four different LazyBridgeFramework patterns.
No def, no loops — only agent declarations and framework composition.

Architecture map
----------------

  Variant 1 — Three-Tier Orchestrator                    (LLM-directed)
    orchestrator (Anthropic)
    ├── market_intelligence_tool   ← parallel, 3 Google agents each own WEB_SEARCH
    └── analysis_chain_tool        ← chain, agents own output_schema

  Variant 2 — Nested Pipeline, no orchestrator          (pipeline-directed)
    full_pipeline (chain)
    ├── market_tool    ← LazyTool wrapping the parallel session
    ├── risk_analyst   ← receives market_tool output as task, emits RiskProfile
    └── report_writer  ← receives risk_analyst context, emits InvestmentReport

  Variant 3 — Fan-Out then Synthesize                   (mixed)
    full_pipeline (chain)
    ├── generalist          ← Anthropic + DB tool, broad sweep
    ├── sector_intel_tool   ← parallel, 3 specialists each own their prompt
    └── synthesizer         ← OpenAI, final report from combined briefings

  Variant 4 — Fully Structured, Multi-Provider Pipeline (typed end-to-end)
    pipeline (chain)
    ├── market_researcher  output_schema=MarketBriefing   (Google + WEB_SEARCH)
    ├── risk_analyst       output_schema=RiskProfile      (Anthropic)
    └── report_writer      output_schema=InvestmentReport (OpenAI)

Key features demonstrated
-------------------------
- output_schema on the agent: chain/parallel call json()/ajson() automatically
- native_tools on the agent, not the pipeline
- tools on the agent for custom callables
- LazyTool as participant inside a chain (Variant 2)
- Fan-out with generalist → parallel specialists → chain synthesis (Variant 3)
- Fully typed Pydantic pipeline without a single def (Variant 4)

Run: python investment_research_platform.py [1|2|3|4]
Install:  pip install lazybridgeframework pydantic
API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
"""

from __future__ import annotations

import sys
from pydantic import BaseModel, Field

from lazybridgeframework import (
    LazyAgent,
    LazySession,
    LazyTool,
    NativeTool,
)

TASK = (
    "Conduct a full investment analysis for Q2 2025. "
    "Focus on AI infrastructure, the energy transition, and CRISPR therapeutics."
)

# ── Shared Pydantic schemas ────────────────────────────────────────────────

class MarketBriefing(BaseModel):
    sectors: list[str] = Field(description="Sectors analysed")
    summaries: dict[str, str] = Field(description="Key findings per sector")
    overall_outlook: str = Field(description="One-sentence macro outlook")


class SectorRisk(BaseModel):
    sector: str
    risk_level: str = Field(description="low | medium | high | very_high")
    key_risks: list[str] = Field(description="Top 3 risks")
    opportunity_score: int = Field(description="1-10 opportunity score")


class RiskProfile(BaseModel):
    overall_sentiment: str = Field(description="bullish | neutral | bearish")
    sectors: list[SectorRisk]
    recommended_action: str = Field(description="invest | hold | avoid | monitor")
    rationale: str


class InvestmentReport(BaseModel):
    title: str
    executive_summary: str
    top_opportunities: list[str] = Field(description="Top 3 opportunities")
    top_risks: list[str] = Field(description="Top 3 risks to monitor")
    sector_allocations: dict[str, str] = Field(description="% allocation per sector")
    conclusion: str


# ── Mock DB tool (replaces live DB in examples) ───────────────────────────

def fetch_sector_data(sector: str) -> str:
    """Fetch latest sector data from the research database."""
    db = {
        "tech":    "AI capex up 40% YoY; NVIDIA, TSMC beating estimates; cloud margins expanding",
        "energy":  "Solar LCOE at new lows; grid battery deployments +120%; oil demand plateauing",
        "biotech": "CRISPR IND filings at record high; GLP-1 market expanding; biosimilar pressure",
    }
    return db.get(sector.lower(), f"No data for sector: {sector}")


# =============================================================================
# VARIANT 1 — Three-Tier Orchestrator (LLM-directed)
# =============================================================================
#
# An LLM orchestrator sees two black-box tools and decides how to use them.
# All agent capabilities (native tools, output schemas) are declared at
# construction time — the pipeline and orchestrator know nothing about them.
# =============================================================================

def run_variant_1() -> None:
    print("\n── Variant 1: Three-Tier Orchestrator ──\n")

    # Layer 1: three parallel Google researchers, each owns WEB_SEARCH
    market_sess = LazySession(tracking="basic")
    LazyAgent("google", name="tech_researcher",    session=market_sess,
              native_tools=[NativeTool.WEB_SEARCH],
              system="Technology analyst — AI, semiconductors, cloud. "
                     "Use web search and produce a concise sector briefing.")
    LazyAgent("google", name="energy_researcher",  session=market_sess,
              native_tools=[NativeTool.WEB_SEARCH],
              system="Energy analyst — renewables, oil & gas, grid. "
                     "Use web search and produce a concise sector briefing.")
    LazyAgent("google", name="biotech_researcher", session=market_sess,
              native_tools=[NativeTool.WEB_SEARCH],
              system="Biotech analyst — pharma, genomics, medical devices. "
                     "Use web search and produce a concise sector briefing.")

    market_intel_tool = market_sess.as_tool(
        "market_intelligence",
        "Live parallel web research across tech, energy, and biotech. "
        "Returns a combined briefing from three specialist analysts.",
        mode="parallel",
    )

    # Layer 2: chain — agents own their output_schema, no def needed
    analysis_sess = LazySession(tracking="basic")
    LazyAgent("anthropic", name="risk_analyst",  session=analysis_sess,
              output_schema=RiskProfile,
              system="Senior risk analyst. Given market research, "
                     "produce a structured risk assessment.")
    LazyAgent("openai",    name="report_writer", session=analysis_sess,
              output_schema=InvestmentReport,
              system="CFA-level report writer. Given a risk profile, "
                     "produce a comprehensive structured investment report.")

    analysis_chain_tool = analysis_sess.as_tool(
        "analysis_chain",
        "Risk assessment → structured investment report (Anthropic → OpenAI).",
        mode="chain",
    )

    # Layer 3: orchestrator sees only two black-box tools
    orchestrator = LazyAgent(
        "anthropic", name="orchestrator",
        system="Investment strategy coordinator. Use market_intelligence to gather "
               "live sector data, then pass those findings to analysis_chain to "
               "produce the final report. Present the key conclusions clearly.",
    )
    orchestrator.loop(TASK, tools=[market_intel_tool, analysis_chain_tool])


# =============================================================================
# VARIANT 2 — Nested Pipeline, no orchestrator (pipeline-directed)
# =============================================================================
#
# A parallel LazyTool is the first participant in a chain. Its output becomes
# the risk_analyst's task directly. No LLM orchestrator, no Python glue.
#
#   [LazyTool → LazyAgent(RiskProfile) → LazyAgent(InvestmentReport)]
# =============================================================================

def run_variant_2() -> None:
    print("\n── Variant 2: Nested Pipeline (no orchestrator) ──\n")

    # Parallel tier
    market_sess = LazySession(tracking="basic")
    for name, system in [
        ("tech_researcher",
         "Technology analyst — AI, semiconductors, cloud. Web search, concise briefing."),
        ("energy_researcher",
         "Energy analyst — renewables, oil & gas, grid. Web search, concise briefing."),
        ("biotech_researcher",
         "Biotech analyst — pharma, genomics, medical devices. Web search, concise briefing."),
    ]:
        LazyAgent("google", name=name, session=market_sess,
                  native_tools=[NativeTool.WEB_SEARCH], system=system)

    market_tool = market_sess.as_tool(
        "market_research", "Parallel web research across tech, energy, and biotech.",
        mode="parallel",
    )

    # Sequential tier — output_schema drives automatic json() calls
    analysis_sess = LazySession(tracking="basic")
    risk_analyst  = LazyAgent("anthropic", name="risk_analyst",  session=analysis_sess,
                               output_schema=RiskProfile,
                               system="Senior risk analyst. Produce a structured risk profile.")
    report_writer = LazyAgent("openai",    name="report_writer", session=analysis_sess,
                               output_schema=InvestmentReport,
                               system="CFA report writer. Produce a structured investment report.")

    # Full pipeline: LazyTool → LazyAgent → LazyAgent, zero extra code
    full_pipeline = LazySession(tracking="basic").as_tool(
        "full_investment_pipeline",
        "End-to-end: parallel web research → risk assessment → structured report.",
        mode="chain",
        participants=[market_tool, risk_analyst, report_writer],
    )

    result = full_pipeline.run({"task": TASK})
    print(result)


# =============================================================================
# VARIANT 3 — Fan-Out then Synthesize (mixed)
# =============================================================================
#
# A generalist researcher sweeps the DB, then three sector specialists analyse
# in parallel (each tailored), then a synthesizer chains after the combined
# briefings. All wired as a single chain — no def, no loop.
#
#   [generalist → sector_intel_tool → synthesizer]
# =============================================================================

def run_variant_3() -> None:
    print("\n── Variant 3: Fan-Out then Synthesize ──\n")

    db_tool = LazyTool.from_function(fetch_sector_data)

    # Step 1: generalist does a broad sweep with the DB tool
    generalist = LazyAgent(
        "anthropic", name="generalist",
        tools=[db_tool],
        system="Generalist researcher. Fetch tech, energy, and biotech data "
               "and summarise the key cross-sector investment themes.",
    )

    # Step 2: three sector specialists run in parallel on the generalist's findings
    specialist_sess = LazySession(tracking="basic")
    LazyAgent("anthropic", name="tech_specialist",   session=specialist_sess,
              system="Deep-tech specialist. Analyse AI and semiconductor opportunities.")
    LazyAgent("anthropic", name="energy_specialist", session=specialist_sess,
              system="Energy transition specialist. Evaluate renewables and grid plays.")
    LazyAgent("openai",    name="macro_specialist",  session=specialist_sess,
              system="Macro strategist. Assess cross-sector risks and portfolio implications.")

    sector_intel_tool = specialist_sess.as_tool(
        "sector_intel", "Parallel specialist analysis across tech, energy, and macro.",
        mode="parallel",
    )

    # Step 3: synthesizer chains after the combined parallel briefings
    synthesizer = LazyAgent(
        "openai", name="synthesizer",
        output_schema=InvestmentReport,
        system="Senior portfolio strategist. Synthesise specialist briefings "
               "into a structured investment report with allocation recommendations.",
    )

    # Chain: LazyAgent → LazyTool → LazyAgent — zero glue code
    full_pipeline = LazySession(tracking="basic").as_tool(
        "fan_out_pipeline",
        "Broad sweep → parallel specialisation → structured synthesis.",
        mode="chain",
        participants=[generalist, sector_intel_tool, synthesizer],
    )

    result = full_pipeline.run({"task": TASK})
    print(result)


# =============================================================================
# VARIANT 4 — Fully Structured, Multi-Provider Pipeline (typed end-to-end)
# =============================================================================
#
# Every agent owns output_schema. The chain calls json() at each step
# automatically, serialises the Pydantic result to JSON, and passes it as the
# next agent's task. Three different providers, three Pydantic models, zero def.
#
#   Google(MarketBriefing) → Anthropic(RiskProfile) → OpenAI(InvestmentReport)
# =============================================================================

def run_variant_4() -> None:
    print("\n── Variant 4: Fully Structured Multi-Provider Pipeline ──\n")

    # Layer 1: three Google agents run in parallel, each produces a MarketBriefing
    market_sess = LazySession(tracking="basic")
    LazyAgent("google", name="tech_researcher",    session=market_sess,
              native_tools=[NativeTool.WEB_SEARCH],
              output_schema=MarketBriefing,
              system="Technology analyst — AI, semiconductors, cloud. "
                     "Use web search and produce a structured market briefing.")
    LazyAgent("google", name="energy_researcher",  session=market_sess,
              native_tools=[NativeTool.WEB_SEARCH],
              output_schema=MarketBriefing,
              system="Energy analyst — renewables, oil & gas, grid. "
                     "Use web search and produce a structured market briefing.")
    LazyAgent("google", name="biotech_researcher", session=market_sess,
              native_tools=[NativeTool.WEB_SEARCH],
              output_schema=MarketBriefing,
              system="Biotech analyst — pharma, genomics, medical devices. "
                     "Use web search and produce a structured market briefing.")

    parallel_market_tool = market_sess.as_tool(
        "parallel_market_research",
        "Three Google agents research tech, energy, and biotech in parallel. "
        "Each produces a structured MarketBriefing.",
        mode="parallel",
    )

    # Layer 2 + 3: risk analyst and report writer chain after the parallel output
    analysis_sess = LazySession(tracking="basic")
    risk_analyst  = LazyAgent("anthropic", name="risk_analyst",  session=analysis_sess,
                               output_schema=RiskProfile,
                               system="Senior risk analyst. Given structured market briefings, "
                                      "produce a structured risk profile per sector.")
    report_writer = LazyAgent("openai",    name="report_writer", session=analysis_sess,
                               output_schema=InvestmentReport,
                               system="CFA report writer. Given a structured risk profile, "
                                      "produce a comprehensive structured investment report.")

    # Full pipeline: parallel LazyTool → LazyAgent(RiskProfile) → LazyAgent(InvestmentReport)
    # Three providers, three Pydantic models, one call — zero def
    pipeline = LazySession(tracking="basic").as_tool(
        "fully_structured_pipeline",
        "Parallel Google research → Anthropic risk assessment → OpenAI report.",
        mode="chain",
        participants=[parallel_market_tool, risk_analyst, report_writer],
    )

    result = pipeline.run({"task": TASK})
    print(result)


# =============================================================================
# Entry point
# =============================================================================

_VARIANTS = {
    "1": run_variant_1,
    "2": run_variant_2,
    "3": run_variant_3,
    "4": run_variant_4,
}

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "1"
    fn = _VARIANTS.get(choice)
    if fn is None:
        print("Usage: python investment_research_platform.py [1|2|3|4]")
        sys.exit(1)
    fn()
