"""
write_paper.py  —  Multi-provider pipeline that writes the LazyBridge technical paper
======================================================================================

LazyBridge writes about itself using its own primitives.

Architecture
------------
Phase 0  Doc skill (build BM25 index of lazy_wiki/bot/ — once, cached on disk)
         Researchers and writer use this tool to look up accurate API details.

Phase 1  Research (parallel, 3 × Google Gemini — cheap, fast, independent)
         · res_philosophy    — the one-primitive thesis and what it solves
         · res_competitive   — LazyBridge vs LangChain, concrete code comparison
         · res_llm_first     — LLM-first design, bot wiki, Claude skills, verify=
         Each researcher has tools=[doc_tool] and queries the index autonomously.

Phase 2  Multi-provider debate (parallel, Claude + GPT + Gemini — smart, independent)
         · analyst_claude    — architecture, composability, production readiness
         · analyst_openai    — developer experience, adoption, pragmatism
         · analyst_google    — multi-provider strategy, enterprise, ecosystem

Phase 3  Synthesis (Claude Sonnet — structured PaperOutline via output_schema)
         Resolves the debate, extracts the strongest arguments, produces a paper outline.

Phase 1–3 are composed as a single declarative chain:
    research_tool → debate_tool → synthesizer

Phase 4  Writing (Claude Sonnet + doc_tool + verify= quality gate, max 2 retries)
         verify= runs outside the chain — it is a loop-time parameter, not a chain step.

Output:  artifacts/lazybridge_paper.md

Run:
    python examples/write_paper.py
"""

import os
import sys
from pathlib import Path
from pydantic import BaseModel

# ── Path setup ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Load .env (key names only — no values printed) ────────────────────────────
_env = REPO_ROOT / ".env"
if _env.exists():
    _keys = []
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip("'\"")
            os.environ.setdefault(_k, _v)
            _keys.append(_k)
    print(f"[.env] loaded: {', '.join(_keys)}")
else:
    print("[.env] not found — API keys must be set in environment")

from lazybridge import LazyAgent, LazySession                          # noqa: E402
from lazybridge.tools.doc_skills import build_skill, skill_tool        # noqa: E402


# ── Output schema for Phase 3 ─────────────────────────────────────────────────

class PaperOutline(BaseModel):
    title:                    str
    subtitle:                 str
    hook:                     str        # opening paragraph
    sections:                 list[str]  # section titles in order
    code_examples_to_include: list[str]  # specific code patterns to show
    tone:                     str        # e.g. "honest technical, peer-to-peer"
    strongest_arguments:      list[str]  # top arguments distilled from the debate
    caveats_to_acknowledge:   list[str]  # honest limitations to include


TASK = (
    "Research and write a technical paper about the LazyBridge framework — "
    "its architectural philosophy, how it compares to LangChain, and its "
    "LLM-first design features (bot wiki, verify= quality gate, lazybridge.tools)."
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Doc skill   (BM25 index of lazy_wiki/bot/, cached on disk)
# ══════════════════════════════════════════════════════════════════════════════

_skill_meta = build_skill(
    source_dirs=[str(REPO_ROOT / "lazy_wiki" / "bot")],
    skill_name="lazybridge_docs",
    output_root=str(REPO_ROOT / "artifacts" / "skills"),
    description=(
        "LazyBridge framework documentation — API reference, patterns, "
        "composability model, multi-provider usage, and built-in tools."
    ),
)
doc_tool = skill_tool(
    _skill_meta["skill_dir"],
    name="query_lazybridge_docs",
    description=(
        "Search the LazyBridge documentation for accurate API details, "
        "code patterns, parameter names, and architectural explanations. "
        "Use this whenever you need to verify a claim or look up exact syntax."
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Research   (3 × Google, parallel, doc-grounded via tools)
# ══════════════════════════════════════════════════════════════════════════════

sess_research = LazySession(tracking="basic", console=True)

res_philosophy = LazyAgent(
    "google",
    name="res_philosophy",
    session=sess_research,
    tools=[doc_tool],
    system=(
        "You are a software architecture researcher. Be specific, technical, honest. "
        "Write a 500+ word analysis of the LazyBridge framework: its composability model, "
        "what complexity it removes vs raw SDK usage, and how LazySession.as_tool() enables "
        "declarative pipeline composition. "
        "IMPORTANT — the doc tool contains API reference only (class definitions, method "
        "signatures, code examples). Query it with specific API terms like 'LazyAgent loop', "
        "'LazySession as_tool chain parallel', 'output_schema participants'. "
        "Do NOT query for 'philosophy', 'thesis', or other conceptual terms — they are not "
        "indexed. Use your own knowledge for narrative framing; use the tool to verify code."
    ),
)

res_competitive = LazyAgent(
    "google",
    name="res_competitive",
    session=sess_research,
    tools=[doc_tool],
    system=(
        "You are a comparative framework analyst. Be concrete with code. "
        "Write a 500+ word comparison of LazyBridge vs LangChain. Show the same "
        "multi-provider sequential pipeline in both frameworks side by side. Count lines, "
        "abstractions, concepts. Be fair about LangChain strengths. "
        "IMPORTANT — the doc tool contains LazyBridge API reference only; it has NO LangChain "
        "content. Use your general knowledge for LangChain code. Query the tool only for "
        "LazyBridge specifics: 'LazyAgent chat loop', 'LazySession as_tool', 'LazyTool "
        "from_function', 'output_schema', 'verify'. Do NOT query 'LangChain' or 'comparison'."
    ),
)

res_llm_first = LazyAgent(
    "google",
    name="res_llm_first",
    session=sess_research,
    tools=[doc_tool],
    system=(
        "You are a developer-experience researcher focused on LLM tooling. "
        "Analyse three specific LazyBridge features: "
        "(1) the LLM-facing bot wiki (the lazy_wiki/bot/ directory), "
        "(2) the verify= quality gate parameter on loop(), "
        "(3) the lazybridge.tools subpackage. "
        "IMPORTANT — the doc tool contains API reference. Query with specific terms: "
        "'verify quality gate loop', 'doc_skills build_skill skill_tool', 'INDEX bot wiki'. "
        "Do NOT query vague terms like 'philosophy' or 'feature' — they return nothing. "
        "Minimum 500 words."
    ),
)

research_tool = sess_research.as_tool(
    "parallel_research",
    "Three Google Gemini researchers independently analyse LazyBridge from different angles.",
    mode="parallel",
    participants=[res_philosophy, res_competitive, res_llm_first],
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Multi-provider debate   (Claude + GPT + Gemini, independent)
# ══════════════════════════════════════════════════════════════════════════════

sess_debate = LazySession(tracking="basic", console=True)

analyst_claude = LazyAgent(
    "anthropic",
    name="analyst_claude",
    session=sess_debate,
    system="""You are a senior software architect evaluating LazyBridge.
Your lens: composability, production readiness, correctness of architectural claims.
Push back where claims are oversold. Distinguish genuinely novel ideas from "simpler LangChain".
Be honest about what one-primitive composability actually buys at scale.
Grade the verify= feature, the multi-provider story, and lazybridge.tools on architectural merit.""",
)

analyst_openai = LazyAgent(
    "openai",
    name="analyst_openai",
    session=sess_debate,
    system="""You are a pragmatic backend engineer evaluating LazyBridge.
Your lens: day-to-day usability, adoption curve, maintenance, debugging experience.
Would you use this in production? What would make you switch from LangChain?
Evaluate the learning curve honestly. What is still missing for serious production use?
Be specific about what "zero boilerplate" actually saves in a real project.""",
)

analyst_google = LazyAgent(
    "google",
    name="analyst_google",
    session=sess_debate,
    system="""You are an enterprise technology strategist evaluating LazyBridge.
Your lens: vendor risk, multi-provider strategy, long-term viability, Apache 2.0 implications.
Be honest about the single-maintainer bus factor. Is multi-provider truly a differentiator
or is it a commodity feature? What would need to be true for an enterprise team to adopt this
over LangChain or building their own thin wrapper?""",
)

debate_tool = sess_debate.as_tool(
    "multi_provider_debate",
    "Three AI providers independently assess LazyBridge's technical merits and limitations.",
    mode="parallel",
    participants=[analyst_claude, analyst_openai, analyst_google],
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Synthesis   (Claude Sonnet, Pydantic output via output_schema)
# ══════════════════════════════════════════════════════════════════════════════

synthesizer = LazyAgent(
    "anthropic",
    name="synthesizer",
    output_schema=PaperOutline,
    system="""You distill multi-perspective technical debates into structured paper outlines.
Extract only the arguments with real evidence behind them. Acknowledge genuine limitations.
The paper must convince a senior backend engineer — not a marketing audience.
Where analysts disagreed, pick the better-supported position and note the dissent.""",
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Writing   (Claude Sonnet + doc_tool + verify= quality gate)
# ══════════════════════════════════════════════════════════════════════════════
# verify= is a loop-time parameter — it cannot be expressed inside a chain step.
# The pipeline chain returns the PaperOutline; writer.loop() is called explicitly.

writer = LazyAgent(
    "anthropic",
    name="technical_writer",
    tools=[doc_tool],
    system="""You are a senior technical writer for a developer audience.
Writing rules — non-negotiable:
  · Every claim needs a code example or a concrete before/after comparison.
  · Adjectives without evidence are banned. Not "powerful" — "3 lines vs 47 lines".
  · Acknowledge limitations honestly — one section must address when NOT to use this.
  · Tone: colleague explaining something genuinely useful to another engineer.
  · Format: Markdown, ready to publish on dev.to or Medium.
  · Length: 1800–2600 words. Not a word more for padding.
  · End with a real quick-start users can run in 5 minutes.
Use the query_lazybridge_docs tool to verify exact API signatures, parameter names,
and code patterns before including them in examples.""",
)

verifier = LazyAgent(
    "anthropic",
    name="verifier",
    system="""Review technical paper drafts about LazyBridge.

Check each criterion — ALL must pass for APPROVED:
1. CODE: Does every major architectural claim have a supporting code example?
2. HONEST: Is at least one section dedicated to limitations / when NOT to use it?
3. COVERAGE: Does it cover all five topics — (a) one-primitive thesis,
   (b) multi-provider without lock-in, (c) LLM-first design / bot wiki,
   (d) verify= quality gate, (e) lazybridge.tools?
4. TONE: No marketing adjectives without evidence. Reads like a peer, not a pitch.
5. ACTIONABLE: Does it end with something a reader can run in under 5 minutes?

Reply APPROVED if all 5 pass.
Reply REJECTED and list exactly which criteria failed and what is missing.""",
)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE   research → debate → synthesis  (declarative chain)
# ══════════════════════════════════════════════════════════════════════════════

pipeline = LazySession(tracking="basic", console=True).as_tool(
    "research_debate_synthesize",
    "Parallel research → parallel multi-provider debate → structured paper outline.",
    mode="chain",
    participants=[research_tool, debate_tool, synthesizer],
)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEP = "═" * 64
    print(f"\n{SEP}")
    print("  LazyBridge Paper Pipeline  —  writing about itself")
    print(SEP)

    n_files = (
        len(_skill_meta["indexed_files"])
        if isinstance(_skill_meta["indexed_files"], list)
        else _skill_meta["indexed_files"]
    )
    print(f"\n[Phase 0] Doc skill ready — {n_files} files / {_skill_meta['total_chunks']} chunks indexed")

    print("\n[Phases 1–3] Research → Debate → Synthesis…")
    outline = pipeline.run({"task": TASK})
    # pipeline.run() returns the synthesizer's typed output (PaperOutline) — serialize to JSON
    outline_str = (
        outline.model_dump_json(indent=2)
        if hasattr(outline, "model_dump_json")
        else str(outline)
    )
    print(f"  ✓ outline produced ({len(outline_str):,} chars)")

    print("\n[Phase 4] Writing paper with verify= quality gate (max 2 retries)…")
    paper_result = writer.loop(
        f"Write the complete, publish-ready Markdown technical paper based on this outline:\n\n{outline_str}",
        verify=verifier,
        max_verify=2,
    )
    print(f"  ✓ paper: {len(paper_result.content):,} chars / ~{len(paper_result.content.split()):,} words")
    if getattr(paper_result, "verify_log", None):
        print(f"  ✓ verify passed after {len(paper_result.verify_log) + 1} attempt(s)")

    out_dir = REPO_ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lazybridge_paper.md"
    out_path.write_text(paper_result.content, encoding="utf-8")

    print(f"\n{SEP}")
    print(f"  Paper saved → {out_path.relative_to(REPO_ROOT)}")
    print(SEP)
    print("\n── Preview (first 800 chars) " + "─" * 36)
    print(paper_result.content[:800])
    print("─" * 64)
