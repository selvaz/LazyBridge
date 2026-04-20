#!/usr/bin/env python3
"""
LazyBridge Framework Monitor

Phase 1: SDK search    — 3 parallel searchers per provider (different angles)
Phase 2: Code readers  — 1 per module, reads actual source, maps impact
Phase 3: Verification  — each finding re-verified with web search
Phase 4: Synthesis     — action plan with code examples, judge-verified

Usage:
    python framework_monitor.py
    python framework_monitor.py --months 1
    python framework_monitor.py --providers anthropic openai
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from lazybridge import (
    CallbackExporter,
    Event,
    LazyAgent,
    LazyContext,
    LazySession,
    LazyStore,
    LazyTool,
    Memory,
)
from lazybridge.core.types import NativeTool

# ── limits ────────────────────────────────────────────────────────────────────
SEARCH_MAX_STEPS = 3
READER_MAX_STEPS = 6
VERIFY_MAX_STEPS = 3

# ── display ───────────────────────────────────────────────────────────────────
R, B, D = "\033[0m", "\033[1m", "\033[2m"
STYLES = {
    "Searcher":    ("\033[90m", "📡"),
    "Reader":      ("\033[93m", "📖"),
    "Verifier":    ("\033[92m", "🔎"),
    "Synthesizer": ("\033[95m", "⚗️ "),
    "Judge":       ("\033[91m", "⚖️ "),
}


def on_event(e: dict) -> None:
    name, etype, data = e["agent_name"], e["event_type"], e.get("data", {})
    c, emoji = STYLES.get(name, ("\033[90m", "·"))
    if etype == Event.TOOL_CALL:
        msg = str(data.get("arguments", {}).get(
            "task", data.get("arguments", {}).get("path", "…")))
        print(f"\n{c}{D}  {emoji} [{name}] → {data.get('name','?')}(\"{msg[:80]}\"){R}")
    elif etype == Event.AGENT_FINISH:
        out = str(data.get("result", ""))
        print(f"\n{c}{B}── {emoji} {name} ──────────────────────────────────{R}")
        print(f"{c}{D}{out[:400]}{'…' if len(out) > 400 else ''}{R}")


# ── session & store ───────────────────────────────────────────────────────────
SESS      = LazySession(db="framework_monitor.db", tracking="verbose", console=True,
                        exporters=[CallbackExporter(on_event)])
artifacts = LazyStore(db="framework_monitor_artifacts.db")

# ── framework root ────────────────────────────────────────────────────────────
FRAMEWORK_ROOT = Path(__file__).parent / "lazybridge"

# ── lazy contexts — read from artifacts after each phase writes them ──────────
# Evaluated at execution time, so Phase 2 agents see Phase 1 results, etc.
research_ctx = LazyContext.from_function(lambda: artifacts.read("research") or "")
analysis_ctx = LazyContext.from_function(lambda: artifacts.read("analysis") or "")

# ── source reading tools ───────────────────────────────────────────────────────
def _read_source(path: str, start: int = 0, end: int = 0) -> str:
    """Read a LazyBridge source file.
    path: relative to lazybridge/ (e.g. 'lazy_agent.py' or 'core/providers/anthropic.py')
    start/end: line numbers 1-indexed, 0=all. Read 30-50 lines at a time."""
    full = FRAMEWORK_ROOT / path.lstrip("/")
    if not full.exists():
        matches = list(FRAMEWORK_ROOT.rglob(Path(path).name))
        if not matches:
            return f"[not found: {path}. Use list_sources to see available files]"
        full = matches[0]
    lines = full.read_text(encoding="utf-8").splitlines()
    if start and end:   lines, prefix = lines[start-1:end], start
    elif start:         lines, prefix = lines[start-1:min(start+60, len(lines))], start
    else:               lines, prefix = lines[:80], 1
    return f"# {full.relative_to(FRAMEWORK_ROOT.parent)}\n" + \
           "\n".join(f"{prefix+i:4}: {l}" for i, l in enumerate(lines))


def _list_sources(subdir: str = "") -> str:
    """List all .py files. subdir: optional filter e.g. 'core/providers'"""
    root = FRAMEWORK_ROOT / subdir if subdir else FRAMEWORK_ROOT
    return "\n".join(
        str(p.relative_to(FRAMEWORK_ROOT))
        for p in sorted(root.rglob("*.py"))
        if "__pycache__" not in str(p)
    )


read_tool = LazyTool.from_function(
    _read_source, name="read_source",
    description=(
        "Read LazyBridge source file. "
        "path=relative to lazybridge/ (e.g. 'lazy_agent.py'), "
        "start=first line (optional), end=last line (optional). "
        "Read 30-60 lines at a time for efficiency."
    ))
list_tool = LazyTool.from_function(
    _list_sources, name="list_sources",
    description="List .py source files. subdir=optional filter e.g. 'core/providers'")

# ── navigator (skill map — static, cheap, always in context) ──────────────────
NAVIGATOR = LazyContext.from_text("""
LazyBridge v0.6 — Codebase Navigator
(use read_source to get actual code, list_sources to explore)

KEY FILES & LINE RANGES:
  lazy_agent.py        LazyAgent(line 162), __init__(~205), loop(~1035),
                       chat(~745), as_tool(~1344), _loop_logic(~567),
                       _build_effective_system(~299), _record_response(~?)
  lazy_tool.py         LazyTool(127), from_function(165), from_agent(200),
                       parallel(1002), chain(1077), _DelegateConfig(88),
                       NormalizedToolSet(1472), _run_delegate(303)
  pipeline_builders.py build_chain_func(242), build_parallel_func(147),
                       _clone_for_invocation(501), _ChainState(31)
  memory.py            Memory(34), strategies: full/rolling/auto
  lazy_context.py      LazyContext(47), from_function/from_agent/from_store/+
  lazy_store.py        LazyStore(200), _SQLiteBackend(99), _InMemoryBackend(51)
  lazy_session.py      LazySession(385), Event enum(80), TrackLevel(68)
  human.py             HumanAgent(48) — duck-type LazyAgent with stdin
  supervisor.py        SupervisorAgent(57)
  guardrails.py        ContentGuard(130), LLMGuard(252), GuardChain(186)
  exporters.py         CallbackExporter(62), FilteredExporter(90)

PROVIDERS (core/providers/):
  base.py         BaseProvider ABC, _TIER_ALIASES dict, supported_native_tools
  anthropic.py    AnthropicProvider — _TIER_ALIASES, _init_client, complete()
  openai.py       OpenAIProvider
  google.py       GoogleProvider
  deepseek.py     DeepSeekProvider

CURRENT MODEL TIERS:
  anthropic: top=claude-opus-4-7, medium=claude-sonnet-4-6,
             cheap=claude-haiku-4-5, super_cheap=claude-3-haiku
  openai:    top=gpt-5.4, expensive=gpt-5, medium=gpt-4o,
             cheap=gpt-4o-mini, super_cheap=gpt-3.5-turbo

CURRENT NativeTool enum values:
  WEB_SEARCH, CODE_EXECUTION, FILE_SEARCH, COMPUTER_USE,
  GOOGLE_SEARCH, GOOGLE_MAPS

KEY INVARIANTS:
  - parallel() sends {"task": str} — function params must be named 'task'
  - chain() clones agents → use return value not agent.result after chain
  - LazyContext.from_agent(a) reads a._last_output — populated after run
  - verify= on as_tool() = Option B (in _DelegateConfig.verify)
  - verify= on LazyAgent() = Option C (self.verify, only in _run_delegate)
  - empty content = LLM ended on tool_use → fix: force_final_after_tools=True
  - Memory shared between chain clone and original (shallow copy)
""")

# ── module → files mapping ────────────────────────────────────────────────────
MODULES = {
    "providers":   ["core/providers/anthropic.py", "core/providers/openai.py",
                    "core/providers/google.py",    "core/providers/deepseek.py",
                    "core/providers/base.py"],
    "agent_core":  ["lazy_agent.py", "memory.py", "lazy_context.py"],
    "tools_chain": ["lazy_tool.py", "pipeline_builders.py"],
    "types_infra": ["core/types.py", "core/tool_schema.py", "lazy_store.py",
                    "lazy_session.py", "exporters.py"],
}

# ── search configs: 3 providers × different focus angles ─────────────────────
SEARCH_CONFIGS = [
    ("anthropic", "official changelog, release notes, new models, API changes"),
    ("openai",    "migration guides, deprecated features, new parameters"),
    ("google",    "community reports, third-party coverage, SDK updates"),
]


# ── Phase 1: SDK searcher factory ─────────────────────────────────────────────
def _make_searcher(search_provider: str, sdk: str, focus: str) -> LazyAgent:
    return LazyAgent(
        search_provider, model="cheap", name="Searcher", session=SESS,
        memory=Memory(),
        native_tools=[NativeTool.WEB_SEARCH],
        system=(
            f"You are a changelog monitor for the {sdk.capitalize()} SDK. "
            f"Focus: {focus}. "
            f"Search only for changes in the LAST 1-2 MONTHS. Ignore older news. "
            f"Be specific: include model names, parameter names, version numbers. "
            f"Cite URLs."
        ),
    )


# ── Phase 2: code reader factory ─────────────────────────────────────────────
def _make_reader(module: str, files: list[str]) -> LazyAgent:
    # research_ctx is lazy — reads from artifacts when the agent runs (after Phase 1)
    return LazyAgent(
        "anthropic", model="medium", name="Reader", session=SESS,
        memory=Memory(),
        tools=[read_tool, list_tool],
        context=NAVIGATOR + research_ctx,
        system=(
            f"You analyze the LazyBridge module: {module}\n"
            f"Relevant files: {', '.join(files)}\n\n"
            f"WORKFLOW:\n"
            f"1. The SDK changelog research is already in your context\n"
            f"2. Use read_source to read the ACTUAL current code of your files\n"
            f"3. Identify specific impacts: what needs to change and exactly where\n\n"
            f"FORMAT each finding as:\n"
            f"### Finding: <title>\n"
            f"- **Type**: breaking_change | new_feature | deprecation | model_update\n"
            f"- **Priority**: HIGH | MEDIUM | LOW\n"
            f"- **File**: <filename>\n"
            f"- **Lines**: <start>-<end>\n"
            f"- **Current code**:\n```python\n<actual code snippet>\n```\n"
            f"- **Recommended change**:\n```python\n<new code snippet>\n```\n"
            f"- **Reason**: <why this matters>\n"
            f"- **Verify query**: <web search query to confirm this finding>\n"
        ),
    )


# ── Phase 3: finding verifier ─────────────────────────────────────────────────
verifier = LazyAgent(
    "openai", model="medium", name="Verifier", session=SESS,
    memory=Memory(),
    native_tools=[NativeTool.WEB_SEARCH],
    # research_ctx is lazy — reads from artifacts at execution time
    context=NAVIGATOR + research_ctx,
    system=(
        "You verify code reader findings against live web sources.\n"
        "For each finding in your task:\n"
        "1. Run the 'verify query' from the finding as a web search\n"
        "2. Confirm or refute the finding based on current official docs\n"
        "3. Add a **Verified**: ✅ confirmed | ⚠️ partially confirmed | ❌ not confirmed\n"
        "4. Add **Source**: URL that confirms or refutes\n"
        "Keep findings that are confirmed or partially confirmed. "
        "Mark unconfirmed ones clearly but keep them."
    ),
)

# ── Phase 4: synthesis + judge ────────────────────────────────────────────────
judge = LazyAgent(
    "anthropic", model="cheap", name="Judge", session=SESS,
    tools=[read_tool],
    # research_ctx is lazy — reads from artifacts at execution time
    context=NAVIGATOR + research_ctx,
    system=(
        "Verify the action plan:\n"
        "1. Every code example must match actual LazyBridge patterns\n"
        "2. Every finding must have a verified source\n"
        "3. Use read_source to spot-check code examples if unsure\n"
        "Reply 'approved' or 'retry: <specific issue>'."
    ),
)

synthesizer = LazyAgent(
    "anthropic", model="medium", name="Synthesizer", session=SESS,
    # both contexts are lazy — read from artifacts at execution time (after phases 1+2)
    context=NAVIGATOR + research_ctx + analysis_ctx,
    system=(
        "Produce a prioritized LazyBridge update report.\n\n"
        "FORMAT:\n"
        "# LazyBridge Update Report — {date}\n\n"
        "## 🔴 URGENT — breaking changes\n"
        "## 🟡 IMPORTANT — new features to add\n"
        "## 🟢 PLANNED — deprecations\n"
        "## ℹ️ MONITOR — no action yet\n\n"
        "For each item:\n"
        "- File + line range\n"
        "- Current code (snippet)\n"
        "- Recommended change (snippet)\n"
        "- Verified source URL\n\n"
        "Only include items with at least ⚠️ partial verification. "
        "No vague suggestions — concrete patches only."
    ),
)

synthesis_tool = synthesizer.as_tool(
    name="monitor_pipeline",
    description="Synthesis + judge verification",
    verify=judge,
    max_verify=2,
)

# ── report export ─────────────────────────────────────────────────────────────
def _save_report(verdict: str, providers: list[str], months: int) -> str:
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"lazybridge_update_{ts}.md"
    sections = [
        f"# LazyBridge Update Report\n\n"
        f"**Date:** {ts}  \n"
        f"**Providers monitored:** {', '.join(providers)}  \n"
        f"**Period:** last {months} month(s)\n",
        "---\n## SDK Research\n",    artifacts.read("research") or "_none_",
        "---\n## Module Analysis\n", artifacts.read("analysis") or "_none_",
        "---\n## Verified Findings\n", artifacts.read("verified") or "_none_",
        "---\n## Action Plan\n",     verdict,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))
    print(f"\n\033[90m  📄 Report saved → {path}\033[0m")
    return path


# ── run ───────────────────────────────────────────────────────────────────────
def run(providers: list[str] | None = None, months: int = 2) -> str:
    providers = providers or ["anthropic", "openai", "google", "deepseek"]
    cutoff    = (datetime.datetime.now() - datetime.timedelta(days=months * 30)
                 ).strftime("%B %Y")

    print(f"\n\033[96m{B}🔍 LAZYBRIDGE MONITOR  ·  "
          f"providers: {', '.join(providers)}  ·  last {months} month(s){R}\n")

    artifacts.clear()
    verifier.memory.clear()

    # ── Phase 1: parallel SDK search ─────────────────────────────────────────
    # Each searcher is exposed via .as_tool() so _arun_delegate calls aloop()
    # with force_final_after_tools=True.  native_tools= must be passed explicitly
    # so the delegate's branch logic (not agent.native_tools) triggers aloop.
    print(f"\033[90m{B}── PHASE 1: SDK SEARCH ──────────────────────────────────{R}")

    searcher_tools = [
        _make_searcher(search_provider, sdk, focus).as_tool(
            name=f"{sdk}_{search_provider}",
            description=f"Search {sdk} SDK via {search_provider}: {focus}",
            native_tools=[NativeTool.WEB_SEARCH],
        )
        for sdk in providers
        for search_provider, focus in SEARCH_CONFIGS
    ]

    # build_aparallel_func → asyncio.gather → each tool.arun() → _arun_delegate
    # → agent.aloop(task, native_tools=[WEB_SEARCH], force_final_after_tools=True)
    research = LazyTool.parallel(
        *searcher_tools,
        name="sdk_search",
        description="Parallel SDK search",
    ).run({"task": (
        f"Find all {', '.join(providers)} SDK/API changes from {cutoff} to now. "
        f"Focus on: new models, deprecated models, new API features, breaking changes, "
        f"new tool types, parameter changes. Be specific with names and versions."
    )})

    artifacts.write("research", research)
    # research_ctx now returns this string to all subsequent agents

    # ── Phase 2: parallel code analysis ──────────────────────────────────────
    # Readers use as_tool(): _has_tools=True (agent.tools=[read_tool,list_tool])
    # → _arun_delegate calls aloop(task, force_final_after_tools=True).
    # research_ctx in each reader's context is lazy — reads from artifacts now.
    print(f"\n\033[93m{B}── PHASE 2: CODE ANALYSIS ───────────────────────────────{R}")

    reader_tools = [
        _make_reader(module, files).as_tool(
            name=module,
            description=f"Analyze {module} module for SDK-driven changes",
        )
        for module, files in MODULES.items()
    ]

    analysis = LazyTool.parallel(
        *reader_tools,
        name="code_analysis",
        description="Parallel module analysis",
    ).run({"task": (
        f"Analyze how recent SDK changes (last {months} months) impact your module. "
        f"Read the actual source files first, then identify specific patches needed."
    )})

    artifacts.write("analysis", analysis)
    # analysis_ctx now returns this string to synthesizer

    # ── Phase 3: verify findings ──────────────────────────────────────────────
    print(f"\n\033[92m{B}── PHASE 3: VERIFICATION ────────────────────────────────{R}")

    verified = verifier.loop(
        f"Verify all findings from the module analysis. "
        f"For each finding that has a 'Verify query', run it as a web search "
        f"and add verification status.\n\n"
        f"Current analysis:\n\n{analysis_ctx.build()}",
        max_steps=VERIFY_MAX_STEPS,
    ).content

    artifacts.write("verified", verified)

    # ── Phase 4: synthesis + judge ────────────────────────────────────────────
    print(f"\n\033[95m{B}── PHASE 4: ACTION PLAN ─────────────────────────────────{R}")

    verdict = synthesis_tool.run({"task": (
        f"Produce the final update plan for LazyBridge based on:\n"
        f"- SDK changes from the last {months} month(s)\n"
        f"- Module analysis with code snippets\n"
        f"- Verified findings\n\n"
        f"Verified findings:\n{verified}"
    )})

    artifacts.write("verdict", verdict)
    report = _save_report(verdict, providers, months)

    print(f"\n\033[95m{B}{'═'*60}\n  🔍  UPDATE PLAN\n{'═'*60}{R}\n{verdict}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LazyBridge Framework Monitor")
    parser.add_argument("--providers", nargs="+",
                        choices=["anthropic", "openai", "google", "deepseek"],
                        default=["anthropic", "openai", "google", "deepseek"])
    parser.add_argument("--months", type=int, default=2,
                        help="How many months back to check (default: 2)")
    args = parser.parse_args()
    run(providers=args.providers, months=args.months)
