# Examples

## Research Pipeline — the flagship comparison

Three implementations of the same pipeline:
- **Researcher** (Anthropic): searches for company info using tool calling
- **Writer** (OpenAI): produces a structured Pydantic report
- **Orchestrator** (Anthropic): drives the pipeline for multiple companies

| File | Approach | Lines | Dependencies |
|---|---|---|---|
| `research_pipeline_raw_sdk.py` | Anthropic + OpenAI SDK, fully manual | ~110 | 2 packages |
| `research_pipeline_langgraph.py` | LangGraph + LangChain | ~65 | 4 packages |
| `research_pipeline_lazybridge.py` | LazyBridge | ~35 | 1 package |

### What the comparison shows

| | Raw SDK | LangGraph | LazyBridge |
|---|---|---|---|
| Manual tool loops | 3 `while` loops | 0 | 0 |
| Explicit state declaration | no | yes (TypedDict) | no |
| Separate LLM clients / classes | 2 | 2 | 1 unified API |
| Multi-provider swap | rewrite | change class | change string |
| Session tracking | absent | LangSmith (external) | built-in |

### Running

Each file is self-contained. Set your API keys and run:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

python examples/research_pipeline_lazybridge.py
```

---

## Investment Research Platform — four architectural variants

`investment_research_platform.py` shows the same investment research workflow implemented
four different ways, demonstrating the full range of LazyBridge composition patterns.

```
Variant 1 — Three-Tier Orchestrator          (LLM-directed)
  orchestrator (Anthropic)
  ├── market_intelligence_tool  ← parallel, 3 Google agents with WEB_SEARCH
  └── analysis_chain_tool       ← chain, agents own output_schema

Variant 2 — Nested Pipeline, no orchestrator (pipeline-directed)
  full_pipeline (chain)
  ├── market_tool    ← LazyTool wrapping the parallel session
  ├── risk_analyst   ← receives market output as task
  └── report_writer  ← receives risk_analyst context

Variant 3 — Fan-Out then Synthesize          (mixed)
  full_pipeline (chain)
  ├── generalist          ← Anthropic + DB tool, broad sweep
  ├── sector_intel_tool   ← parallel specialists
  └── synthesizer         ← OpenAI, final report

Variant 4 — Fully Structured Multi-Provider  (typed end-to-end)
  pipeline (chain)
  ├── market_researcher  output_schema=MarketBriefing   (Google + WEB_SEARCH)
  ├── risk_analyst       output_schema=RiskProfile      (Anthropic)
  └── report_writer      output_schema=InvestmentReport (OpenAI)
```

### Features demonstrated

- `output_schema` on the agent: chain/parallel dispatch `json()` automatically
- `native_tools` declared at agent construction, not at pipeline call time
- `LazyTool` as a participant inside a chain (Variant 2)
- Fan-out with generalist → parallel specialists → synthesis (Variant 3)
- Fully typed Pydantic pipeline without a single `def` (Variant 4)

### Running

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...

python examples/investment_research_platform.py        # Variant 1 (default)
python examples/investment_research_platform.py 2      # Variant 2
python examples/investment_research_platform.py 3      # Variant 3
python examples/investment_research_platform.py 4      # Variant 4
```

---

## Write Paper — LazyBridge writes about itself

`write_paper.py` is a four-phase pipeline that produces a publish-ready technical paper
about the LazyBridge framework, using LazyBridge's own primitives.

```
Phase 0  Doc skill — BM25 index of lazy_wiki/bot/, cached on disk

Phase 1  Parallel research (3 × Google Gemini)
         Each researcher has tools=[doc_tool] and queries the index autonomously.

Phase 2  Parallel multi-provider debate (Claude + GPT + Gemini)
         Each analyst evaluates from its own lens (architecture / DX / enterprise).

Phase 3  Synthesis (Claude Sonnet, output_schema=PaperOutline)
         Resolves the debate into a structured outline — auto-dispatched by the chain.

         ↑ Phases 1–3 are a single declarative chain: research_tool → debate_tool → synthesizer

Phase 4  Writing (Claude Sonnet + doc_tool + verify= quality gate, max 2 retries)
         verify= is a loop-time parameter; called explicitly after the chain.
```

### Features demonstrated

- `LazySession.as_tool(mode="parallel")` for independent parallel agents
- `LazySession.as_tool(mode="chain")` nesting a parallel `LazyTool` as a chain participant
- `output_schema` on the synthesizer: chain calls `.json()` automatically, no manual parsing
- `tools=[doc_tool]` on research agents: self-serve BM25 queries instead of injected context
- `verify=` quality gate on `writer.loop()`: iterative self-correction up to `max_verify` times
- `lazybridge.ext.doc_skills`: index local docs, expose as a `LazyTool`

### Running

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...

python examples/write_paper.py
# Output → artifacts/lazybridge_paper.md
```

---

## Supervised Pipeline — human-in-the-loop with SupervisorAgent

`supervised_pipeline.py` wires `researcher → SupervisorAgent → writer` via `LazyTool.chain(...)`. The supervisor is driven by a scripted `input_fn` that replays three REPL commands (`search(...)`, `retry researcher: ...`, `continue`), so the script runs non-interactively under CI. Remove the `input_fn=` argument to get the real terminal REPL.

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

python examples/supervised_pipeline.py
```

Reference: [`docs/course/13-human-in-the-loop.md`](../docs/course/13-human-in-the-loop.md) · [`lazy_wiki/bot/13_supervisor.md`](../lazy_wiki/bot/13_supervisor.md).

---

## Human-in-the-loop with a browser UI

`human_gui_demo.py` is the same `researcher → supervisor → writer` chain
as `supervised_pipeline.py`, except the supervisor's REPL runs in a local
browser tab instead of stdin. Stdlib-only (no extra `pip install`).

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

python examples/human_gui_demo.py
```

The script prints a `http://127.0.0.1:<port>/?t=<token>` URL and opens the
page automatically. Each REPL prompt renders on the page with the previous
output and a textarea; Ctrl/⌘-Enter submits. Details:
[`lazybridge/gui/human/README.md`](../lazybridge/gui/human/README.md).

---

## Shared GUI for every LazyBridge object

`gui_demo.py` opens a single browser tab that hosts live panels for
every `LazyAgent`, `LazyTool`, and `LazySession` you call `.gui()` on.
Inspect state, edit system prompts, run chat/loop/text against the
real provider, invoke tools from a schema-generated form.

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

python examples/gui_demo.py
```

Full API: [`lazybridge/gui/README.md`](../lazybridge/gui/README.md).
