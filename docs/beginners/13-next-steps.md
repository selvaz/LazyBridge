# Step 13: Where to go next

You did it. Twelve steps in, you've seen every basic LazyBridge primitive
a developer hits in the wild:

- **The single agent** — `Agent`, `LLMEngine`, `Envelope`, `output=`, `verify=`, `verbose=`
- **Tools** — type-hinted Python functions, `Tool.wrap`, the implicit loop
- **The four composition patterns** — sub-agent-as-tool, `Agent.chain`,
  `Agent.parallel`, `Plan` + sentinels
- **Conditional flow** — `routes=` / `routes_by=` / `after_branches=` / `when` DSL
- **Cross-model verification** — `verify=judge_agent` with a different LLM family
- **Human-in-the-loop** — `HumanEngine` / `human_agent`
- **Honest comparison** — when to reach for LazyBridge, LangGraph, or CrewAI

That's enough to build real production agents. Everything from here is
either **deeper** (the same primitives, with more knobs) or **further out**
(production concerns: observability, evaluation, persistence, distributed
runs).

This page is your map.

---

## A short suggested first project

If you want to consolidate what you learned before opening the deeper
docs, build something like this end-to-end:

> **A "pull request reviewer" agent.**
> Given a PR URL, fetch the diff, classify it (`docs` / `bugfix` /
> `feature` / `risky`), run different reviewers for each category, ask a
> human for sign-off on `risky` PRs, and post a final summary comment.

It exercises: tools (`fetch_diff`, `post_comment`), structured output
(`Classification` model), `routes_by=` (category branching),
`HumanEngine` (the `risky` gate), `verify=` (a second-pass quality
judge), and observability (you want to see *every* decision in the
trace).

Skim the [recipes](../recipes/index.md) for similar patterns to crib
from.

---

## The four tiers of LazyBridge docs

After this beginner section, the rest of the documentation is organised
by **how much capability you need**. Each tier reuses everything from
the previous one — no surprises.

### Basic — primitives reference

[Guides → Basic](../guides/basic/agent.md) — formal reference for what
you saw in Steps 3–4:

- [Agent](../guides/basic/agent.md) — full constructor signature, every
  parameter, with examples
- [Tool](../guides/basic/tool.md) — `Tool.wrap`, `mode="signature"` /
  `"llm"` / `"hybrid"`, schema introspection edge cases
- [Envelope](../guides/basic/envelope.md) — every field, error envelopes,
  payload shapes, metadata semantics
- [Native tools](../guides/basic/native-tools.md) — `NativeTool.WEB_SEARCH`,
  `CODE_EXECUTION`, `COMPUTER_USE` and the safety opt-in

Read these when you want the **exhaustive surface** of what you already
roughly know.

### Mid — common production needs

[Guides → Mid](../guides/mid/memory.md) — features most apps end up needing:

- [Memory](../guides/mid/memory.md) — conversation continuity across calls
- [Store](../guides/mid/store.md) — cross-run / cross-agent state persistence
  (the same `from_agent("name")` sentinel from Step 9 reads this)
- [Session](../guides/mid/session.md) — event tracking and observability
  hooks for the whole run
- [Guards](../guides/mid/guards.md) — hard policy gates (the sibling of
  `verify=` from Step 6)
- [verify=](../guides/mid/verify.md) — the deep reference for Step 6,
  including the three placement variants
- [Chain](../guides/mid/chain.md), [Parallel](../guides/mid/parallel.md),
  [As tool](../guides/mid/as-tool.md) — formal docs for Step 5/7/8
- [HumanEngine](../guides/mid/human-engine.md) — the deep reference for
  Step 11, including the custom UI protocol
- [MCP](https://tools.lazybridge.com/mcp/) — Model Context Protocol servers as tools
- [Multimodal](../guides/mid/multimodal.md) — images and audio as inputs
- [Evals](../guides/mid/evals.md) — evaluation harness for agent quality

### Full — the deep composition layer

[Guides → Full](../guides/full/plan.md) — everything `Plan` and routing
can do beyond what Steps 9–10 covered:

- [Plan](../guides/full/plan.md) — full Plan semantics, including
  per-step `writes=`, `input=`, `output=`, `sources=`
- [Step](../guides/full/step.md) — all Step parameters in detail
- [Sentinels](../guides/full/sentinels.md) — `from_prev`, `from_step`,
  `from_start`, `from_agent`, `from_memory`, `from_parallel`,
  `from_parallel_all`
- [Routing](../guides/full/routing.md) — beyond `routes_by=`:
  multi-predicate routing, nested branches, loop control
- [Parallel plan steps](../guides/full/parallel-plan-steps.md) — bands,
  fan-in via `from_parallel`, concurrency limits
- [Checkpoint & resume](../guides/full/checkpoint.md) — persisted Plan
  state, multi-day workflows, `on_concurrent` policy
- [Exporters](../guides/full/exporters.md) — `EventExporter`,
  `JsonFileExporter`, `OTelExporter`, custom callback exporters
- [GraphSchema](../guides/full/graph-schema.md) — typed payload contracts
  between steps
- [SupervisorEngine](../guides/full/supervisor.md) — the REPL-style HIL
  engine teased at the end of Step 11

### Advanced — when you're building infrastructure

[Guides → Advanced](../guides/advanced/engine-protocol.md) — for
contributors, integrators, and people running LazyBridge at scale:

- [Engine protocol](../guides/advanced/engine-protocol.md) — write your
  own engine (custom orchestration semantics)
- [BaseProvider](../guides/advanced/base-provider.md) — add a new LLM
  provider (the same surface OpenAI/Anthropic/Gemini/DeepSeek/LM Studio
  use)
- [Providers](../guides/advanced/providers.md) — the built-in provider
  catalogue, tier aliases, pricing tables
- [External tool gateway](https://tools.lazybridge.com/gateway/) —
  registering remote tools via the HTTP gateway
- [Plan serialization](../guides/advanced/plan-serialize.md) — saving
  Plans to JSON / YAML
- [OpenTelemetry](../guides/advanced/otel.md) — production tracing setup
- [Visualizer](../guides/advanced/visualizer.md) — live UI for agent runs

---

## Recipes — patterns to crib from

[Recipes](../recipes/index.md) is the "tested code, copy and adapt"
section:

- [React agent](../recipes/react-agent.md) — Step 4 deepened
- [Researcher (single agent)](../recipes/researcher-single.md) — Step 5
  flavour
- [Researcher → reporter](../recipes/researcher-reporter.md) — Step 7 in
  full
- [Supervisor pattern](../recipes/supervisor-pattern.md) — SupervisorEngine
  in practice
- [Plan tool](../recipes/plan-tool.md), [Agent builds a plan](../recipes/agent-builds-plan.md) —
  meta-planning
- [Blackboard planner](../recipes/blackboard-planner.md),
  [Dynamic re-planning](../recipes/dynamic-replanning.md) — when the
  workflow shape itself adapts
- [Live visualization](../recipes/live-visualization.md),
  [Visualization mock](../recipes/visualization-mock.md) — the
  Visualizer in practice

Most recipes are also runnable scripts in the
[`examples/` directory](https://github.com/selvaz/LazyBridge/tree/main/examples).

---

## Decision pages — "should I use X or Y?"

When the choice between two LazyBridge features isn't obvious, the
[Decisions](../decisions/index.md) section has short opinionated answers:

- [Pick your tier](../decisions/pick-tier.md) — Basic vs Mid vs Full
- [Return type](../decisions/return-type.md) — when to use `output=`
  vs plain string
- [State layer](../decisions/state-layer.md) — `Memory` vs `Store` vs
  sentinels
- [Composition](../decisions/composition.md) — `chain` vs `Plan` vs
  sub-agent-as-tool
- [Parallelism](../decisions/parallelism.md) — `Agent.parallel` vs
  parallel tool calls
- [HumanEngine vs SupervisorEngine](../decisions/human-engine-vs-supervisor.md) —
  when each fits
- [`verify=` placement](../decisions/verify-placement.md) — agent-level
  vs tool-level vs plan-level
- [Checkpoint & resume](../decisions/checkpoint.md) — when persistence
  is worth the ceremony
- [Do I need Advanced?](../decisions/need-advanced.md) — escape hatch
  to deeper APIs

---

## Reference — when you need the exact signature

[Reference](../reference/index.md) is auto-generated from the source
docstrings. Use it as a search target:

- [Agent + Envelope](../reference/agent.md)
- [Tool family](../reference/tools.md)
- [State primitives](../reference/state.md)
- [Session & observability](../reference/session.md)
- [Guards](../reference/guards.md)
- [Engines](../reference/engines.md)
- [Sentinels & predicates](../reference/sentinels.md)
- [Extensions](../reference/extensions.md) — `lazybridge.ext.*`
- [Custom providers](../reference/providers.md)
- [Configs & testing](../reference/configs.md) — `MockAgent`,
  test utilities

---

## For LLM assistants (Claude / Cursor / Copilot)

If you generate LazyBridge code with an LLM, point it at:

- [`llms.txt`](https://core.lazybridge.com/llms.txt) — concise index
- [`llms-full.txt`](https://core.lazybridge.com/llms-full.txt) — full
  consolidated docs in one file
- [Codegen contract](../for-llms/codegen-contract.md) — strict
  conventions (canonical imports, what to avoid, what to prefer)
- [Claude Skill install](../for-llms/claude-skill.md) — drop-in skill
  for Claude that teaches the codegen rules
- [Error recovery cheat-sheet](../for-llms/error-recovery.md) — common
  error → fix mappings

These pages exist so an LLM writing your LazyBridge code produces
canonical, current-version code instead of an `0.4`-era pastiche of
`LazyAgent` and `mode="auto"`.

---

## Stay current

- **CHANGELOG** — [github.com/selvaz/LazyBridge/blob/main/CHANGELOG.md](https://github.com/selvaz/LazyBridge/blob/main/CHANGELOG.md).
  Versioned, with concrete migration notes for every breaking change.
- **Migrations** — the [migrations folder](../migrations/0.7-to-0.79.md)
  has dedicated upgrade guides between minor versions.
- **GitHub** — [github.com/selvaz/LazyBridge](https://github.com/selvaz/LazyBridge).
  Issues, discussions, and the public roadmap.

---

## A final word

This tutorial deliberately stayed in the **basic** tier. There's a real
risk of "tutorial-fatigue", where a beginner section keeps adding
edge-case features instead of stopping when the foundation is solid.
The foundation is solid now — every step's content stays unchanged when
you graduate to the mid/full/advanced guides; the deeper docs add knobs,
they don't change the surface.

Build something with what you have. Come back to the guides when a
specific feature seems missing — and most of the time, it'll already be
there, just under a name you hadn't met yet.

Good luck, and welcome to LazyBridge.

---

[← LazyBridge vs LangGraph vs CrewAI](../comparison.md){ .md-button }
[Start over →](index.md){ .md-button }
