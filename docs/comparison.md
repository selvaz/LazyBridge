# LazyBridge vs LangGraph vs CrewAI

Three different frameworks, three different bets. This page gives you a
**decision framework** for picking the right tool for your project and is
honest about when LazyBridge *isn't* it.

> The three frameworks optimise for **different things**. There's no
> winner in the abstract — there's the one that fits *your* project's
> shape, team, and constraints.

!!! info "Comparing more than three?"
    This page is the focused, decision-tree head-to-head of LangGraph, CrewAI
    and LazyBridge. For the broader category view — Pydantic AI, LlamaIndex,
    Haystack, AutoGen, Microsoft Agent Framework and Google ADK alongside these —
    start from [Best Python AI Agent Frameworks in 2026](best-python-ai-agent-frameworks.md).

---

## What each framework optimises for

### LangGraph

**Optimised for: complex, stateful, long-running agentic workflows where
graph control is the dominant concern.**

LangGraph is a graph DSL. You declare nodes, edges, state schemas, and
reducers. The framework runs your graph; the framework gives you very
fine-grained control over what happens between nodes, when state persists,
when to interrupt, and what to stream.

It shines when:

- The workflow shape is genuinely a graph — multiple entry points,
  complex branching, cycles, fan-in/fan-out at non-trivial scales
- You need first-class **streaming, checkpointing, and resumability** —
  long-running agents that survive restarts, multi-day workflows, HITL
  with persisted state
- You're already in the **LangChain ecosystem** (300+ integrations) and
  want continuity with `langchain_*` packages
- **LangSmith** observability is a requirement (custom tracing UI,
  evaluation dashboards, prompt management)

It costs you:

- Verbose for simple cases — the graph wiring is overhead when your
  workflow is "do A, then B, then C"
- A learning curve — `StateGraph`, `Annotated[...]` reducers, `Send`,
  `Command`, conditional edges, checkpointer config
- Transitive dependencies through `langchain_core`, `langchain_*`
  provider adapters, and any tools you wire in

### CrewAI

**Optimised for: business-style "team of experts" mental models, where
the abstractions match how humans describe the work.**

CrewAI builds on the metaphor *role, goal, backstory*. An Agent has a
role like "Senior Researcher". A Task has a description and an expected
output. A Crew runs Tasks via a `Process` — sequential or hierarchical.

It shines when:

- The use case fits the **roles + tasks + crew** metaphor well
  (research teams, content production, business workflows, structured
  pipelines that map cleanly to org-chart roles)
- You want a **strong opinion** that gives you guardrails and patterns
  out of the box — not a kit of primitives to assemble
- **Non-engineer stakeholders** read the agent code; the role/goal
  metaphor is more legible than "an Agent with system prompt and tools"

It costs you:

- Lock-in to the Crew/Task abstraction — escaping to the underlying
  provider API isn't a smooth path
- **No native conditional routing** — you reach for plain Python
  branching (Step 10's CrewAI comparison)
- **No parallel execution** of agents at the framework level — you
  drop to `asyncio.gather`
- HIL is limited to `human_input=True` on a Task; no web UI, no
  typed Pydantic forms, no timeout/default story
- The opinionated abstractions can chafe when your shape is **not** a
  team of experts

### LazyBridge

**Optimised for: composition primitives — small, orthogonal building
blocks that combine into any workflow shape with minimal ceremony.**

LazyBridge has one core abstraction (`Agent = Engine + Tools + State`)
and a small set of composition primitives that **all return the same
shape** — every primitive returns an Envelope, every composition returns
an Agent.

It shines when:

- You want **multi-provider out of the box** — change one string to
  switch between Claude, GPT, Gemini, DeepSeek, Mistral, LM Studio
- You want **cross-model verification** to be a one-line concern
  (`verify=judge_agent` works with any model family — Step 6)
- The workflow is composition-heavy but the *individual pieces* aren't
  unusual (chain, parallel, sub-agent-as-tool, simple routing)
- You want **minimal dependencies** (no LangChain object hierarchy, no
  Crew opinions) and a small mental model
- **Code review readability** matters — every LazyBridge primitive
  reads like "what is happening" not "what graph wiring is happening"
- HIL is a first-class engine swap, not a bolted-on flag

It costs you:

- A **smaller community** than LangGraph or CrewAI — fewer Stack Overflow
  answers, fewer tutorials in the wild, fewer integrations packaged
- **No first-party observability platform** like LangSmith yet — you
  bring your own exporters (OTel, structured logs, JSON files — all
  built in, but no hosted UI shipped by LazyBridge)
- **Less battle-tested** at the very-large-graph scale that LangGraph
  was built to handle

---

## The feature matrix at a glance

The eight dimensions most projects actually weigh when choosing:

| Dimension | LangGraph | CrewAI | LazyBridge |
|---|---|---|---|
| Core abstraction | Graph (nodes + edges + state) | Crew (agents + tasks + process) | Agent (engine + tools + state) |
| Multi-provider | via `langchain_*` adapters | via underlying SDK | **built-in (one string)** |
| Structured output | manual State plumbing | task `expected_output=` | **`output=PydanticModel`** |
| Cross-model verify | manual loop | manual loop | **`verify=judge_agent`** |
| Sub-agent composition | wrap a node | hierarchical process | **`tools=[other_agent]` / chain / parallel** |
| Conditional routing | `add_conditional_edges` | none — plain Python | **`routes=` / `routes_by=`** |
| Human-in-the-loop | `interrupt()` + checkpointer | `human_input=True` (terminal only) | **`HumanEngine` (terminal/web/custom)** |
| Dependencies | langchain-core + N adapters | crewai + langchain-tools | **stdlib + pydantic + httpx** |

??? tip "Twelve more dimensions, for the curious"

    Compile-time validation, built-in checkpointing, streaming, built-in
    OTel, MCP server tools, provider-native tools (web search /
    code-exec), parallel fan-out, sequential pipeline expression, tools
    schema generation, simple-call line count, ecosystem size, maturity:

    | Dimension | LangGraph | CrewAI | LazyBridge |
    |---|---|---|---|
    | Simple call | ~10 lines (StateGraph) | ~12 lines (Agent + Task + Crew) | **3 lines** |
    | Tools (schema) | manual schema + ToolNode | `@tool` decorator | **type hints + docstring** |
    | Sequential pipeline | linear `add_edge` graph | `Process.sequential` | **`Agent.chain`** |
    | Parallel fan-out | `Send` + reducer State | drop to `asyncio.gather` | **`Agent.parallel`** |
    | Compile-time validation | yes (StateGraph compile) | partial | yes (Plan compile) |
    | Built-in checkpointing | yes (multiple backends) | no | yes (Plan checkpoint) |
    | Streaming | yes (rich modes) | partial | yes (`agent.stream`) |
    | Built-in OTel | via LangSmith | no | yes (`OTelExporter`) |
    | MCP server tools | via tooling layer | via custom | **built-in (`MCP.stdio`)** |
    | Native tools (web search etc.) | per-provider plumbing | limited | **built-in (`NativeTool.*`)** |
    | Ecosystem | very large (LangChain) | medium | small (focused) |
    | Maturity | high | medium-high | early but stable |

For most decisions, the eight dimensions above are enough. The decision
tree below is a faster route to "which one" than the matrix.

---

## Same task, three frameworks — concept count *(simple linear pipeline)*

The classic three-step pipeline (research → write → edit) one last time,
counting the **distinct concepts** a beginner has to learn:

| | LangGraph | CrewAI | LazyBridge |
|---|---|---|---|
| Composition expression | `add_node × 3 + add_edge × 4 + START + END` | `Crew(agents=, tasks=, process=)` | `Agent.chain(a, b, c)` |
| Required concepts | StateGraph, TypedDict, add_messages, Annotated reducer, nodes, edges, START, END, compile, invoke, state['messages'] plumbing | Agent, role, goal, backstory, Task, description, expected_output, context, Crew, process | Agent, LLMEngine, chain |
| Total concept count | ~11 | ~10 | **3** |

!!! note "What this count means"
    This isn't about lines of code — it's about how many things a new
    reader has to load into their head before they can change anything.

    **The numbers above are for a 3-stage *linear* pipeline.** For a
    branching workflow with routing, parallel bands, and HIL, every
    framework expands — LangGraph adds reducers and conditional edges,
    CrewAI adds Process flavours and custom hooks, LazyBridge adds
    `routes_by=` / `after_branches=` / `Plan` / `HumanEngine`. The
    *gap* tends to stay roughly constant; LazyBridge doesn't stay at
    "3 concepts" as you grow.

    LangGraph and CrewAI **earn the extra concept count** on
    non-trivial graphs — they're not paying for nothing.

---

## Choose your framework — decision tree

```
Do you need long-running, stateful, persistable workflows
(workflows that survive restarts / span days)?
│
├── YES ──► LangGraph (checkpointer ecosystem is mature)
│
└── NO ──►  Does the team think in "role + goal + backstory" terms,
            and the workflow shape is mostly a team-of-experts pipeline?
            │
            ├── YES ──► CrewAI (role metaphor is its strength)
            │
            └── NO ──►  Do you need:
                        - multi-provider as a first-class concern, or
                        - cross-model verification, or
                        - small dependency footprint, or
                        - composition primitives that all stay the same shape?
                        │
                        ├── YES (any of) ──► LazyBridge
                        │
                        └── NO ──────────────► Pick the one your team already knows.
                                              Migration cost > framework choice cost.
```

---

## When LazyBridge is the wrong choice

To stay credible, four cases where you should reach for something else:

1. **You need first-party hosted observability today.**
   LangSmith (LangChain) has the most polished platform for tracing,
   evaluation, and prompt management. LazyBridge ships exporters
   (OTel / JSON / console / custom callback) but no hosted UI.

2. **Your workflow has > 30 nodes with complex branching.**
   At that scale LangGraph's explicit graph DSL pays for itself.
   LazyBridge's `Plan` can express the same thing but the
   resulting code is denser to read.

3. **Stakeholders read the agent code.**
   If non-engineers are reviewing the workflow definitions, CrewAI's
   role/goal/backstory metaphor is a real win — it reads like a job
   description.

4. **You're deeply invested in the LangChain ecosystem.**
   `langchain_*` adapters, `langchain-community` tools, LangSmith,
   LangGraph Cloud — if you've built infrastructure on this stack,
   the switching cost outweighs the surface-area savings.

These are real cases. Be honest about whether they apply.

---

## Migration paths

If you're already on one of the others, what does moving look like?

### From LangChain / LangGraph to LazyBridge

- **Agent core:** `ChatAnthropic / ChatOpenAI + bind_tools → Agent(engine=LLMEngine(...), tools=[...])`. One-to-one mapping.
- **Sequential graphs:** `StateGraph + add_edge × N → Agent.chain(a, b, c)`.
- **Conditional edges:** `add_conditional_edges + router → Step(routes=)` or `routes_by=`.
- **Tools:** `@tool` decorator → plain Python function with type hints + docstring.
- **State:** if you used `MessagesState`, switch to `Memory`. For typed cross-step state, the `Store` + sentinels combo replaces State reducers.
- **Watch out:** if you used LangSmith heavily, plan your observability story before migrating (OTel exporter is the recommended path).

### From CrewAI to LazyBridge

- **Agents:** `Agent(role=..., goal=..., backstory=...)` → fold role/goal/backstory into a `system="..."` string. Often shorter.
- **Tasks:** `Task(description=..., expected_output=..., context=[...])` → just call the agent with the input string; use `Agent.chain` for sequential, `Plan` for explicit DAG.
- **Sequential Crews:** `Crew(process=Process.sequential)` → `Agent.chain(...)`.
- **Hierarchical Crews:** `Process.hierarchical` → `Agent(engine=LLMEngine(...), tools=[sub_agent_1, sub_agent_2])` (sub-agent-as-tool).
- **Tools:** CrewAI tools (and `langchain-tools` ones) are wrappable; if they were `@tool`-decorated, the underlying function works directly in LazyBridge.

### From LazyBridge to LangGraph or CrewAI

This is just as legitimate. LazyBridge isn't a religion — if your project's
shape changes (you grow into 50-node workflows, or your team prefers
explicit roles), the lift is small in the other direction too because
LazyBridge agents are thin wrappers around the same provider SDKs the
others use.

---

## What LazyBridge intentionally doesn't do

A short list of features LazyBridge *deliberately* leaves out:

- **No proprietary tracing/eval platform.** Exporters are open; the
  hosted UI isn't part of the project. Use OTel + your favourite
  backend (Honeycomb, Datadog, Jaeger).
- **No first-class "memory store" abstraction beyond `Memory` + `Store`.**
  We don't ship vector DB adapters. Bring your own retriever; wrap it
  as a tool.
- **No prompt template engine.** System prompts are Python strings.
  When you need templating, `str.format` and `f-strings` are right
  there.
- **No agent marketplaces or pre-built personas.** Build your own
  agents from the primitives. Less to memorise, less to keep in sync.

This list is short because the philosophy is: ship the primitives, let
users build the rest. Each thing we *don't* ship is a thing we *won't*
break in a future release.

---

## Summary

| Framework | Pick it when |
|---|---|
| **LangGraph** | Complex stateful graphs, long-running workflows with persistence, you're already in LangChain |
| **CrewAI** | Team-of-experts metaphor fits, opinionated abstractions are an asset, stakeholders read the code |
| **LazyBridge** | Multi-provider matters, cross-model verify matters, you want a small composition surface, minimal dependencies |

You now have all the context you need to choose. The last step points
you to where the LazyBridge docs go after this tutorial — guides,
recipes, reference, and the topics we deliberately left for later.

---

[**Get started →**](quickstart.md){ .md-button .md-button--primary }
[Why LazyBridge](why.md){ .md-button }
