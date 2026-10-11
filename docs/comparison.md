# LazyBridge vs LangGraph vs CrewAI

These frameworks expose different interfaces for organizing work. Choose from
the workflow, state, tool and deployment contracts your application needs.
This page compares those contracts, without ranking ecosystem size, maturity
or implementation cost.

!!! info "Comparing more than three?"
    For Pydantic AI, LlamaIndex, Haystack, AutoGen, Microsoft Agent Framework
    and Google ADK, see [Best Python AI Agent Frameworks in 2026](best-python-ai-agent-frameworks.md).

Capabilities below reflect the linked official docs at the time of writing
(October 2026). Check the version and integrations you plan to deploy.

## What each framework optimises for

### LangGraph

LangGraph exposes state, nodes, edges and a functional API. Its
[overview](https://docs.langchain.com/oss/python/langgraph/overview) documents
durable execution, streaming and human-in-the-loop support.

Consider it when:

- Graph topology is a useful domain model for branching, cycles or fan-out.
- You want persisted graph state and resumable human review; read its
  [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) and
  [interrupt](https://docs.langchain.com/oss/python/langgraph/interrupts) guides.
- Your existing model, tool and deployment integrations already fit the stack.
- You plan to use [LangSmith](https://docs.langchain.com/langsmith/home) for
  tracing, evaluation or deployment services.

Tradeoffs to evaluate:

- A graph API makes topology explicit, but a simple linear task may need less
  state and edge configuration in your application.
- State updates and reducers need deliberate design, particularly at fan-in;
  see the [graph API](https://docs.langchain.com/oss/python/langgraph/graph-api).
- Dependency and deployment requirements depend on the selected model adapters,
  tools and checkpoint backend. LangGraph does not require LangChain's agent API.

### CrewAI

CrewAI describes agents through roles/goals, tasks through descriptions and
expected outputs, and crews through sequential or hierarchical processes.
See [agents](https://docs.crewai.com/en/concepts/agents),
[tasks](https://docs.crewai.com/en/concepts/tasks) and
[processes](https://docs.crewai.com/en/concepts/processes).

Consider it when:

- Roles, tasks and delegation reflect the domain people discuss.
- You want those concepts visible in the workflow definition.
- Its [Flows](https://docs.crewai.com/en/concepts/flows) fit event-driven
  state, routing, parallel listeners and persistence.
- Stakeholders find role/goal/task descriptions useful during review.

Tradeoffs to evaluate:

- Decide whether a Crew, a Flow, or a combination best expresses the workflow.
- Translate provider-specific options through the selected model integration;
  test any behavior your application depends on.
- Define human approval and final acceptance explicitly. Task human input and
  guardrails are not a substitute for application release criteria.

### LazyBridge

LazyBridge separates `Agent = Engine + Tools + State`. Agent calls return an
Envelope; composition helpers return Agents. Deterministic Plans, coding
runtimes, provider APIs and human decisions implement the engine boundary.

Consider it when:

- **Claude Code and Codex should be engines inside Python**, sharing the same
  application tools, Memory, Store and observability Session.
- One engine writes and the other reviews through `verify=`, with rejection
  feedback bounded by `max_verify` total attempts.
- Native sessions/threads need durable handles or `session_alias` names.
- Agent-as-Tool recursion, linear chains, parallel joins and declared Plans
  match your workflow's interfaces.
- Human input or supervision should be another engine choice.
- [LazyTools](https://tools.lazybridge.com/) should own connector/MCP and
  repository integration, while Core owns composition.

Tradeoffs to evaluate:

- Core supplies exporters rather than a first-party hosted observability UI.
- Integrations and guarded external actions live in LazyTools, with separate
  installation and API contracts.
- Shared Python state does not migrate Claude's native transcript into Codex.
  Native tools and permissions remain runtime-specific.
- Native transcript locking is in-process; cross-process coordination is the
  caller's responsibility. Uncertain stateful turns require inspection.
- Plan checkpoint/resume covers declared step boundaries. It is not a general
  durable execution service or a promise of exactly-once external side effects.

## The feature matrix at a glance

| Dimension | LangGraph | CrewAI | LazyBridge |
|---|---|---|---|
| Core abstraction | State/node/edge; functional API | Agent/task/crew plus Flow | Agent/engine/tools/state |
| Models | Selected provider integrations | Selected provider integrations | LLMEngine catalogue or local Claude/Codex runtime |
| Structured output | Model and typed-state interfaces | Task output_pydantic/output_json | Agent output=PydanticModel; native engine schema where representable |
| Cross-model review | Express a reviewer in workflow logic | Express a reviewer task/Flow | verify=judge with bounded feedback |
| Sub-agent composition | Nodes/subgraphs | Delegation and hierarchical processes | tools=[named_agent], chain, parallel, Plan |
| Conditional routing | Conditional edges/Command | Flow routing | Step routes/routes_by |
| Human-in-the-loop | Interrupt with persisted state | Task human input and Flow/application hooks | HumanEngine, SupervisorEngine and approval gates |
| Dependencies | Runtime plus selected integrations | Runtime plus selected integrations | Core plus selected provider/engine/extension extras |

Sources: [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts),
[CrewAI tasks](https://docs.crewai.com/en/concepts/tasks) and
[Flows](https://docs.crewai.com/en/concepts/flows).

??? tip "More dimensions to evaluate"

    | Dimension | LangGraph | CrewAI | LazyBridge |
    |---|---|---|---|
    | Tool schemas | Tool/model integration | Tool declarations | Type hints/docstrings or explicit Tool schema |
    | Sequential pipeline | Edges or functional workflow | Sequential process or Flow | Agent.chain or Plan |
    | Parallel fan-out | Graph supersteps/Send | Async tasks and parallel Flow listeners | Agent.parallel or Plan bands |
    | Structural validation | Graph compile checks | Task/Flow validation plus application checks | Plan compile; tool-map checks when bound to Agent |
    | Persistence | Checkpointers and threads | Flow persistence | Store/checkpoint; native sessions/threads separately |
    | Streaming | Runtime stream modes | Crew/agent streaming interfaces | agent.stream; engine-specific native events |
    | Observability | LangSmith or instrumentation | [CrewAI tracing](https://docs.crewai.com/en/observability/tracing)/integrations | Opt-in Session; JSON/console/custom/OTel exporters |
    | MCP | [LangChain tool integration](https://docs.langchain.com/oss/python/langchain/mcp) | [MCP tooling](https://docs.crewai.com/en/mcp/overview) | LazyTools MCP providers; Claude internal adapter |
    | Native provider tools | Selected model integration | Selected model integration | LLMEngine.native_tools; coding-engine native policies |
    | Coding-engine continuity | Application integration | Application integration | persist_session/persist_thread; SessionRegistry aliases |

    Read the current [CrewAI crew](https://docs.crewai.com/en/concepts/crews)
    and [Flow](https://docs.crewai.com/en/concepts/flows) docs for the selected
    execution interface. These cells do not score implementation quality.

## Same task, three frameworks: a linear workflow

For research → write → edit, each framework can express the same behavior.
The useful comparison is where instructions, state and transitions live.

| Concern | LangGraph | CrewAI | LazyBridge |
|---|---|---|---|
| Composition | Nodes with sequential edges | Tasks in a sequential Crew, or a Flow | Agent.chain(a, b, c), or explicit Plan steps |
| Instructions | Model/node logic | Role/goal/task description | Engine system instructions and task/context |
| Handoffs | State fields and update/reducer rules | Task context or Flow state | Envelopes, from_step and typed Step inputs |
| Branching extension | Conditional edges/Command | Flow router/listeners | Step routes/routes_by and after_branches |
| Review extension | Reviewer node/workflow | Reviewer task/Flow | verify= on an Agent or tool view |

Sources: [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
and [CrewAI processes](https://docs.crewai.com/en/concepts/processes).

There is no fixed concept-count or line-count advantage: the implementation
depends on the selected APIs and requirements. Here is the LazyBridge linear
shape with deterministic mock workers, runnable without credentials:

```python
from lazybridge import Agent
from lazybridge.testing import MockAgent

research = MockAgent("Sources collected", name="research")
write = MockAgent("Draft written", name="write")
edit = MockAgent("Draft edited", name="edit")
pipeline = Agent.chain(research, write, edit, name="article")
result = pipeline("Explain deployment policy")
assert result.ok and result.text() == "Draft edited"
```

Use [Plans](guides/full/plan.md) when you want explicit routing, parallel bands,
typed handoffs or checkpoints. Both other frameworks also support workflows
beyond this linear example.

## Choose your framework: decision tree

```text
Which interface is central to your application?
├── Explicit graph state/topology and checkpointed interrupts
│   └── Evaluate LangGraph's graph, persistence and deployment interfaces.
├── Roles/tasks/processes or event-driven Flows
│   └── Evaluate CrewAI's Crew and Flow interfaces.
└── Engines sharing recursive Tools, state and Envelopes
    └── Evaluate LazyBridge's Plans, coding runtimes and verification.

Then test your actual persistence, approval and deployment requirements.
If several fit, existing integrations and team familiarity may decide.
```

Persistence alone does not select a winner: the frameworks persist different
units of work. Compare restart boundaries and side-effect semantics directly.

## When LazyBridge is the wrong choice

1. **You require a specific hosted observability/deployment platform.**
   Check its supported runtimes. LazyBridge's open exporters do not themselves
   provide a hosted UI. [LangSmith](https://docs.langchain.com/langsmith/home)
   documents its own tracing, evaluation and deployment services.

2. **Graph topology is your team's primary way to understand the system.**
   Evaluate LangGraph's graph interface. There is no node-count threshold at
   which a Plan becomes inappropriate; readability depends on the workflow.

3. **Roles and task descriptions are your review language.**
   Evaluate CrewAI's agent/task interfaces with the people maintaining them.
   A stakeholder preference is a project constraint, not a framework ranking.

4. **Your application already relies on another stack's state and integrations.**
   Estimate migration and operational costs before switching. A shorter example
   is not evidence of savings in a deployed application.

## Migration paths

### From LangChain / LangGraph to LazyBridge

- **Model calls:** move instructions/options to LLMEngine, or choose a coding
  engine. This is a conceptual mapping, not a drop-in provider-object conversion.
- **Sequential graphs:** Agent.chain can express a linear sequence; Plan is
  appropriate for declared data flow and routing.
- **Conditional edges:** model the branch with Step routes/routes_by and an
  explicit reconvergence policy. Test the translated behavior.
- **Tools:** extract the underlying typed callable or build a Tool adapter;
  do not assume every third-party decorated object is directly wrappable.
- **State:** Memory holds conversation history; Store holds application records.
  Sentinels describe Plan handoffs. These do not automatically reproduce graph
  reducer semantics or migrate checkpoints.
- **Observability:** choose Session/exporters and a backend before migrating
  dashboards or acceptance checks.

### From CrewAI to LazyBridge

- **Agents:** translate role/goal/backstory into engine instructions where
  appropriate; preserve requirements rather than mechanically concatenate them.
- **Tasks:** pass task and context separately; expected_output is not itself a
  schema. Use output= and validators for typed acceptance.
- **Sequential Crews:** use Agent.chain or an explicit Plan.
- **Hierarchical Crews:** named sub-agents can be Tools of a coordinating Agent.
  This does not reproduce every delegation or manager behavior automatically.
- **Flows:** translate events, routing and persistence deliberately; a Plan's
  checkpoints are not a migration format for Flow state.
- **Tools:** wrap a callable or implement an adapter. Validate argument schemas,
  approval behavior and side effects before using it in the new workflow.

### From LazyBridge to LangGraph or CrewAI

Functions, typed outputs and explicit task/context contracts can be reused.
Wrap Agent.run as node/task application logic, or rebuild the model/tool loop
using the target integration. Translate Memory, Store, native IDs, checkpoints
and Session records explicitly; migration effort depends on those contracts.

## What LazyBridge intentionally doesn't do

- Core does not ship a proprietary hosted tracing/evaluation platform.
  Use open exporters with a backend you operate or purchase.
- Core does not own vector indexing or retrieval infrastructure. Wrap a
  retriever as a Tool or use a LazyTools integration.
- System instructions are Python strings; Core does not require a prompt
  template engine. Python formatting or your chosen library can supply one.
- Core does not ship an agent marketplace or require predefined personas.
- Native transcript migration, cross-process transcript locks and automatic
  replay of uncertain side-effecting turns are outside the current engine contract.

## Claude Code writes, Codex reviews

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

reviewer = Agent(name="reviewer", engine=CodexEngine(
    system='Return "approved" or "rejected: <reason>". Check edge cases.',
))
writer = Agent(name="writer", engine=ClaudeCodeEngine(), verify=reviewer, max_verify=3)
print(writer("Draft a weekday deployment policy.").text())
```

This loop repairs output through feedback; exhaustion returns the final draft,
so result.ok is not release approval. Use tests and explicit acceptance records.
For native continuity choose persistence or [aliases](guides/full/session-registry.md).
For repository jobs, durable tickets and opposite-engine review routing, see the
[LazyTools code bridge](https://tools.lazybridge.com/code-bridge/).
The quota routing benchmark is **in progress**.

## Selection guide

| Framework | Evaluate when |
|---|---|
| LangGraph | Graph state, topology and checkpointed interrupts are central |
| CrewAI | Role/task/crew or Flow interfaces match your domain |
| LazyBridge | Coding engines, provider APIs and Plans should share Python Tools, state and bounded review |

[**Get started →**](quickstart.md){ .md-button .md-button--primary }
[Why LazyBridge](why.md){ .md-button }
