---
title: Best Python AI Agent Frameworks in 2026
description: Compare graph-first, crew-first, type-first, data-first, enterprise-first and composition-first Python AI agent frameworks for building LLM workflows.
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Best Python AI Agent Frameworks in 2026",
  "description": "A technical comparison of graph-first, crew-first, type-first, data-first, enterprise-first and composition-first Python AI agent frameworks.",
  "author": { "@type": "Organization", "name": "LazyBridge Labs" },
  "publisher": {
    "@type": "Organization",
    "name": "LazyBridge Labs",
    "url": "https://lazybridge.com/"
  },
  "mainEntity": {
    "@type": "SoftwareApplication",
    "name": "LazyBridge",
    "applicationCategory": "DeveloperApplication",
    "programmingLanguage": "Python",
    "license": "Apache-2.0",
    "codeRepository": "https://github.com/selvaz/LazyBridge",
    "downloadUrl": "https://pypi.org/project/lazybridge/",
    "description": "Composition-first Python runtime for governed LLM workflows."
  },
  "about": [
    {"@type": "SoftwareApplication", "name": "LangGraph"},
    {"@type": "SoftwareApplication", "name": "CrewAI"},
    {"@type": "SoftwareApplication", "name": "Pydantic AI"},
    {"@type": "SoftwareApplication", "name": "LlamaIndex"},
    {"@type": "SoftwareApplication", "name": "LazyBridge"}
  ]
}
</script>

# Best Python AI Agent Frameworks in 2026

No Python AI agent framework is best for every project.

The right choice depends on the shape of the system you are building: a graph, a team of agents, a typed application, a RAG pipeline, an enterprise workflow, a multi-agent conversation, or a composition-first runtime that grows from a function call into a governed service.

This guide compares the main Python AI agent frameworks by their primary design metaphor.

!!! tip "Looking for a focused 1-to-1 decision?"
    This page is the **category hub** — a broad comparison across eight frameworks.
    For a deeper, decision-tree-style head-to-head of just three, see
    [LazyBridge vs LangGraph vs CrewAI](comparison.md).

## Quick comparison

| Framework | Primary metaphor | Best for | Avoid if |
|---|---|---|---|
| LangGraph | Graph-first | Long-running, stateful graph workflows with persistence, human-in-the-loop and durable execution | You want the smallest possible mental model for simple workflows |
| CrewAI | Crew-first | Role-based teams of agents, tasks, processes and business automation | Your workflow does not map naturally to roles and tasks |
| Pydantic AI | Type-first | Type-safe Python agents, structured outputs, dependency injection and validation | Orchestration is more important than typed application structure |
| LlamaIndex | Data-first | RAG-heavy agents, private data, indexes, query engines and retrieval workflows | Your core problem is not retrieval or knowledge grounding |
| Haystack | Pipeline-first | Production RAG and search-heavy pipelines | You want a compact agent runtime rather than a broader retrieval pipeline framework |
| Microsoft Agent Framework | Enterprise-first | Microsoft, Azure, Semantic Kernel and enterprise workflow environments | You want a small independent Python-first runtime |
| Google ADK | Google-first | Gemini and Google Cloud-oriented agent systems | You want provider-neutral primitives from the start |
| AutoGen | Conversation-first | Multi-agent conversations and collaborative agent interactions | You want deterministic orchestration and tool composition as the base layer |
| LazyBridge | Composition-first | Governed Python LLM workflows where agents, functions, plans, humans, MCP servers and external systems should compose through one tool interface | You need the largest mature ecosystem and community today |

## How to choose

Choose **LangGraph** if your workflow is truly a graph: long-running state, branching, loops, human review, checkpointing and fine-grained runtime control.

Choose **CrewAI** if your workflow reads naturally as a team: researcher, analyst, writer, reviewer, manager.

Choose **Pydantic AI** if your main problem is reliable typed outputs, dependency injection, validation and IDE-friendly Python application structure.

Choose **LlamaIndex** if the agent mostly works over documents, indexes, retrieval pipelines or private knowledge.

Choose **Microsoft Agent Framework** if your environment is already centered on Microsoft, Azure, Semantic Kernel or enterprise workflow tooling.

Choose **LazyBridge** if your workflow is composition-heavy and you want one Python-first model where functions, agents, deterministic plans, humans, MCP servers and external tools share the same runtime surface.

## The core distinction: framework metaphor

Most AI agent frameworks organize systems around one dominant metaphor.

| Metaphor | Framework examples | What it optimizes |
|---|---|---|
| Graph-first | LangGraph | explicit topology, durable execution, graph state |
| Crew-first | CrewAI | agent roles, teams, tasks, business processes |
| Type-first | Pydantic AI | typed dependencies, validated outputs, IDE help |
| Data-first | LlamaIndex | retrieval, indexes, private knowledge |
| Pipeline-first | Haystack | search, RAG, production document pipelines |
| Conversation-first | AutoGen | multi-agent dialogue |
| Enterprise-first | Microsoft Agent Framework | platform workflows, deployment, governance |
| Composition-first | LazyBridge | recursive composition across agents, tools, plans, humans and services |

LazyBridge belongs in the last category: **composition-first**.

## A different category: composition-first runtimes

Most frameworks start with agents, graphs, crews, chains or workflows.

LazyBridge starts with a smaller model:

```txt
Agent = Engine + Tools + State
```

The **Engine** decides what happens next. The **Tools** define what the agent can do. The **State** carries continuity, memory, persistence, events and observability.

That split matters because LazyBridge does not treat an agent as “an LLM wrapper”.

An Agent can be driven by:

- an `LLMEngine`;
- a deterministic `Plan`;
- a `HumanEngine`;
- a `SupervisorEngine`;
- a custom `Engine` (the [engine protocol](guides/advanced/engine-protocol.md)).

This means the same `Agent` surface can represent a one-shot model call, a deterministic pipeline, a human approval step, a supervised workflow or a larger nested system.

See the [mental model](concepts/mental-model.md) for the full picture.

## LazyBridge in one sentence

LazyBridge is a composition-first Python runtime for governed LLM workflows: Core defines how work composes, LazyTools defines what agents can safely touch, and LazyPulse turns workflows into always-on policy-gated services.

## The LazyBridge ecosystem

LazyBridge is split into three focused packages.

| Package | Role | What it adds |
|---|---|---|
| LazyBridge Core | Composition runtime | `Agent`, `Engine`, `Tool`, `Envelope`, `Plan`, `Step`, `Store`, `Session`, `HumanEngine`, `SupervisorEngine`, guards, verification, tracing |
| LazyTools | Capability layer | Gmail, Telegram, MCP, external gateways, document readers, skills, allowlists, confirmation gates |
| LazyPulse | Governed service layer | tick loop, inbound adapters, trust policy, task lifecycle, human review and audit trail |

The split is intentional:

- Core does not need to import connector packages.
- Tools does not need to know about the scheduler.
- Pulse runs Agents and applies policy before external work reaches the model.

A compact way to understand the ecosystem:

```txt
Core = how work composes
Tools = what the system can touch
Pulse = when work runs and under which policy
```

## Original design ideas in LazyBridge

### 1. Agents are not LLMs

In LazyBridge, an Agent is not synonymous with a model call.

The Agent is a runtime shell. The engine decides what kind of decision-making happens inside it.

That engine can be dynamic, deterministic, human-driven or supervised.

```python
from lazybridge import Agent, LLMEngine, Plan, Step

dynamic_agent = Agent(
    engine=LLMEngine("claude-sonnet-4-6"),
)

deterministic_pipeline = Agent(
    engine=Plan(Step("research"), Step("write")),
    tools=[research_agent, writer_agent],
    name="pipeline",
)
```

The outside surface is still an Agent.

### 2. Pipelines are agents

LazyBridge does not need a separate pipeline type.

A pipeline is an Agent whose engine is a `Plan`.

Because Agents are Tools, a pipeline can be nested inside a larger agent or a larger plan without glue code. See [layered composition](concepts/layered-composition.md).

```txt
Plan-backed Agent → Tool → Step in a larger Plan
```

This makes composition structural rather than syntactic.

### 3. Tools are recursive

LazyBridge uses one recursive primitive: `Tool`.

A Tool can be:

- a Python function;
- a callable;
- another Agent;
- the same Agent under an alias;
- a Plan-backed Agent;
- a provider-native capability;
- an MCP server;
- a pre-built JSON schema;
- an external tool catalogue.

The consuming agent sees the same Tool contract in every case.

This is the practical meaning of [“everything is a tool”](concepts/everything-is-a-tool.md).

### 4. Envelope makes runs observable

Every agent call returns an [`Envelope`](guides/basic/envelope.md).

An Envelope carries:

- task;
- context;
- multimodal input;
- typed payload;
- metadata;
- error state;
- token counts;
- cost;
- latency;
- model;
- provider;
- run id;
- nested token and cost rollups.

The important part is transitive observability: when an agent calls another agent as a tool, that nested run does not disappear. Its cost, tokens, errors and latency roll up into the parent Envelope.

### 5. Plan separates orchestration from reasoning

A [`Plan`](guides/full/plan.md) is a deterministic engine.

It declares steps, data flow, routes, typed handoffs, parallel bands, store writes and checkpoint/resume behavior.

Use an LLM when the model should decide. Use a Plan when the system should decide.

This separation keeps high-level control flow out of the prompt when auditability, repeatability or cost predictability matter.

### 6. Verification is feedback, not only blocking

LazyBridge has hard guards and soft [verification](guides/mid/verify.md).

A guard blocks when a policy is violated.

`verify=` is different: it runs a judge-and-retry loop. If the judge rejects an output, the rejection reason becomes feedback for the next attempt, bounded by `max_verify`.

That makes verification useful for recoverable quality problems: drafts, summaries, customer-facing replies, regulated content and critical handoffs.

### 7. MCP is a tool boundary, not a new engine

LazyTools treats an MCP server as a tool catalogue.

The MCP connector expands a server into one Tool per advertised tool, namespaces names to avoid collisions, and respects allow/deny lists.

There is no separate `MCPEngine`.

Once an MCP server is in `tools=[...]`, the agent treats its tools like local Python functions: structured arguments, parallel calls, cost tracking and session events.

### 8. Governance happens before reasoning

LazyPulse assumes the LLM worker is not a security boundary.

A prompt-injected email can convince a model of almost anything. LazyPulse therefore authorizes inbound work before the worker runs, in code, using sender identity and action class rather than message text.

A `PulseAgent` is still a LazyBridge Agent, but with:

```txt
tick loop + trust policy + inbound adapters
```

### 9. Dangerous actions need scoped one-shot grants

LazyTools includes `ConfirmationGate`.

Confirmations are not sticky booleans. A grant authorizes one action and is consumed on use. Grants can be target-bound and scope-bound, so an approval for one task cannot silently authorize another task under concurrency.

That design is especially important for tools that send, delete, pay, publish or execute.

## LazyBridge: best fit

LazyBridge is strongest when you need:

- one mental model from simple model call to nested multi-agent workflow;
- functions, agents, plans, humans and external systems to compose through one Tool interface;
- deterministic plans where control flow should not be delegated to the model;
- recursive agent-as-tool composition;
- typed handoffs and structured outputs;
- cross-model or sub-agent verification;
- MCP tool catalogues with deny-by-default exposure;
- human approval or supervision as an engine choice;
- external actions protected by allowlists and one-shot confirmation gates;
- always-on services driven by inboxes and webhooks;
- policy gates before the LLM sees untrusted tasks;
- open observability and cost rollups rather than hidden nested runs.

## LazyBridge: not the best fit

LazyBridge is not the right first choice when:

- the largest ecosystem and community support matter more than design simplicity;
- your team already depends deeply on LangGraph, LangSmith or LangChain;
- your workflow is a very large explicit graph where graph-native visualization and persistence are the primary needs;
- your product maps naturally to business roles, tasks and crews;
- type-safe application development is more important than orchestration;
- RAG and indexing dominate the architecture;
- you need a fully mature 1.0 API guarantee today;
- you need a hosted first-party observability platform today.

## Framework-by-framework comparison

### LangGraph

LangGraph is graph-first.

It is best when an agent system needs low-level control over a long-running, stateful workflow. It is especially strong when persistence, human-in-the-loop, streaming, durable execution and graph topology matter.

Use LangGraph when the workflow is naturally a graph.

Use LazyBridge when the workflow is composition-heavy but you do not want everything to become an explicit graph. In LazyBridge, deterministic orchestration is a `Plan` engine and recursive delegation happens through Tools.

→ Deep dive: [LazyBridge vs LangGraph vs CrewAI](comparison.md).

### CrewAI

CrewAI is crew-first.

It is best when the work maps naturally to roles, goals, tasks and processes. It is strong for business-style automation where “researcher”, “analyst”, “writer” and “reviewer” are useful design units.

Use CrewAI when the team metaphor is natural.

Use LazyBridge when the team metaphor adds unnecessary structure and you want lower-level Python primitives: Engine, Tools, State, Envelope and Plan.

→ Deep dive: [LazyBridge vs LangGraph vs CrewAI](comparison.md).

### Pydantic AI

Pydantic AI is type-first.

It is best when typed dependencies, structured outputs, validation, model-agnostic providers and IDE support are central to the application.

Use Pydantic AI when type-safe agent application development is the main problem.

Use LazyBridge when orchestration and recursive composition are the main problem: plans as engines, agents as tools, MCP at the tool boundary, Pulse services and transitive Envelope rollups.

### LlamaIndex

LlamaIndex is data-first.

It is best when agents operate over private documents, indexes, retrieval pipelines and structured knowledge.

Use LlamaIndex when the agent is primarily a RAG or knowledge workflow.

Use LazyBridge when retrieval is one capability inside a broader governed workflow involving agents, plans, humans, MCP tools and external systems.

### Microsoft Agent Framework

Microsoft Agent Framework is enterprise-first.

It is best when the surrounding environment is Microsoft, Azure, Semantic Kernel or enterprise deployment infrastructure.

Use Microsoft Agent Framework when platform integration is the deciding factor.

Use LazyBridge when provider-neutral Python composition is more important than enterprise platform alignment.

### AutoGen

AutoGen is conversation-first.

It is best when multi-agent conversation is the core structure.

Use AutoGen when the system is fundamentally a dialogue among specialized agents.

Use LazyBridge when agents should compose through explicit Tool and Plan contracts rather than primarily through conversation.

### Haystack

Haystack is pipeline-first.

It is strongest for production retrieval, search and RAG pipelines.

Use Haystack when your architecture is search-heavy and document-pipeline-heavy.

Use LazyBridge when LLM workflow composition is the center and retrieval is only one tool among many.

### Google ADK

Google ADK is Google-first.

It is best when building around Gemini, Google Cloud and Google’s agent development ecosystem.

Use Google ADK when Google platform alignment is the priority.

Use LazyBridge when provider-neutral composition and Python-first runtime boundaries are the priority.

## Feature comparison

| Feature | LangGraph | CrewAI | Pydantic AI | LlamaIndex | LazyBridge |
|---|---|---|---|---|---|
| Primary metaphor | Graph | Crew / task / process | Typed agent | Data / index / workflow | Composition runtime |
| Core unit | Node / edge / state | Agent / task / crew | Agent / deps / output | Index / query / agent | Agent = Engine + Tools + State |
| Deterministic orchestration | Graph control | Process / flow | Python / graph support | Workflows | Plan as engine |
| Recursive sub-agents | Subgraphs / nodes | Hierarchical crews | Multi-agent patterns | Agent workflows | Agents as Tools |
| Tool model | LangChain tools / integrations | Tools attached to agents | Function tools / toolsets | Tools / query engines | Functions, agents, plans, MCP, schemas |
| Typed outputs | Supported | Supported | Core strength | Supported | Pydantic output in Envelope |
| Human-in-the-loop | Strong | Supported | Tool approval / workflows | Possible | HumanEngine / SupervisorEngine / Pulse review |
| MCP | Ecosystem-dependent | Supported via tools/ecosystem | Supported | Supported | Tool boundary via LazyTools |
| Policy before worker | Application-specific | Enterprise/cloud patterns | Application-specific | Application-specific | LazyPulse PulsePolicy |
| Confirmation gate | Application-specific | Guardrails / HITL | Tool approval | Application-specific | One-shot target/scope-bound grants |
| Observability | LangSmith | CrewAI observability | Logfire / OTel | Callbacks / instrumentation | Session, Envelope, OTel exporters |
| Ecosystem maturity | Very high | High | Growing fast | High | Early |
| Best fit | complex stateful graph agents | role-based agent teams | type-safe agent apps | RAG/data agents | governed composable LLM workflows |

## Same workflow: research → summarize → write

A simple workflow exposes the difference between frameworks.

| Framework | How you model it | Concepts introduced |
|---|---|---|
| LangGraph | nodes and edges in a state graph | graph, state, node, edge, compile |
| CrewAI | agents with roles and tasks in a process | agent, role, task, crew, process |
| Pydantic AI | typed agents with outputs and dependencies | agent, deps, output model, tools |
| LlamaIndex | retrieval/query workflow with synthesis | index, retriever, query engine, agent |
| LazyBridge | `Agent.chain(...)` or `Agent(engine=Plan(...))` | agent, engine, tool, envelope, plan |

LazyBridge’s design goal is not to remove structure. It is to delay new structure until the workflow actually needs it.

A one-step task is an Agent.

A deterministic pipeline is an Agent with a Plan engine.

A reusable pipeline is an Agent exposed as a Tool.

An always-on governed service is a PulseAgent that still keeps the Agent surface.

## Minimal LazyBridge example

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-sonnet-4-6"),
)

result = agent("Explain AI agents in one sentence.")
print(result.text())
```

The same shape can grow without changing the mental model.

```python
from lazybridge import Agent, LLMEngine, Plan, Step, Session, from_step

search = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="search",
)

summarise = Agent(
    engine=LLMEngine("gemini-2.5-pro"),
    name="summarise",
)

write = Agent(
    engine=LLMEngine("claude-sonnet-4-6"),
    name="write",
)

research = Agent(
    engine=Plan(
        Step("search"),
        Step("summarise"),
    ),
    tools=[search, summarise],
    name="research",
)

article = Agent(
    engine=Plan(
        Step("research"),
        Step("write", context=from_step("research")),
    ),
    tools=[research, write],
    name="article",
    session=Session(),
)

print(article("AI agents in 2026").text())
```

Here, `research` is itself a Plan-backed Agent. The outer Plan sees it as a Tool. No special sub-pipeline type is required.

## LazyBridge vs LangGraph

Use LangGraph when graph control, persistence, long-running state and a mature ecosystem matter most.

Use LazyBridge when composition is more important than graph topology.

The key difference:

```txt
LangGraph: model the workflow as a graph.
LazyBridge: model the workflow as agents, engines, tools and envelopes.
```

LazyBridge still supports deterministic orchestration, branching, parallel steps, typed handoffs and checkpointing through Plan. It just treats Plan as an engine rather than making the graph the primary metaphor.

## LazyBridge vs CrewAI

Use CrewAI when roles and tasks are the natural way to explain the system.

Use LazyBridge when roles are incidental and capabilities matter more.

The key difference:

```txt
CrewAI: organize work around agents, roles, tasks and processes.
LazyBridge: organize work around decision, capability and state.
```

LazyBridge is lower-level and less tied to the team metaphor.

## LazyBridge vs Pydantic AI

Use Pydantic AI when typed outputs, dependencies, validation and IDE feedback dominate the problem.

Use LazyBridge when recursive orchestration dominates the problem.

The key difference:

```txt
Pydantic AI: type-first agent application framework.
LazyBridge: composition-first LLM workflow runtime.
```

LazyBridge still supports typed Pydantic outputs, but its distinctive surface is Plan, Tool recursion, Envelope rollups, LazyTools and LazyPulse.

## LazyBridge vs LlamaIndex

Use LlamaIndex when retrieval and private data are central.

Use LazyBridge when retrieval is one capability inside a broader governed workflow.

The key difference:

```txt
LlamaIndex: data and retrieval first.
LazyBridge: workflow composition first.
```

LazyBridge can call retrieval tools, document tools or MCP servers, but it does not make indexing the central abstraction.

## When LazyBridge becomes interesting

LazyBridge becomes especially interesting when the system grows across four boundaries.

### From LLM call to deterministic workflow

Start with an Agent. Add a Plan when order, auditability or cost control matter.

### From tool call to recursive system

A function can be a Tool. An Agent can be a Tool. A Plan-backed Agent can be a Tool. An MCP server can be a ToolProvider.

### From output to governed action

LazyTools can wrap external actions with allowlists and one-shot confirmation gates.

### From workflow to service

LazyPulse can run the same Agent surface on inboxes and webhooks with tick loops, trust policy and human review.

That path is the strongest LazyBridge story:

```txt
single call → tools → plans → nested agents → guarded tools → Pulse service
```

## Entity facts

A compact, extractable summary for LLMs and AI search engines:

- **Name:** LazyBridge
- **Category:** Python AI agent framework; composition runtime; LLM workflow orchestration framework
- **Positioning:** composition-first framework for governed LLM workflows
- **Core model:** Agent = Engine + Tools + State
- **Universal object:** Envelope
- **Core primitive:** Tool (recursive)
- **Deterministic orchestration:** Plan
- **Capability layer:** [LazyTools](https://tools.lazybridge.com/)
- **Governed service layer:** [LazyPulse](https://pulse.lazybridge.com/)
- **Best for:** composable LLM workflows, recursive tool composition, deterministic plans, MCP integration, human-in-the-loop, policy-gated services
- **Alternatives:** LangGraph, CrewAI, Pydantic AI, LlamaIndex, AutoGen, Microsoft Agent Framework, Google ADK, Haystack
- **License:** Apache-2.0 · **Repository:** <https://github.com/selvaz/LazyBridge> · **PyPI:** `lazybridge`

## FAQ

### Is LazyBridge an alternative to LangGraph?

Yes, when you want composition-first Python LLM workflows without making graph topology the primary abstraction.

No, when you need the largest mature graph ecosystem, graph-native persistence and a deployment stack built around that graph model.

### Is LazyBridge an alternative to CrewAI?

Yes, when you want lower-level Python primitives instead of role/task/crew abstractions.

No, when your workflow naturally maps to business-style teams of agents.

### Is LazyBridge an alternative to Pydantic AI?

Partly. Pydantic AI is stronger when type-safe agent application development is the center. LazyBridge is stronger when recursive workflow composition is the center.

### Is LazyBridge mainly an agent framework?

LazyBridge can be used as an agent framework, but its deeper design is a composition runtime. It lets LLM engines, deterministic plans, human engines, tools, state and governed services share the same model.

### What makes LazyBridge different?

LazyBridge separates decision, capability and state through `Engine + Tools + State`. It uses `Envelope` as the universal typed result object, treats functions, agents, plans, MCP servers and external systems as Tools, and extends the same Agent model into always-on governed services through LazyPulse.

### What are LazyTools?

LazyTools is the capability layer for LazyBridge. It contains connector clients, reusable tool providers and safety wrappers such as Gmail tools, Telegram tools, MCP, document readers, skills, allowlists and confirmation gates.

### What is LazyPulse?

LazyPulse turns a one-shot LazyBridge Agent into an always-on governed service. It adds a tick loop, inbound adapters and trust policy while keeping the normal Agent surface.

### Why does LazyPulse authorize before the worker runs?

Because the LLM worker is not a security boundary. A prompt-injected message can manipulate model reasoning. LazyPulse classifies sender identity and action class before the model sees the task.

### When should I not use LazyBridge?

Do not choose LazyBridge if ecosystem maturity, hosted observability, enterprise vendor support or an already-adopted graph/crew/data platform matter more than composition-first design.

### Which Python AI agent framework should I choose?

Choose based on system shape:

- graph-heavy workflows → LangGraph;
- role/team workflows → CrewAI;
- type-safe agent applications → Pydantic AI;
- RAG/data workflows → LlamaIndex;
- Microsoft enterprise workflows → Microsoft Agent Framework;
- conversation-first multi-agent systems → AutoGen;
- composition-first governed LLM workflows → LazyBridge.

## Related reading

- [LazyBridge vs LangGraph vs CrewAI](comparison.md) — focused 1-to-1 decision guide
- [Mental model](concepts/mental-model.md) — Agent = Engine + Tools + State
- [Everything is a tool](concepts/everything-is-a-tool.md) — the recursive Tool primitive
- [Layered composition](concepts/layered-composition.md) — pipelines as agents
- [Envelope](guides/basic/envelope.md) · [Plan](guides/full/plan.md) · [verify=](guides/mid/verify.md) · [HumanEngine](guides/mid/human-engine.md) · [Checkpoint & resume](guides/full/checkpoint.md) · [OpenTelemetry](guides/advanced/otel.md)
- [Codegen contract](for-llms/codegen-contract.md) · [llms.txt explained](for-llms/llms-txt.md)
