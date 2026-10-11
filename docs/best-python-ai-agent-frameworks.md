---
title: Best Python AI Agent Frameworks in 2026
description: Compare graph-first, crew-first, type-first, data-first, platform-oriented and composition-first Python AI agent frameworks for building LLM workflows.
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Best Python AI Agent Frameworks in 2026",
  "description": "A technical comparison of graph-first, crew-first, type-first, data-first, platform-oriented and composition-first Python AI agent frameworks.",
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

This guide compares Python AI agent frameworks by their design metaphor.
The metaphors below are a reading aid, not exclusive capability boundaries or
a ranking. Capabilities reflect the linked official documentation at the time
of writing (October 2026); check those docs for the version you will adopt.

!!! tip "Looking for a focused 1-to-1 decision?"
    This page is the **category hub** — a broad comparison across different framework designs.
    For a deeper, decision-tree-style head-to-head of just three, see
    [LazyBridge vs LangGraph vs CrewAI](comparison.md).

## Quick comparison

| Framework | Primary metaphor | Typical emphasis | Questions to evaluate |
|---|---|---|---|
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | Graph-first | Stateful orchestration, durable execution, streaming and human review | Do you want graph state and topology to be explicit? |
| [CrewAI](https://docs.crewai.com/en/concepts/crews) | Crew-first | Agents, tasks, processes; Flows for event-driven orchestration | Does your domain map to roles and tasks, or to a Flow? |
| [Pydantic AI](https://pydantic.dev/docs/ai/overview/) | Type-first | Typed dependencies, outputs, tools and validation | Are typed application interfaces the main concern? |
| [LlamaIndex](https://developers.llamaindex.ai/python/framework/) | Data-first | Ingestion, indexes, retrieval, agents and workflows | How much of your application is knowledge grounding? |
| [Haystack](https://docs.haystack.deepset.ai/docs/intro) | Pipeline-first | Components, document stores, retrieval pipelines and agents | Do reusable document-processing components fit your architecture? |
| [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/) | Platform-oriented | Python/.NET agents, workflows, tools and integrations | Do its language and platform integrations match your deployment? |
| [Google ADK](https://adk.dev/) | Agent/workflow toolkit | Agents, tools, sessions and workflow agents | Do its runtime and deployment interfaces fit your application? |
| [AutoGen](https://microsoft.github.io/autogen/stable/) | Conversation and event-driven agents | AgentChat, Core, extensions and Studio | Do you want conversational teams or event-driven components? |
| LazyBridge | Composition-first | Coding runtimes, provider APIs, Plans and humans behind one Agent contract | Do you want engines to share Python tools and state, with opposite-engine review? |

## How to choose

Choose **LangGraph** if your workflow is truly a graph: long-running state, branching, loops, human review, checkpointing and fine-grained runtime control.

Choose **CrewAI** if your workflow reads naturally as a team: researcher, analyst, writer, reviewer, manager.

Choose **Pydantic AI** if your main problem is reliable typed outputs, dependency injection, validation and IDE-friendly Python application structure.

Choose **LlamaIndex** if the agent mostly works over documents, indexes, retrieval pipelines or private knowledge.

Consider **Microsoft Agent Framework** when Python/.NET interoperability and its platform integrations fit your environment; it supports multiple model providers, not only Microsoft services.

Consider **Haystack** for component-based retrieval pipelines, **AutoGen** for
AgentChat or event-driven Core, and **Google ADK** for its agent/workflow toolkit.
The official docs in the table describe those choices beyond these metaphors.

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
| Conversation/event-driven | AutoGen | AgentChat dialogue and Core event-driven components |
| Agent/workflow toolkit | Google ADK | agents, tools, sessions and workflow agents |
| Platform-oriented | Microsoft Agent Framework | platform workflows, deployment, governance |
| Composition-first | LazyBridge | recursive composition across agents, tools, plans, humans and services |

LazyBridge belongs in the last category: **composition-first**.

## A different category: composition-first runtimes

Most frameworks start with agents, graphs, crews, chains or workflows.

LazyBridge starts with this decomposition:

```txt
Agent = Engine + Tools + State
```

The **Engine** decides what happens next. The **Tools** define what the agent can do. The **State** carries continuity, memory, persistence, events and observability.

That split matters because LazyBridge does not treat an agent as “an LLM wrapper”.

An Agent can be driven by:

- a `ClaudeCodeEngine` or `CodexEngine`, using the local authenticated runtime;
- an `LLMEngine`, using a provider API;
- a deterministic `Plan`;
- a `HumanEngine`;
- a `SupervisorEngine`;
- a custom `Engine` (the [engine protocol](guides/advanced/engine-protocol.md)).

This means the same `Agent` surface can represent a one-shot model call, a deterministic pipeline, a human approval step, a supervised workflow or a larger nested system.

See the [mental model](concepts/mental-model.md) for the full picture.

## LazyBridge in one sentence

LazyBridge composes Claude Code, Codex, provider APIs, deterministic Plans and human decisions through one Agent contract. LazyTools supplies integrations; LazyPulse adds governed service execution.

## The LazyBridge ecosystem

LazyBridge is split into three focused packages.

| Package | Role | What it adds |
|---|---|---|
| LazyBridge Core | Composition runtime | `Agent`, `Engine`, `Tool`, `Envelope`, `Plan`, `Step`, `Store`, `Session`, `HumanEngine`, `SupervisorEngine`, guards, verification, tracing |
| [LazyTools](https://tools.lazybridge.com/) | Integration layer | Connectors, MCP, code support, repository code bridge, projects, quota routing and guarded actions |
| [LazyPulse](https://pulse.lazybridge.com/) | Governed service layer | tick loop, inbound adapters, trust policy, task lifecycle, human review and audit trail |

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
from lazybridge import Agent, ClaudeCodeEngine, Plan, Step
from lazybridge.testing import MockAgent

dynamic_agent = Agent(
    engine=ClaudeCodeEngine(),
)

deterministic_pipeline = Agent(
    engine=Plan(Step("research"), Step("write")),
    tools=[MockAgent("research", name="research"), MockAgent("draft", name="write")],
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
- an MCP catalogue exposed by a LazyTools ToolProvider;
- a pre-built JSON schema;
- an external tool catalogue.

The consuming agent sees the same Python Tool contract in those cases.
Provider-native capabilities are a separate boundary: use
`LLMEngine(native_tools=[...])`. Claude built-ins and Codex sandbox tools use
their engine's native policy, rather than `Tool.wrap()`.

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

The important part is transitive observability: when an agent calls another agent as a tool, that nested run does not disappear. Nested token and cost metadata roll up into the parent Envelope. Parent latency measures the enclosing run; nested errors remain visible in tool results and Session events, and do not necessarily make the parent fail.

### 5. Plan separates orchestration from reasoning

A [`Plan`](guides/full/plan.md) is a deterministic engine.

It declares steps, data flow, routes, typed handoffs, parallel bands, store writes and checkpoint/resume behavior.

Use an LLM when the model should decide. Use a Plan when the system should decide.

This separation keeps high-level control flow out of the prompt when auditability, repeatability or cost predictability matter.

### 6. Verification is feedback, not only blocking

LazyBridge has hard guards and soft [verification](guides/mid/verify.md).

A guard blocks when a policy is violated.

`verify=` is different: it runs a judge-and-retry loop. If the judge rejects an output, the rejection reason becomes feedback for the next attempt, bounded by `max_verify` total attempts (not additional retries). The final draft is returned on exhaustion; `Envelope.ok` is execution success, not approval.

That makes verification useful for recoverable quality problems: drafts, summaries, customer-facing replies, regulated content and critical handoffs.

### 7. MCP is a tool boundary, not a new engine

LazyTools treats an MCP server as a tool catalogue.

The MCP connector expands a server into one Tool per advertised tool, namespaces names to avoid collisions, and respects allow/deny lists.

There is no separate `MCPEngine`.

A LazyTools MCP provider in `tools=[...]` normalizes advertised tools into
the application Tool contract. Exposure requires an explicit allow/deny policy.
See [LazyTools MCP](https://tools.lazybridge.com/mcp/) for connector setup;
Core does not duplicate the connector API or own its server lifecycle.

### 8. Governance happens before reasoning

LazyPulse assumes the LLM worker is not a security boundary.

Untrusted messages can manipulate model reasoning. LazyPulse applies inbound policy using sender identity and action class before the worker runs; this reduces reliance on prompt-only authorization, without claiming to solve every prompt-injection risk. See the [LazyPulse documentation](https://pulse.lazybridge.com/).

A `PulseAgent` is still a LazyBridge Agent, but with:

```txt
tick loop + trust policy + inbound adapters
```

### 9. Dangerous actions need scoped one-shot grants

LazyTools includes `ConfirmationGate`; its [documentation](https://tools.lazybridge.com/safety/#confirmationgate) defines grant, target and scope behavior.

Confirmations are not sticky booleans. A grant authorizes one action and is consumed on use. Grants can be target-bound and scope-bound, so an approval for one task cannot silently authorize another task under concurrency.

That design is especially important for tools that send, delete, pay, publish or execute.

## LazyBridge: best fit

LazyBridge is strongest when you need:

- one mental model from simple model call to nested multi-agent workflow;
- functions, agents, plans, humans and external systems to compose through one Tool interface;
- deterministic plans where control flow should not be delegated to the model;
- recursive agent-as-tool composition;
- typed handoffs and structured outputs;
- Claude Code and Codex as engines, sharing Python tools and application state;
- opposite-engine `verify=` review and bounded feedback;
- durable native sessions/threads and `session_alias` bindings;
- cross-model or sub-agent verification;
- MCP tool catalogues with deny-by-default exposure;
- human approval or supervision as an engine choice;
- external actions protected by allowlists and one-shot confirmation gates;
- always-on services driven by inboxes and webhooks;
- policy gates before the LLM sees untrusted tasks;
- open observability and cost rollups rather than hidden nested runs.

## LazyBridge: not the best fit

LazyBridge is not the right first choice when:

- your required integrations and support arrangements are already met by another stack;
- your team already depends deeply on LangGraph, LangSmith or LangChain;
- your workflow is a very large explicit graph where graph-native visualization and persistence are the primary needs;
- your product maps naturally to business roles, tasks and crews;
- type-safe application development is more important than orchestration;
- RAG and indexing dominate the architecture;
- you need compatibility guarantees beyond the current release and extension contracts;
- you need a hosted first-party observability platform today.

## Framework-by-framework comparison

### LangGraph

LangGraph exposes graph state, nodes and edges, plus a functional API. Its
[overview](https://docs.langchain.com/oss/python/langgraph/overview) documents
durable execution, streaming and human-in-the-loop support; its
[persistence guide](https://docs.langchain.com/oss/python/langgraph/persistence)
describes checkpointers and threads.

Consider it when graph topology and checkpointed state are central. Consider
LazyBridge when you prefer recursive Tool composition and a Plan engine for
deterministic orchestration. Neither preference implies the other framework
cannot compose agents.

Deep dive: [LazyBridge vs LangGraph vs CrewAI](comparison.md).

### CrewAI

CrewAI describes [agents](https://docs.crewai.com/en/concepts/agents),
[tasks](https://docs.crewai.com/en/concepts/tasks) and
[processes](https://docs.crewai.com/en/concepts/processes). Its
[Flows](https://docs.crewai.com/en/concepts/flows) also provide state,
conditional routing, parallel listeners and persistence. CrewAI is broader
than a sequential crew metaphor.

Consider it when roles and tasks are useful domain units, or when its Flow API
fits your events. Consider LazyBridge when Engine, Tool, Envelope and Plan are
the interfaces you want to expose throughout your Python application.

Deep dive: [LazyBridge vs LangGraph vs CrewAI](comparison.md).

### Pydantic AI

Pydantic AI's [official overview](https://pydantic.dev/docs/ai/overview/)
documents typed agents, dependencies, structured outputs, function tools,
model-provider integrations and instrumentation. It also covers multi-agent
workflows; type-first does not mean orchestration is absent.

Consider it when typed application dependencies and outputs drive the design.
Consider LazyBridge when recursive Plans and coding-engine interchangeability
are the interfaces you want. Both can validate Pydantic outputs.

### LlamaIndex

The [LlamaIndex framework docs](https://developers.llamaindex.ai/python/framework/)
cover ingestion, indexes, retrieval, agents, structured outputs, workflows and
MCP integration. Data-first describes its emphasis, not a restriction to RAG.

Consider it when private documents and knowledge grounding dominate. Consider
LazyBridge when retrieval is a Tool inside a wider workflow of Plans, coding
engines, human decisions and guarded external actions.

### Microsoft Agent Framework

Its [official overview](https://learn.microsoft.com/en-us/agent-framework/overview/)
documents Python and .NET agents, multiple model providers, tools/MCP,
sessions and workflows. Microsoft describes it as a successor to AutoGen and
Semantic Kernel. It is not confined to Azure models.

Consider it when those language and platform interfaces fit your environment.
Consider LazyBridge when its Python Agent/Engine/Tool contract matches the
application you want to maintain. Evaluate deployment and support requirements
directly rather than inferring them from the vendor name.

### AutoGen

The [AutoGen docs](https://microsoft.github.io/autogen/stable/) distinguish
AgentChat's conversational agents, Core's event-driven components, extensions
and Studio. Core supports workflows beyond dialogue, including distributed
scenarios. For new adoption, also read Microsoft's Agent Framework guidance.

Consider AutoGen when AgentChat or Core matches your interaction model.
Consider LazyBridge when you want delegation to pass through explicit Tool
and Plan contracts. This is an interface preference, not a claim that AutoGen
lacks deterministic orchestration.

### Haystack

[Haystack's introduction](https://docs.haystack.deepset.ai/docs/intro)
documents components, pipelines, document stores, agents and tools. Pipelines
can be agent tools, and agents can participate in pipelines.

Consider it for reusable retrieval and document-processing components.
Consider LazyBridge when those capabilities are part of a broader composition
runtime. Pipeline-first does not imply that nested agents are unavailable.

### Google ADK

The [ADK docs](https://adk.dev/) cover agents, tools, sessions, state,
model integrations and sequential/parallel/loop workflow agents. Its Google
integrations are relevant to some deployments, but the toolkit is not limited
to one provider.

Consider it when its agent/workflow runtime fits your deployment. Consider
LazyBridge when coding engines, provider APIs and deterministic Plans should
share Python tools and state through one Agent contract.

## Feature comparison

These cells name documented interfaces, not quality scores. Follow the linked
documentation for configuration and version-specific limits.

| Feature | LangGraph | CrewAI | Pydantic AI | LlamaIndex | LazyBridge |
|---|---|---|---|---|---|
| Primary metaphor | Graph/state | Crew/task plus Flows | Typed agent | Data/agent/workflow | Composition runtime |
| Core unit | State/node/edge | Agent/task/crew/flow | Agent/dependencies/output | Index/query/agent/workflow | Agent = Engine + Tools + State |
| Deterministic orchestration | Graph or functional API | Processes and Flows | Python and graph patterns | Workflows | Plan engine |
| Recursive sub-agents | Subgraphs/nodes | Delegation and hierarchical processes | Multi-agent patterns | Agent workflows | Agents as Tools |
| Tool model | Callables and integrations | Agent tools | Function tools/toolsets | Function and query tools | Functions, agents, Plans, LazyTools catalogues |
| Typed outputs | Model/state integration | output_pydantic/output_json | Typed output validation | Structured output | Pydantic payload in Envelope |
| Human-in-the-loop | Interrupts and checkpoints | Task human input; Flow/application hooks | Approval/workflow interfaces | Human-in-the-loop agents | HumanEngine/SupervisorEngine; approval gates |
| MCP | [LangChain tool integration](https://docs.langchain.com/oss/python/langchain/mcp) | [MCP tools](https://docs.crewai.com/en/mcp/overview) | MCP toolsets | MCP tools/workflows | LazyTools integration layer |
| Policy before worker | Application policy | Application policy | Application policy | Application policy | LazyPulse inbound policy |
| Confirmation gate | Application/tool policy | Guardrails/human input | Tool approval | Application/tool policy | Core approvals; LazyTools scoped grants |
| Observability | [LangSmith](https://docs.langchain.com/langsmith/home) or application instrumentation | [CrewAI tracing](https://docs.crewai.com/en/observability/tracing)/integrations | Logfire/instrumentation | Callbacks/instrumentation | Envelope, opt-in Session/exporters/OTel |
| Coding runtimes | Integrate in a node/tool | Integrate in a task/tool | Integrate in a tool | Integrate in a tool | ClaudeCodeEngine and CodexEngine |
| Best-fit question | Is graph state central? | Are roles/tasks or Flows central? | Are types/dependencies central? | Is knowledge grounding central? | Should engines share Tool and state contracts? |

Sources: [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts),
[CrewAI tasks](https://docs.crewai.com/en/concepts/tasks) and
[Flows](https://docs.crewai.com/en/concepts/flows),
[Pydantic AI](https://pydantic.dev/docs/ai/overview/) and
[LlamaIndex](https://developers.llamaindex.ai/python/framework/).
Observability and MCP depend on the selected integrations, not just the framework name.

## Same workflow: research → summarize → write

A simple workflow exposes the difference between frameworks.

| Framework | How you model it | Concepts introduced |
|---|---|---|
| LangGraph | nodes and edges in a state graph | graph, state, node, edge, compile |
| CrewAI | agents with roles and tasks in a process | agent, role, task, crew, process |
| Pydantic AI | typed agents with outputs and dependencies | agent, deps, output model, tools |
| LlamaIndex | retrieval/query workflow with synthesis | index, retriever, query engine, agent |
| LazyBridge | `Agent.chain(...)` or `Agent(engine=Plan(...))` | agent, engine, tool, envelope, plan |

The interfaces above are described in the official
[LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[CrewAI processes](https://docs.crewai.com/en/concepts/processes),
[Pydantic AI overview](https://pydantic.dev/docs/ai/overview/) and
[LlamaIndex framework docs](https://developers.llamaindex.ai/python/framework/).

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

Consider LangGraph when graph control and checkpointed state matter most; see its [persistence guide](https://docs.langchain.com/oss/python/langgraph/persistence).

Use LazyBridge when composition is more important than graph topology.

The key difference:

```txt
LangGraph: model the workflow as a graph.
LazyBridge: model the workflow as agents, engines, tools and envelopes.
```

LazyBridge still supports deterministic orchestration, branching, parallel steps, typed handoffs and checkpointing through Plan. It just treats Plan as an engine rather than making the graph the primary metaphor.

## LazyBridge vs CrewAI

Consider CrewAI when [roles and tasks](https://docs.crewai.com/en/concepts/tasks) are the natural way to explain the system; its [Flows](https://docs.crewai.com/en/concepts/flows) also support other shapes.

Use LazyBridge when roles are incidental and capabilities matter more.

The key difference:

```txt
CrewAI: organize work around agents, roles, tasks and processes.
LazyBridge: organize work around decision, capability and state.
```

LazyBridge is lower-level and less tied to the team metaphor.

## LazyBridge vs Pydantic AI

Consider Pydantic AI when [typed outputs and dependencies](https://pydantic.dev/docs/ai/overview/) dominate the problem.

Use LazyBridge when recursive orchestration dominates the problem.

The key difference:

```txt
Pydantic AI: type-first agent application framework.
LazyBridge: composition-first LLM workflow runtime.
```

LazyBridge supports typed Pydantic outputs. Its interfaces also include coding engines, Plan, Tool recursion, Envelope usage rollups, LazyTools and LazyPulse.

## LazyBridge vs LlamaIndex

Consider LlamaIndex when [retrieval and private data](https://developers.llamaindex.ai/python/framework/) are central.

Use LazyBridge when retrieval is one capability inside a broader governed workflow.

The key difference:

```txt
LlamaIndex: data and retrieval first.
LazyBridge: workflow composition first.
```

LazyBridge can call retrieval tools, document tools or MCP servers, but it does not make indexing the central abstraction.

## Claude Code writes, Codex reviews

Coding runtimes are engines, not just external commands. This producer asks an
opposite-engine reviewer for bounded feedback:

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

reviewer = Agent(name="reviewer", engine=CodexEngine(
    system='Start with "approved" or "rejected: <reason>". Check edge cases.',
))
writer = Agent(name="writer", engine=ClaudeCodeEngine(), verify=reviewer, max_verify=3)
print(writer("Draft a function that rejects weekend deployments.").text())
```

`max_verify` counts total writer attempts. Exhaustion returns the last draft;
require tests or an explicit acceptance record before release. For durable
native history use `persist_session`/`persist_thread` or
[SessionRegistry aliases](guides/full/session-registry.md). Engine swaps can
reuse Python tools, Memory, Store and Session; they do not migrate a native
transcript between providers.

The [complete collaboration guide](guides/full/claude-writes-codex-reviews.md)
covers in-process review. For detached repository work, use the
[LazyTools code bridge](https://tools.lazybridge.com/code-bridge/), with durable
tickets and opposite-engine review routing. LazyTools owns MCP, code-support
and project integrations. The quota routing benchmark is **in progress**.

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

Evaluate the graph stack directly when checkpointed graph state and its deployment interfaces are your requirements.

### Is LazyBridge an alternative to CrewAI?

Yes, when you prefer Engine/Tool/Plan interfaces over CrewAI’s [agent/task and Flow interfaces](https://docs.crewai.com/en/concepts/flows).

Evaluate CrewAI directly when its team or Flow interfaces match your application.

### Is LazyBridge an alternative to Pydantic AI?

Partly. Compare Pydantic AI’s [typed application interfaces](https://pydantic.dev/docs/ai/overview/) with LazyBridge’s recursive Tool and Plan interfaces; the useful choice depends on your application.

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

Use system shape to shortlist candidates, then verify the capabilities in the official docs linked above:

- graph-heavy workflows → LangGraph;
- role/team workflows → CrewAI;
- type-safe agent applications → Pydantic AI;
- RAG/data workflows → LlamaIndex;
- Python/.NET platform workflows → Microsoft Agent Framework;
- conversational or event-driven multi-agent systems → AutoGen;
- component-based search pipelines → Haystack;
- ADK agent/workflow interfaces → Google ADK;
- composition-first governed LLM workflows → LazyBridge.

## Related reading

- [LazyBridge vs LangGraph vs CrewAI](comparison.md) — focused 1-to-1 decision guide
- [Mental model](concepts/mental-model.md) — Agent = Engine + Tools + State
- [Everything is a tool](concepts/everything-is-a-tool.md) — the recursive Tool primitive
- [Layered composition](concepts/layered-composition.md) — pipelines as agents
- [Envelope](guides/basic/envelope.md) · [Plan](guides/full/plan.md) · [verify=](guides/mid/verify.md) · [HumanEngine](guides/mid/human-engine.md) · [Checkpoint & resume](guides/full/checkpoint.md) · [OpenTelemetry](guides/advanced/otel.md)
- [Codegen contract](for-llms/codegen-contract.md) · [llms.txt explained](for-llms/llms-txt.md)
