# LazyBridge — Full tier
Production pipelines. Declared workflows with typed hand-offs between
steps, conditional routing via sentinels, checkpoint/resume after
crashes, OTel/JSON export, tool-level verifiers, serialisable plans.

## Plan

**signature**

Plan(
    *steps: Step,
    max_iterations: int = 100,
    store: Store | None = None,
    checkpoint_key: str | None = None,
    resume: bool = False,
) -> Engine

Step(
    target: str | Callable | Agent,      # tool name, function, Agent
    task: Sentinel | str = from_prev,
    context: Sentinel | str | None = None,
    sources: list = (),
    writes: str | None = None,            # Store key under which payload is saved
    input: type = Any,
    output: type = str,                   # Pydantic triggers structured output + routing
    parallel: bool = False,
    name: str | None = None,
)

PlanCompileError  # raised at Agent construction if Plan is invalid
PlanState         # checkpoint shape: plan_id, current_step, next_step, store, history, status
StepResult        # single step record: step_name, envelope, ts

Usage: Agent(engine=Plan(Step(a), Step(b)))

**rules**

- ``max_iterations`` caps the total number of step executions in one
  ``run`` to guard against runaway routing loops (default 100). Raise
  it for legitimate long plans; lower it to fail fast during dev.
- Step names are unique. ``PlanCompileError`` fires at Agent construction
  if duplicates or dangling references exist.
- Sentinels: ``from_prev`` (previous step's output, default),
  ``from_start`` (original user task), ``from_step("name")``,
  ``from_parallel("name")``. See the sentinels page.
- ``output=SomeModel`` activates structured output at that step. If the
  model has a ``next: Literal["a", "b", ...]`` field, the plan routes to
  the matching step on completion.
- ``parallel=True`` marks a step as a concurrent branch; combine with
  ``from_parallel`` on the join step.
- ``writes="key"`` stores the step's payload into ``store[key]`` if a
  Store is passed. Required for checkpoint data to survive across runs.
- ``checkpoint_key`` + ``store`` enable state persistence after every
  step; ``resume=True`` reads the checkpoint and picks up at the next
  unrun step (failed runs restart from the failing step, not the next).

**example**

```python
from lazybridge import Agent, Plan, Step, Store, from_prev, from_step
from pydantic import BaseModel
from typing import Literal

class Hits(BaseModel):
    items: list[str]
    next: Literal["rank", "empty"] = "rank"

class Ranked(BaseModel):
    top: list[str]

store = Store(db="research.sqlite")

plan = Plan(
    Step(searcher, name="search",  writes="hits",   output=Hits),
    Step(ranker,   name="rank",    task=from_prev,  output=Ranked),
    Step(writer,   name="write",   task=from_step("rank")),
    Step(apology,  name="empty"),  # reached only if Hits.next == "empty"
    store=store, checkpoint_key="research", resume=True,
    max_iterations=20,
)

Agent.from_engine(plan)("AI trends April 2026")
```

**pitfalls**

- Forgetting ``output=Model`` on a step and then expecting the next step
  to read ``.field`` — the next step will see a plain string. Declare
  types everywhere you need them.
- Cyclic ``depends`` or references to unknown step names → ``PlanCompileError``.
- ``resume=True`` without ``store=`` is a silent no-op (no checkpoint
  to read or write to). Pass both.
- A step that fails persists a ``status="failed"`` checkpoint pointing
  back at itself. Subsequent ``resume=True`` runs retry that step.

**see-also**

[sentinels](sentinels.md), [parallel_steps](parallel-steps.md),
[checkpoint](checkpoint.md), [verify](verify.md),
[plan_serialize](plan-serialize.md),
decision tree: [composition](../decisions/composition.md)

## Sentinels (from_prev / from_start / from_step / from_parallel)

**signature**

from_prev                # singleton — previous step's output (default)
from_start               # singleton — original user task
from_step(name: str)     # named prior step's output
from_parallel(name: str) # named parallel branch's output

# Used on Step(..., task=<sentinel>) or Step(..., context=<sentinel>).

**rules**

- ``from_prev`` (default): the previous step's output becomes the next
  step's task. This is real chain semantics — each step sees what its
  predecessor produced, not the original user input.
- ``from_start``: explicit reference to the initial envelope. Use it
  when you want a step to operate on the original user request
  regardless of what preceded it.
- ``from_step("n")``: reach back to a specific prior step's result.
  PlanCompiler verifies ``"n"`` names an earlier step, else raises.
- ``from_parallel("n")``: alias for ``from_step`` intended for parallel
  branch joins. Indicates to readers that the step being referred to
  ran concurrently with siblings.
- A plain string passed as ``task=`` is used verbatim — useful for
  hard-coded prompts at intermediate steps.

**example**

```python
from lazybridge import Plan, Step, from_prev, from_start, from_step

plan = Plan(
    Step(researcher,    name="research",  output=Hits),
    Step(fact_checker,  name="check",     task=from_prev),    # check researcher's output
    Step(writer,        name="write",     task=from_start),   # writer sees ORIGINAL user task
    Step(editor,        name="edit",      task=from_step("write"),
                                          context=from_step("check")),
)
```

**pitfalls**

- ``from_prev`` after a parallel branch returns the join step's output,
  not one of the branches. Use ``from_parallel("<branch-name>")`` for a
  specific branch.
- Sentinels are module-level imports; don't shadow them with local
  variables of the same name.
- When passing a ``str`` as ``task=``, it's treated as a LITERAL, not a
  sentinel. Don't write ``task="from_prev"`` expecting the sentinel.

**see-also**

[plan](plan.md), [parallel_steps](parallel-steps.md),
decision tree: [composition](../decisions/composition.md)

## Parallel plan steps

**signature**

Step(target, *, parallel: bool = False, name: str | None = None, ...)
from_parallel(name: str) -> Sentinel

# Typical shape: N parallel branches followed by a join step.
Plan(
    Step(a, name="a", parallel=True),
    Step(b, name="b", parallel=True),
    Step(c, name="c", parallel=True),
    Step(join, name="join",
         task=from_parallel("a"),
         context=from_parallel("b")),
)

**rules**

- ``parallel=True`` marks a step as a branch that runs concurrently
  with other consecutive parallel steps in the plan.
- The plan engine dispatches all consecutive ``parallel=True`` steps
  via ``asyncio.gather`` before proceeding.
- A non-parallel step immediately after parallel steps acts as an
  implicit join: it sees ``from_prev`` as the last completed branch's
  output; use ``from_parallel("name")`` to reach a specific branch.
- Parallel steps may have their own ``writes=`` — each branch's
  payload is persisted under the respective Store key.
- Errors in a parallel branch surface as an error ``Envelope`` for
  that branch only; sibling branches continue.

**example**

```python
from lazybridge import Agent, Plan, Step, from_parallel, Store

store = Store(db="monitor.sqlite")

plan = Plan(
    # Three independent searchers fan out in parallel.
    Step(anthropic_search, name="search_a", parallel=True, writes="findings_a"),
    Step(openai_search,    name="search_o", parallel=True, writes="findings_o"),
    Step(google_search,    name="search_g", parallel=True, writes="findings_g"),

    # Join: synthesiser reads all three branches via context=.
    Step(synthesiser, name="synth",
         task=from_parallel("search_a"),
         context=from_parallel("search_o"),  # could concatenate more
         writes="plan"),

    store=store,
)
Agent.from_engine(plan)("framework update — April 2026")
```

**pitfalls**

- Interleaving parallel and sequential steps without care: the engine
  only bundles CONSECUTIVE ``parallel=True`` steps. Insert them in a
  run.
- Forgetting the join step — after N parallel steps the next
  non-parallel step IS the join. If you want all three outputs you
  must read them via ``from_parallel("…")`` on the join step;
  otherwise only ``from_prev`` (last completed) is visible.
- Checkpointing across a parallel block is coarse-grained: the engine
  saves after the block completes, not per-branch. If branch A
  succeeds but B crashes, resume retries the whole block, not just B.
  (Tracked for future work.)

**see-also**

[plan](plan.md), [sentinels](sentinels.md),
[agent_parallel](agent-parallel.md),
decision tree: [parallelism](../decisions/parallelism.md)

## SupervisorEngine

**signature**

SupervisorEngine(
    *,
    tools: list[Tool | Callable | Agent] = None,
    agents: list[Agent] = None,         # agents the human can retry
    store: Store | None = None,
    input_fn: Callable[[str], str] | None = None,
    ainput_fn: Callable[[str], Awaitable[str]] | None = None,
    timeout: float | None = None,
    default: str | None = None,
) -> Engine

Usage: Agent(engine=SupervisorEngine(tools=[...], agents=[researcher]))

REPL commands:
  continue [optional text]        accept; return to caller
  retry <agent>: <feedback>       re-run a registered agent with feedback
  store <key>                     print store[key]
  <tool>(<args>)                  invoke a registered tool

**rules**

- ``tools=`` accepts functions, Tool instances, and Agent instances
  uniformly (wrap_tool is applied at __init__). Same contract as
  ``Agent(tools=...)``.
- The REPL runs on a worker thread so the caller's event loop is not
  blocked. ``input_fn`` is called there; use scripted inputs in tests.
- ``retry <agent>: <feedback>`` re-runs the named agent with the
  feedback appended to the task. The output replaces the current
  supervisor buffer.
- Unknown commands print help and re-prompt. ``continue`` is the only
  terminator.
- Session propagation: an Agent wrapping a SupervisorEngine receives
  session events for AGENT_START / AGENT_FINISH like any other engine.

**example**

```python
from lazybridge import Agent, SupervisorEngine, Tool, Store

def search(query: str) -> str:
    """Search the web for query."""
    return f"hits for {query}"

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])
store = Store()
store.write("policy", "publish only peer-reviewed sources")

supervisor = Agent(
    engine=SupervisorEngine(
        tools=[search],
        agents=[researcher],
        store=store,
    ),
    name="supervisor",
)

writer = Agent("claude-opus-4-7", name="writer")

# Pipeline: researcher drafts → supervisor inspects / revises → writer finalises.
pipeline = Agent.chain(researcher, supervisor, writer)
pipeline("AI policy brief")
```

**pitfalls**

- ``input_fn`` is called from a worker thread. If it accesses
  thread-unsafe state (like ``readline`` history), guard it.
- ``agents=`` expects v1 ``Agent`` instances. Duck-typed objects work
  if they expose ``__call__`` / ``run`` and a ``name`` attribute.
- The REPL blocks the human user — if ``timeout=None`` (the default),
  an unattended pipeline hangs forever. Set ``timeout=``+``default=``
  for unattended runs.
- Tool calls in the REPL go via ``run_sync``. If a tool's ``func`` is
  async, it's driven to completion automatically (post-v1 fix).

**see-also**

[human_engine](human-engine.md), [agent](agent.md),
[plan](plan.md),
decision tree: [human_engine_vs_supervisor](../decisions/human-engine-vs-supervisor.md)

## Checkpoint & resume

**signature**

Plan(
    *steps,
    store: Store,
    checkpoint_key: str,
    resume: bool = False,
) -> Engine

# Persisted shape at store[checkpoint_key]:
#   {
#     "next_step": str | None,
#     "kv": {"writes_key": payload, ...},
#     "completed_steps": [str],
#     "status": "running" | "failed" | "done",
#   }

**rules**

- Checkpoint scatta dopo ogni step riuscito e dopo ogni step fallito.
- Success path: ``status="running"`` (step successivo pendente) →
  ``status="done"`` quando ``next_step is None``.
- Fail path: lo step fallito NON viene aggiunto a ``completed_steps``;
  il checkpoint salva ``next_step=<nome step fallito>`` + ``status="failed"``.
  Un successivo run con ``resume=True`` ri-parte da quello step.
- Success + ``resume=True`` + ``status="done"`` → short-circuit: il Plan
  ritorna un Envelope con payload = ``kv`` cached, senza rieseguire.
- Il checkpoint è JSON-encoded via ``Store.write``; gli step ``writes=``
  devono essere JSON-serialisable (string, dict, Pydantic model via
  ``.model_dump()``).

**example**

```python
from lazybridge import Agent, Plan, Step, Store

store = Store(db="pipeline.sqlite")

def build_plan():
    return Plan(
        Step(researcher, name="search",  writes="hits"),
        Step(ranker,     name="rank",    writes="ranked"),
        Step(writer,     name="write",   writes="draft"),
        store=store,
        checkpoint_key="pipeline",
        resume=True,
    )

# Run 1 — crash dopo rank: status="failed", next_step="write".
try:
    Agent.from_engine(build_plan())("AI trends")
except KeyboardInterrupt:
    pass

# Run 2 — resume dallo step fallito, non rifà search+rank.
Agent.from_engine(build_plan())("AI trends")

# Run 3 — il plan è già "done": short-circuit, ritorna kv cached.
result = Agent.from_engine(build_plan())("AI trends")
print(result.payload)  # {"hits": ..., "ranked": ..., "draft": ...}
```

**pitfalls**

- Cambiare la definizione del ``Plan`` (aggiungere / rimuovere step,
  rinominare) e riprendere da un checkpoint vecchio è un errore: il
  ``next_step`` salvato può non esistere più. Invalida il checkpoint
  (``store.delete(checkpoint_key)``) dopo refactor dei step.
- Non-JSON-serialisable ``writes`` (es. un file handle) si rompono in
  silenzio (vengono convertiti a stringa via ``default=str``).
- Il resume non ri-inietta session / exporter del run originario; passa
  gli stessi ``session=`` + ``store=`` su ogni run per continuità.

**see-also**

[plan](plan.md), [store](store.md),
decision tree: [checkpoint](../decisions/checkpoint.md)

## Exporters

**signature**

# Protocol
class EventExporter(Protocol):
    def export(self, event: dict) -> None: ...

# Built-ins
CallbackExporter(fn: Callable[[dict], None])
ConsoleExporter(*, stream=sys.stdout)                 # pretty stdout
FilteredExporter(inner: EventExporter, *, event_types: set[str])
JsonFileExporter(path: str)                           # JSONL
StructuredLogExporter(logger_name: str = "lazybridge")
OTelExporter(endpoint: str = None, *, exporter: Any = None)  # OpenTelemetry spans

Usage:
  Session(exporters=[
      ConsoleExporter(),
      JsonFileExporter("events.jsonl"),
      OTelExporter(endpoint="http://jaeger:4318"),
  ])

**rules**

- Each event is a ``dict`` with at minimum ``event_type``, ``session_id``,
  ``run_id`` (possibly ``None``). Agent/engine-specific fields are
  merged in by the emitter.
- Exporters fire in registration order; an exception in one does NOT
  block others (caught silently — wrap with ``CallbackExporter`` for
  debugging).
- ``FilteredExporter`` is a combinator — pass an inner exporter and a
  set of event_type strings to forward.
- ``OTelExporter`` requires ``pip install lazybridge[otel]``.

**example**

```python
from lazybridge import (
    Agent, Session,
    ConsoleExporter, JsonFileExporter, FilteredExporter,
    CallbackExporter, OTelExporter, EventType,
)

def on_error(event):
    if event["event_type"] == EventType.TOOL_ERROR:
        alert_pagerduty(event)

sess = Session(
    db="events.sqlite",
    exporters=[
        JsonFileExporter("run.jsonl"),
        FilteredExporter(
            CallbackExporter(on_error),
            event_types={EventType.TOOL_ERROR, EventType.AGENT_FINISH},
        ),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
)

Agent.chain(researcher, writer, session=sess)("…")
```

**pitfalls**

- Slow exporters block the engine — ``emit`` is synchronous per
  exporter. For high-volume paths, wrap with a queue + worker (or push
  to a log aggregator via ``JsonFileExporter``).
- Exporter exceptions are caught silently; if events don't arrive,
  temporarily wrap with ``CallbackExporter(print)`` to confirm.
- ``StructuredLogExporter`` is a thin wrapper over Python ``logging``
  — it inherits your logger's handlers / format.

**see-also**

[session](session.md),
decision tree: [parallelism](../decisions/parallelism.md)

## GraphSchema

**signature**

GraphSchema(session_id: str = "") -> GraphSchema

graph.add_agent(agent: Agent) -> None
graph.add_router(router) -> None
graph.add_edge(from_id, to_id, *, label="", kind=EdgeType.TOOL) -> None
graph.nodes() -> list[_BaseNode]
graph.edges() -> list[Edge]
graph.edges_from(node_id) / edges_to(node_id) -> list[Edge]

graph.to_dict() / to_json(indent=2) / to_yaml() -> str | dict
GraphSchema.from_dict / from_json / from_file -> GraphSchema
graph.save(path: str)     # .json or .yaml by extension

NodeType (StrEnum):  AGENT, ROUTER
EdgeType (StrEnum):  TOOL, CONTEXT, ROUTER

Auto-populated: every Agent(session=s) registers into s.graph.
Every as_tool wrapping records an edge with label="as_tool".

**rules**

- Nodes are ``AgentNode`` (provider, model, system) or ``RouterNode``
  (routes, default). ``add_agent`` reads ``agent.id`` / ``name`` /
  ``engine.provider`` / ``engine.model`` (duck-typed).
- ``session.register_tool_edge(outer, inner, label=…)`` adds an
  ``EdgeType.TOOL`` edge manually if you're wiring outside of
  ``as_tool`` (rare).
- Serialisation is descriptor-only: reconstructing a runnable pipeline
  from a saved graph is the caller's job.

**example**

```python
from lazybridge import Agent, Session

sess = Session()
researcher = Agent("claude-opus-4-7", name="researcher", session=sess)
writer     = Agent("claude-opus-4-7", name="writer",     session=sess)

orchestrator = Agent(
    "claude-opus-4-7",
    name="orchestrator",
    tools=[researcher, writer],   # as_tool edges registered automatically
    session=sess,
)

print(sess.graph.to_json(indent=2))
# {
#   "session_id": "...",
#   "nodes": [AgentNode(researcher), AgentNode(writer), AgentNode(orchestrator)],
#   "edges": [
#     Edge(from=orchestrator, to=researcher, label="as_tool", type="tool"),
#     Edge(from=orchestrator, to=writer,     label="as_tool", type="tool"),
#   ]
# }

# Persist + reload.
sess.graph.save("topology.yaml")
from lazybridge import GraphSchema
replay = GraphSchema.from_file("topology.yaml")
assert len(replay.nodes()) == 3
```

**pitfalls**

- An Agent without ``session=`` is not registered anywhere. If you pass
  it as a nested tool to an Agent with a session, the outer Agent
  propagates its session down and registers the nested one for you.
- ``to_yaml`` requires PyYAML (``pip install lazybridge[yaml]``);
  ``to_json`` is stdlib-only.
- ``from_dict`` reconstructs descriptors only — the ``provider`` /
  ``model`` strings on ``AgentNode`` are not live ``LLMEngine``s.

**see-also**

[session](session.md), [agent](agent.md),
[plan_serialize](plan-serialize.md)

## verify=

**signature**

# Three placements, same judge contract.

# 1. Agent-level (final output gate)
Agent("model", verify=judge_agent, max_verify=3, ...)

# 2. Tool-level (every call through the tool gated — "Option B")
agent.as_tool(name, description, verify=judge_agent, max_verify=3)

# 3. Plan-level (per-step, via agent-as-step with verify=)
Plan(Step(Agent(..., verify=judge_agent), ...))

# Judge contract
# Judge receives the agent's output text (and the original task for
# context) and must respond with a string starting with
# "approved" (case-insensitive) to accept. Anything else is treated
# as a rejection; its text is injected as feedback on the next retry.
# Judges may be Agents or plain callables: `Callable[[str], Any]`.

**rules**

- Retry loop: up to ``max_verify`` attempts. Final attempt is returned
  as-is even if still rejected (no infinite loop).
- Rejection feedback is appended to the task string for the next
  attempt: ``f"{original_task}\n\nFeedback: {judge_verdict}"``.
- Agent-level ``verify=`` gates the Agent's final output, regardless of
  which tool chain the engine chose internally.
- Tool-level ``verify=`` (Option B via ``as_tool``) gates every
  invocation of that specific wrapped agent — useful when one
  sub-agent is the risky one and the rest is fine.
- Plan-level is just a special case of agent-level: wrap the step's
  agent with its own ``verify=``.

**example**

```python
from lazybridge import Agent, LLMEngine, Plan, Step

# Judge needs a system prompt; ``system=`` belongs on the engine.
# Use a cheaper model in real code.
judge = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system='Respond "approved" or "rejected: <short reason>".',
    ),
    name="judge",                    # label in session.usage_summary()
)

# Agent-level:
#   verify=judge   gate the writer's final output through the judge.
#   max_verify=2   at most 2 attempts; the last one is returned as-is.
writer = Agent("claude-opus-4-7", verify=judge, max_verify=2)
writer("write a haiku about bees")

# Tool-level (Option B):
#   name="synthesizer" is the tool name the orchestrator LLM sees (and
#   the default Tool.name when wrapped via as_tool()).
#   as_tool("synth", verify=..., max_verify=...) overrides the tool name
#   and gates every call through this tool (not the orchestrator's own
#   final output).
synthesizer = Agent("claude-opus-4-7", name="synthesizer")
orchestrator = Agent(
    "claude-opus-4-7",
    tools=[synthesizer.as_tool("synth", verify=judge, max_verify=2)],
)

# Plan-level — Step(target, name="…") — ``name`` is the step id used by
# sentinels (from_step("fetch")), checkpoints, and the graph view.
plan = Plan(
    Step(fetcher, name="fetch"),
    Step(Agent("claude-opus-4-7", verify=judge, name="summarise"),
         name="summarise"),
    Step(publisher, name="publish"),
)
```

**pitfalls**

- A strict judge + small ``max_verify`` silently returns poor output.
  Log the retry feedback during development so you know when you're
  hitting the cap.
- Judges as *callables* returning booleans don't produce feedback;
  retries reuse the same task. Return a string verdict if you want the
  feedback loop.
- Nested verify (Agent-level + tool-level + Plan-level all on the
  same path) is allowed but expensive. Pick one per agent unless
  you're intentionally stacking.

**see-also**

[agent](agent.md), [as_tool](as-tool.md), [plan](plan.md),
[guards](guards.md), [evals](evals.md),
decision tree: [verify_placement](../decisions/verify-placement.md)
