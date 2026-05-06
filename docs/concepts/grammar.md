# The LazyBridge Grammar

**Read this before anything else.**

LazyBridge has one central abstraction: `Agent`. Every agent — a single
LLM call, a multi-step pipeline, a human approval gate, a nested
specialist — has the same shape:

```python
Agent(
    engine=...,    # the brain
    tools=[...],   # the capabilities
    memory=...,    # the context
    session=...,   # the observability
)
```

The engine is the only thing that changes. Everything else is uniform
across the entire system.

---

## The Three Pieces

### Engine — the brain

The engine decides what the agent does when called.

| Engine | Decides |
|---|---|
| `LLMEngine("claude-opus-4-7")` | A language model chooses the next action |
| `Plan(Step(...), Step(...))` | A declared sequence of steps runs in order |
| `HumanEngine(timeout=60)` | A person approves or redirects |
| `SupervisorEngine(...)` | A REPL loop with retry / override capability |

All engines implement the same protocol. Swapping the engine changes the
behaviour of the agent without changing anything else about it.

### Tools — the capabilities

`tools=[...]` is what the agent can call. A tool can be:

- a plain Python function
- another `Agent` (via `.as_tool("name")`)
- an MCP server (via `MCP.from_transport(...)`)
- a provider-native tool (via `native_tools=[NativeTool.WEB_SEARCH]`)

The tool surface is always the same, regardless of what's inside.

### State — memory, store, session, guard, output

These carry continuity and observability through the system:

- `memory` — conversation history across turns (in-prompt context)
- `store` — shared blackboard across agents and runs (durable key-value); agents write their last output here automatically after each run
- `session` — structured event log for tracing and observability
- `guard` — pre/post call content policy
- `output` — expected structured output type

---

## The Canonical Form

Declare sub-agents first, then compose them into an orchestrator.

```python
from lazybridge import Agent, LLMEngine, Plan, Step, Memory, Session
from lazybridge.sentinels import from_prev, from_step

# 1. Sub-agents — each with its own engine, tools, and fixed context
researcher = Agent(
    engine=LLMEngine("claude-opus-4-7", system="You are a research expert."),
    tools=[search],
)

writer = Agent(
    engine=LLMEngine("gpt-4o", system="You are a concise technical writer."),
)

# 2. Deterministic orchestrator — Plan engine controls the sequence
pipeline = Agent(
    engine=Plan(
        Step("research"),                                        # calls researcher
        Step("write", task=from_prev, context=from_step("research")),  # calls writer
    ),
    tools=[
        researcher.as_tool("research"),   # "research" = key in tool map
        writer.as_tool("write"),          # "write"    = key in tool map
    ],
    memory=Memory(strategy="summary"),
    session=Session(),
)

# 3. Dynamic orchestrator — LLM engine decides which tools to call
orchestrator = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[
        researcher.as_tool("research"),
        writer.as_tool("write"),
    ],
    memory=Memory(),
    session=Session(),
)
```

The Plan and LLM orchestrators are identical in structure. Only the
engine differs.

---

## The Name Chain

The string passed to `as_tool("name")` is the currency that connects
every part of the system. It must be consistent across:

```
researcher.as_tool("research")   →  key in the tool map: "research"
                                              ↓
Step(target="research")          →  PlanCompiler validates "research" exists ✓
                                              ↓
routes={"research": predicate}   →  routes to the step named "research" ✓
                                              ↓
from_step("research")            →  reads output of step "research" ✓
```

When a string target and step name are the same — which is the normal
case — `Step("research")` is all you need. The `name=` parameter on
`Step` only matters when you want the step's identity to differ from the
tool it calls:

```python
Step(target="research", name="phase_1")
# Calls the "research" tool, but this step is identified as "phase_1"
# in routing rules, sentinels, and checkpoints.
```

If routing silently does nothing, the first thing to check is whether
the target in `routes={"name": ...}` matches `step.name`, not
`step.target`.

---

## Task and Context in Steps

Each `Step` controls what input it passes to the tool it calls:

- **`task`** — the main input string sent to the tool. Default:
  `from_prev` (the output of the previous step). The orchestrator
  generates the task; sub-agents receive it.
- **`context`** — additional context injected alongside the task.
  Useful for passing results from earlier steps to a later one without
  making it the main input.

```python
Step("write",
     task=from_prev,                  # use the previous step's output as the task
     context=from_step("research"),   # inject the "research" step's output as context
)
```

Available sentinels:

| Sentinel | Scope | Resolves to |
|---|---|---|
| `from_prev` | Plan-only | Output of the immediately preceding step |
| `from_start` | Plan-only | The original input to the whole Plan |
| `from_step("name")` | Plan-only | Output of the named step (in-memory, no store needed) |
| `from_parallel("name")` | Plan-only | Output of one specific parallel branch |
| `from_parallel_all("name")` | Plan-only | Outputs of all branches in a parallel band |
| `from_memory("name")` | Universal | Live memory of the agent mounted as `name`, at execution time |
| `from_agent("name")` | Universal | Last stored output of the agent mounted as `name` (from shared Store) |

**Choosing between `from_step` and `from_agent`** — inside the same Plan,
`from_step("research")` is the standard choice: in-memory, no store
required, validated at compile time.  Use `from_agent("research")` only
when the data dependency crosses run or plan boundaries:
- last-known output from a previous execution
- an agent called by an LLM orchestrator outside this Plan
- a standalone agent writing to a shared Store for later consumption

The alias passed to `as_tool("research")` is always the authoritative
key — both for `from_step` (step identity) and `from_agent` (store key).

The fixed context of a sub-agent — its role, its persona, its
constraints — belongs on the engine: `LLMEngine("model", system="...")`.
The dynamic context that flows between steps belongs on `Step`.

---

## String Shortcut

`Agent("claude-opus-4-7")` is sugar for `Agent(engine=LLMEngine("claude-opus-4-7"))`.

It is valid everywhere and useful for quick scripts or simple agents.
Use the explicit `LLMEngine(...)` form when you need to configure the
engine beyond the model string:

```python
# Sugar — fine for simple cases
Agent("claude-opus-4-7", tools=[search])

# Explicit — required when configuring the engine
Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You are a senior software engineer.",
        max_turns=10,
        thinking=True,
        temperature=0.2,
    ),
    tools=[search],
)
```

---

## Factory Methods Are Sugar Too

Every factory method is a shortcut over the canonical form:

| Factory | Expands to |
|---|---|
| `Agent.from_model("model", **kw)` | `Agent(engine=LLMEngine("model"), **kw)` |
| `Agent.from_plan(*steps, **kw)` | `Agent(engine=Plan(*steps), **kw)` |
| `Agent.from_chain(a, b, **kw)` | `Agent(engine=Plan(Step(a), Step(b)), **kw)` |
| `Agent.from_engine(e, **kw)` | `Agent(engine=e, **kw)` |

Use whichever form makes the code clearest. The canonical form is always
valid and always shows exactly what engine is inside.

---

## Growing a System

Start minimal, add only what you need:

```python
# Start — one agent, no config
Agent("claude-opus-4-7")("hello").text()

# Add tools
Agent("claude-opus-4-7", tools=[search])("find AI news").text()

# Add a sub-agent as a tool
researcher = Agent(engine=LLMEngine("claude-opus-4-7"), tools=[search])
Agent("claude-opus-4-7", tools=[researcher.as_tool("research")])("summarise AI news")

# Add memory and observability
Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[researcher.as_tool("research")],
    memory=Memory(),
    session=Session(console=True),
)

# Make execution deterministic
Agent(
    engine=Plan(Step("research"), Step("write")),
    tools=[researcher.as_tool("research"), writer.as_tool("write")],
    memory=Memory(strategy="summary"),
    session=Session(),
)
```

The shape never changes. Only the engine and the mounted capabilities grow.
