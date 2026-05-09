## signature
# Plan-only sentinels (resolve against Plan execution history)
from_prev                    # singleton — previous step's output (default)
from_start                   # singleton — original user task
from_step(name: str)         # named prior step's output
from_parallel(name: str)     # named parallel branch's output
from_parallel_all(name: str) # aggregate every branch in a parallel band;
                             # payload is a labelled-text string, same as task

# Universal sentinels (work inside Plan and standalone)
from_memory(name: str)       # another agent's live memory at execution time
from_agent(name: str)        # last output of named agent from shared Store

# All sentinels are valid on:
Step(..., task=<sentinel>)
Step(..., context=<sentinel>)
Step(..., context=[<sentinel>, <sentinel>, "literal string"])

## rules
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
- ``from_parallel_all("n")``: aggregates every consecutive parallel step
  starting at ``"n"`` into one Envelope whose ``task`` and ``payload``
  are both a labelled-text join (``"[name]\\n<text>\\n\\n..."``).
  ``"n"`` must be the FIRST step of the band; the compile-time check
  rejects mid-band references.
- ``from_memory("n")``: reads the live ``Memory`` of the agent whose
  ``as_tool("n")`` key is ``"n"``. Resolved at step execution, not at
  plan construction. If the agent has no memory or it's empty, contributes
  nothing (silent no-op). PlanCompiler validates that ``"n"`` is an agent
  tool with ``memory=`` attached.
- ``from_agent("n")``: reads the last output of the agent registered as
  ``"n"`` from the shared ``Store``. Requires the tool to be an agent
  (``returns_envelope=True``) AND the source agent must have ``store=``
  attached; PlanCompiler rejects both violations at construction time.
  The store key is always the alias passed to ``as_tool("n")``, never
  the agent's internal ``name=``. If the key is absent at runtime
  (agent hasn't run yet), contributes nothing (silent no-op).
  Prefer ``from_step("n")`` inside a single Plan — ``from_agent`` is
  for cross-run or cross-plan data dependencies.
- A plain string passed as ``task=`` is used verbatim — useful for
  hard-coded prompts at intermediate steps.
- ``context=`` accepts a single sentinel/string OR a **list** of them.
  Each list item resolves independently; the parts join with
  blank-line separators in the step's ``Envelope.context``.  Mix
  sentinels with literal strings to inject fixed boilerplate alongside
  upstream data without an intermediate combiner step.

## narrative
Sentinels are how Plan steps declare "where does my input come from?".
Without them you'd thread arguments manually at every step; with them,
the data flow is a 1-liner per step.

Two categories:

**Plan-only** — resolve against the Plan's execution history:

* `from_prev` — the workhorse. In a straight chain, every step reads
  the one before it. This is the default.
* `from_start` — "I don't care what the previous step said; I want the
  original user task." Useful for verification steps ("does this draft
  answer the user's actual question?") or for branches that skip
  intermediate processing.
* `from_step("name")` — "I need step X specifically, even though it ran
  three steps ago." Sentinels validate against known step names at
  plan compile time; a typo is caught before any LLM call.
* `from_parallel("name")` — same mechanic as `from_step`, but reads
  better at the call site when the referenced step is a parallel branch.
* `from_parallel_all("name")` — aggregate all consecutive parallel siblings
  starting at `name` into one labelled-text Envelope.

**Universal** — work both inside Plan and when agents are called standalone:

* `from_memory("name")` — reads the **live memory** of the agent registered
  under `name` at step execution time. Never at construction — always
  reflects the most recent conversation history of that agent.
* `from_agent("name")` — reads the **last output** of the agent registered
  under `name` from a shared `Store`. Every Agent writes its output to
  `store` after a successful run; `from_agent` reads it back.
  Works inside Plan and outside (LLM orchestrators, standalone calls).

Sentinels are also valid on `context=` — use them to inject prior
context into a step without overriding its task.

## When to use from_agent vs from_step

Inside the same Plan, **prefer `from_step("name")`**:

```python
# Clear, validated at compile time, no store required
Step("write", context=from_step("research"))
```

Use `from_agent("name")` only when you intentionally want the agent's
**last stored output independent of this Plan's step history** — for example:

- Reading across Plan runs (previous execution, stored in SQLite).
- A standalone LLM orchestrator where there is no step history.
- A step that needs the output of an agent called outside this Plan.

```python
# from_agent reads from Store, not from step history.
# Requires the referenced agent to have store= attached.
Step("write", context=from_agent("research"))
```

The store key is always the **tool alias** passed to `as_tool("alias")`,
not the agent's internal `name=` attribute.

## example
```python
from lazybridge import Agent, LLMEngine, Memory, Plan, Step, Store
from lazybridge import from_prev, from_start, from_step, from_memory, from_agent

store = Store(db="pipeline.sqlite")
mem = Memory(strategy="summary")

researcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    memory=mem,
    store=store,
    name="research",
)
fact_checker = Agent(engine=LLMEngine("claude-opus-4-7"), store=store)
writer = Agent(engine=LLMEngine("gpt-4o"))

plan = Agent(
    engine=Plan(
        Step("research"),
        # fact_checker sees researcher's output as task, original user task as context
        Step("fact_check",
             task=from_prev,
             context=from_start),
        # writer sees researcher's live memory AND the fact_checker output
        Step("write",
             context=[from_memory("research"), from_step("fact_check")]),
    ),
    tools=[
        researcher.as_tool("research"),
        fact_checker.as_tool("fact_check"),
        writer.as_tool("write"),
    ],
    store=store,
)

# from_agent works outside Plan too — any agent can read another's last output.
standalone = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[researcher.as_tool("research")],
    store=store,
)
standalone("find AI trends")
# Now any agent sharing the same store can read researcher's output:
value = store.read("__agent_output__:research")
```

```python
# Multi-source context via list — no combiner step needed.
plan2 = Agent(
    engine=Plan(
        Step("research"),
        Step("policy"),
        Step("synth",
             task="Draft a brief citing both sources.",
             context=[
                 from_step("research"),
                 from_step("policy"),
                 "Style: neutral, third person, no superlatives.",
             ]),
    ),
    tools=[researcher.as_tool("research"), policy_loader.as_tool("policy"), synthesiser.as_tool("synth")],
)
```

## pitfalls
- ``from_prev`` after a parallel branch returns the join step's output,
  not one of the branches. Use ``from_parallel("<branch-name>")`` for a
  specific branch.
- Sentinels are module-level imports; don't shadow them with local
  variables of the same name.
- When passing a ``str`` as ``task=``, it's treated as a LITERAL, not a
  sentinel. Don't write ``task="from_prev"`` expecting the sentinel.
- ``from_memory`` reads at execution time — if the agent hasn't run yet,
  it contributes nothing (silent no-op, not an error).
- ``from_agent`` requires the tool to be an agent (via ``as_tool()``),
  not a plain function. PlanCompiler rejects plain-function targets at
  construction time.
- ``from_agent`` requires the source agent to have ``store=`` attached.
  PlanCompiler rejects it at construction time if the agent has no store.
- Inside a plain sequential Plan, ``from_step()`` is clearer and
  lighter — it reads from in-memory step history, needs no store.

## see-also
- [Plan](plan.md) — the engine that interprets sentinels.
- [Parallel plan steps](parallel-steps.md) — `from_parallel_all` aggregation.
- [Envelope](envelope.md) — the object sentinels carry between steps.
- [Store](store.md) — shared blackboard used by `from_agent`.
- [Memory](memory.md) — per-agent conversation context used by `from_memory`.
