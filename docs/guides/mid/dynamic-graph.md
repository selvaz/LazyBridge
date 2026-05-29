# Dynamic graph (AgentPool + conclude)

Build a multi-agent system whose **topology is decided by the LLM at
runtime**: agents delegate to each other by name, and any agent —
however deeply nested — can end the whole task. This is the dynamic
counterpart to a `Plan` (a fixed DAG you declare up front) and to plain
`tools=[other_agent]` composition (a one-way, statically-wired tree).

Two primitives, both ordinary tools the engine does not special-case:

```python
from lazybridge import AgentPool, conclude

pool = AgentPool(max_depth=25)     # routing registry → a `route` tool
# conclude(message)                # non-local exit → returns to the top
```

## Signature

```python
class AgentPool:
    def __init__(self, *, max_depth: int = 25) -> None: ...
    def register(self, *agents: Agent) -> None: ...        # populate AFTER construction
    def roster(self) -> str: ...                            # one line per agent
    async def route(self, agent_name: str, task: str) -> str: ...
    def as_tool(self, name: str = "route") -> Tool: ...     # for tools=[...]

def conclude(message: str) -> str: ...                      # raises ConcludeSignal
```

`pool.as_tool()` returns a `Tool` named `route` with the schema
`(agent_name: str, task: str) -> str`. `conclude` is a plain function;
drop it straight into `tools=[conclude]`.

## Synopsis

`AgentPool` solves the **circular-reference problem**. Agents that
delegate to each other can't all be passed into each other's
`tools=[...]` at construction — the tool map is frozen in
`Agent.__init__`. Instead, every agent references the *pool* (which
already exists) via `pool.as_tool()`, and the pool references the agents
through `register(...)`, called *after* they're built. At call time the
LLM invokes `route("bob", "...")` and the pool dispatches to the agent
registered under that name.

`conclude` provides a **non-local exit**. In a nested call chain
(`A → route → B → route → C`), `C` calling `conclude("done")` unwinds the
entire chain in one step and returns `Envelope(payload="done")` from the
*original* top-level `run()` — no need to thread the answer back up
level by level. It is implemented as a `ConcludeSignal` (a
`BaseException`) so it slips past the engine's tool-error handling; only
`Agent.run` catches it. Nested invocations (`as_tool`, `AgentPool.route`,
Plan agent-steps) run via an internal path that lets the signal keep
propagating.

`max_depth` bounds routing recursion. Because routes can loop
(`A → B → A → …`), the pool tracks call depth with a `contextvars`
counter and, past `max_depth`, returns a "call conclude now" message
instead of recursing — turning a would-be `RecursionError` into a
graceful nudge.

## When to use it

- **The next agent should be chosen by the model, not the wiring.** A
  triage agent that routes to specialists, a debate between personas, a
  "society of mind" where workers hand off opportunistically.
- **Agents need to call each other (cycles).** `AgentPool` is the only
  composition path that expresses `A ⇆ B`; direct `tools=[...]` cannot.
- **Any node may finish the task early.** A worker that discovers the
  answer should not have to pass it back through every caller — it calls
  `conclude`.
- **You want layered routing.** Give different pools to agents at
  different levels (`pool.as_tool("ask_team")` vs `peers.as_tool("ask_peer")`)
  to scope which neighbours each level can reach.

## When NOT to use it

- **The control flow is fixed.** If you already know the step order, use
  [`Plan`](../full/plan.md) (deterministic, checkpointable, validated at
  construction) or [`Agent.chain`](chain.md). A dynamic graph trades that
  guarantees for flexibility.
- **One agent simply calls another.** Plain `tools=[other_agent]` (see
  [As tool](as-tool.md)) is canonical and needs no pool.
- **You need typed hand-offs between agents.** `route` returns text; for
  structured payloads between stages use a `Plan` with
  `Step(output=Model)`.

## Example

```python
from lazybridge import Agent, AgentPool, LLMEngine, conclude

pool = AgentPool()

researcher = Agent(
    name="researcher",
    engine=LLMEngine("claude-opus-4-8", max_tool_calls_per_turn=1,
                     system="Gather facts, then route to 'writer'."),
    tools=[pool.as_tool(), conclude],
)
writer = Agent(
    name="writer",
    engine=LLMEngine("claude-opus-4-8", max_tool_calls_per_turn=1,
                     system="Write the answer, then call conclude(...)."),
    tools=[pool.as_tool(), conclude],
)

pool.register(researcher, writer)        # register AFTER construction

result = researcher.run("Summarise 2026 AI-policy trends in 3 bullets.")
print(result.text())                     # whatever 'writer' passed to conclude(...)
```

`researcher` may `route("writer", ...)`; `writer` ends the task with
`conclude(...)`, and its message surfaces from `researcher.run` — the
original call — regardless of how deep the routing went.

Layered routing — one agent, two pools, distinct tool names:

```python
orchestrator = Agent(
    name="orchestrator",
    engine=LLMEngine("claude-opus-4-8", max_tool_calls_per_turn=1),
    tools=[team.as_tool("ask_team"), peers.as_tool("ask_peer"), conclude],
)
```

## Pitfalls

- **Always set `max_tool_calls_per_turn=1` on members.** Without it the
  model can emit several `route`/`conclude` calls in one turn and the
  graph *branches* (every call still runs, just concurrently —
  `max_parallel_tools` bounds concurrency, **not** the number of calls).
  One call per turn keeps a single, traceable path.
- **`conclude` is not instantaneous if it shares a turn with other
  tools.** Same-turn siblings run to completion first (they execute via
  `asyncio.gather`), so a slow sibling delays the exit. `max_tool_calls_per_turn=1`
  removes the issue entirely.
- **Register after construction.** Agents take `pool.as_tool()`; the pool
  takes the agents via `register(...)`. Calling `route` for an
  unregistered name returns an "Unknown agent" message (not an error) so
  the model can recover.
- **`route` returns text, not a typed envelope.** Nested cost/token
  metadata is still rolled up, but structured payloads are flattened to
  `.text()`. Use a `Plan` step when you need a typed hand-off.
- **`max_depth` is per-pool.** Each pool counts its own routing depth via
  an independent `contextvars` counter; in a layered setup each pool
  bounds only its own recursion. Cross-pool cycles are still bounded, but
  more loosely.
- **`conclude` inside a `Plan` unwinds the whole plan.** A plan step that
  concludes skips the remaining steps and returns to the top-level
  `pipeline.run()` — it does not just end that step.

## See also

- [As tool](as-tool.md) — static one-way `agent → agent` composition.
- [Chain](chain.md) / [Plan](../full/plan.md) — fixed, deterministic
  control flow when the topology is known up front.
- [Everything is a tool](../../concepts/everything-is-a-tool.md) — why
  `route` and `conclude` need no special engine support.
- [Multi-agent graphs](../../reference/multi-agent.md) — API reference
  for `AgentPool`, `conclude`, `ConcludeSignal`.
