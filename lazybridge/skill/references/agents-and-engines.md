# Agents and engines

Read when building or calling an `Agent`, choosing between canonical form and
sugar, using structured output, or wiring `verify=` / `fallback=` / `guard=`
and deadlines. Source of truth: `lazybridge/agent.py`, `lazybridge/envelope.py`,
`lazybridge/engines/llm.py`; docs `docs/guides/basic/agent.md`,
`docs/concepts/canonical-vs-sugar.md`.

## Canonical shape

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-opus-5-5", system="Be concise.", max_turns=10),
    tools=[...],
    name="assistant",
)
result = agent("hello")       # sync, returns Envelope
print(result.text())
```

Lead with this form, one argument per line, `result = agent(task)` on its own
line. Treat shortcuts as sugar and mention them second.

Where each knob lives (verified against the constructors):

| On `LLMEngine(model, ...)` | On `Agent(...)` |
|---|---|
| `provider`, `system`, `thinking`, `max_turns`, `tool_choice` (`"auto"`/`"any"`), `temperature`, `native_tools`, `allow_dangerous_native_tools`, `max_retries`, `retry_delay`, `request_timeout` (120 s default, whole run incl. retries), `max_parallel_tools` (8), `max_tool_calls_per_turn`, `tool_timeout`, `stream_idle_timeout`, `cache`, `strict_multimodal`, `approval_gate`, `cwd` | `engine`, `tools`, `output`, `memory`, `store`, `sources` (static documents prepended every turn), `guard`, `verify`, `max_verify`, `name`, `description`, `session`, `verbose`, `output_validator`, `max_output_retries`, `timeout`, `tool_timeout`, `max_retries`, `retry_delay`, `fallback`, `cache` |

`Agent(model=...)` / `Agent(native_tools=...)` exist as pass-throughs, but the
canonical place for model and native tools is the engine.

## Calling an agent

| Form | When |
|---|---|
| `result = agent(task)` | Canonical. Works even inside a running event loop. |
| `result = await agent.run(task)` | Inside an existing `async def`. |
| `async for chunk in agent.stream(task):` | Incremental tokens/events. |
| `agent(task, images=[ImageContent...])` | Multimodal input (`ImageContent`, `AudioContent`; provider/engine support varies). |

## The Envelope

Fields: `task`, `context`, `images`, `audio`, `payload`, `metadata`, `error`.
`result.ok` is execution success, not acceptance. `result.error` is an
`ErrorInfo` record (not an exception). `result.text()` renders the payload
(Pydantic models and JSON-serialisable values included); `result.payload`
carries the typed value. `metadata` holds model, tokens, `cost_usd`, latency
and `nested_*` roll-ups from sub-agents.

## Structured output

```python
from pydantic import BaseModel


class Summary(BaseModel):
    headline: str
    bullets: list[str]


agent = Agent(engine=LLMEngine("claude-opus-5-5"), output=Summary, name="summariser")
result = agent("Summarise the news")
print(result.payload.headline)       # .payload, never .text() parsing
```

`output_validator=fn` adds application invariants on top of Pydantic; the
agent re-prompts up to `max_output_retries` (default 2) with the error.

## Sugar catalogue (not all are pure aliases)

| Sugar | Canonical / meaning |
|---|---|
| `Agent("claude-opus-5-5", **kw)` | Pure alias of `Agent(engine=LLMEngine("claude-opus-5-5"), **kw)`. Hides the engine; avoid in tutorials. |
| `Agent.from_provider("anthropic", tier="top", **kw)` | Not pure: resolves a tier alias (`super_cheap`/`cheap`/`medium`/`expensive`/`top`, type `Tier`) to the provider's current model. Default tier `medium`. See `providers-and-models.md`. |
| `Agent.chain(a, b, **kw)` | Builds `Agent(engine=Plan(Step(a, name=a.name), Step(b, name=b.name)))`. Linear only. |
| `Agent.parallel(*agents, concurrency_limit=, step_timeout=)` | Returns a `ParallelAgent`; calling it returns ONE `Envelope` whose `.text()` is a labelled join (`[name] ...`), costs rolled up, first error in `.error`. Per-branch envelopes: `await par.run_branches(task)`. |
| `agent.derive(tools=[...], **overrides)` | New agent: a clone with tools APPENDED and other constructor args replaced. Does not mutate the original. |
| `agent.as_tool(name, description, verify=, max_verify=)` | Rename an agent for use as a tool, or attach a tool-level judge loop. |
| `human_agent(...)` / `supervisor_agent(...)` | `lazybridge.ext.hil` factories; kwarg split engine/agent. See `extensions.md`. |

Deleted in 0.7.9 and never to be generated: `Agent.from_model`,
`from_engine`, `from_chain`, `from_plan`, `from_parallel`; the `runtime=`,
`resilience=`, `observability=` config objects. Pass flat kwargs, or share
defaults with `**PROD_DEFAULTS`.

## verify=, fallback=, guard=

- **`verify=judge`** (Agent or sync `Callable[[str], Any]`) is a bounded
  judge-and-retry loop. `max_verify` counts total attempts (default 3, >= 1).
  An Agent judge sees output plus original task; rejection feedback is placed
  in `Envelope.context` as `Feedback from judge: ...`. Approval words include
  approved/accept/allow/pass/ok/yes/good/valid; unknown verdicts reject.
  **On exhaustion the last draft is returned with no approval flag**:
  verify is not release acceptance. Three placements: Agent-level,
  `as_tool(verify=)`, or a Step target that has its own `verify=`.
  Callable judges are synchronous; do not assume async callables.
- **`fallback=other_agent`** fires when the agent returns an error envelope
  (including a `Plan` step that raised). It does **not** reliably fire when a
  tool raises inside an `LLMEngine` tool loop (the model sees a tool error and
  may work around it) nor when a custom engine raises directly. Keep
  `try`/`except` in those cases; return a typed failure from tools that must
  abort the run.
- **`guard=`** is a hard gate (`ContentGuard`, `LLMGuard`, `GuardChain`,
  `DeduplicateGuard`; see `sessions-and-memory.md`). A guard blocks; verify
  re-asks.

## Deadlines and retries

- `Agent(timeout=N)` only fires at an `await`; a blocking sync tool never
  yields one. For tools use `Tool(timeout=)` > `Agent(tool_timeout=)` >
  `LLMEngine(tool_timeout=)` (most specific wins). A timed-out sync tool runs on
  a daemon thread and is abandoned (`ToolTimeoutError`, carries `.tool_name`
  and `.timeout`); a `UserWarning` fires after 8 abandoned workers.
- Engine `request_timeout` (default 120 s) bounds the whole engine run
  *including retries*. Raise it for long tasks and align any outer host (MCP
  client) timeout with it.
- `StreamStallError` signals a stream idle longer than `stream_idle_timeout`.

## Custom engines

Implement the `Engine` protocol from `lazybridge.engines.base` (there is no
`BaseEngine`). See `docs/guides/advanced/engine-protocol.md`. Prefer an
existing engine plus tools before writing one.

## Anti-patterns

- `Agent(..., system="...")` — `system` belongs to the engine.
- Reading structured output via `.text()` and re-parsing it.
- Wrapping every helper in its own sub-agent; pick the lowest rung of the
  progressive-complexity ladder (`docs/concepts/progressive-complexity.md`).
- Passing free-form text between agents where a typed `output=` model or a
  `Store` key would carry the state.
- Treating `result.ok` as "the answer is correct".
