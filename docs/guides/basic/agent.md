# Agent

The single class you build, configure, run, and compose. Every other
primitive in LazyBridge — engines, tools, plans, sessions, memories,
guards — is something you wire **into** an `Agent`.

## Signature

```python
from lazybridge import Agent

agent = Agent(
    engine=...,                    # required: LLMEngine / Plan / HumanEngine / SupervisorEngine / custom
    tools=[...],                   # callables, Tools, Agents, ToolProviders
    output=str,                    # str (default) or a Pydantic model class
    memory=None,                   # Memory instance for conversation continuity
    session=None,                  # Session for event tracking + observability
    name=None,                     # surface name (used as a tool name when this Agent is composed)
    description=None,              # human-readable description (LLM-facing when used as a tool)
    verbose=False,                 # print turn-by-turn updates to stdout
    sources=(),                    # static documents prepended to every turn
    guard=None,                    # Guard / GuardChain — input/output filtering
    verify=None,                   # Agent or callable — judge-and-retry loop
    max_verify=3,                  # retries when verify=...
    native_tools=None,             # list[NativeTool | str] — provider-hosted tools
    allow_dangerous_native_tools=False,  # security gate: opt-in for CODE_EXECUTION / COMPUTER_USE
    output_validator=None,         # callable validator over the payload
    max_output_retries=2,          # retries on output validation failure
    timeout=None,                  # total deadline for the run (seconds)
    max_retries=3,                 # provider transient-error retries
    retry_delay=1.0,               # base delay between retries (exponential backoff)
    fallback=None,                 # secondary Agent invoked on primary failure
    cache=False,                   # bool or CacheConfig — prompt caching
)

# Calling
result = agent(task)               # sync, returns Envelope (canonical)
result = await agent.run(task)     # async equivalent
async for chunk in agent.stream(task): ...   # streaming form
```

For factory and composition shortcuts (`Agent.from_provider`,
`Agent.chain`, `Agent.parallel`), see [Canonical vs sugar](../../concepts/canonical-vs-sugar.md).

## Synopsis

An `Agent` is the composition `Engine + Tools + State`:

- The **engine** decides what happens next. `LLMEngine` is the most
  common — an LLM that picks tools and arguments dynamically. Swap it
  for `Plan` to get deterministic orchestration, `HumanEngine` to gate
  at a human approval, or `SupervisorEngine` for a REPL.
- **Tools** are everything the agent can invoke. Plain Python
  functions, other agents, `Plan`-backed pipelines, MCP servers,
  provider-native capabilities — they all live in `tools=[...]`.
- **State** is what persists across or alongside the run. `Memory`
  carries conversation history; `Session` records events; the result
  `Envelope` carries the typed payload plus token / cost / latency
  metadata.

Calling `agent(task)` runs the engine to completion and returns an
`Envelope`. The same `Agent` shape supports a one-shot helper, a
hierarchical multi-agent system, and a checkpointed production
pipeline — only the `engine=` argument changes.

## When to use it

- **Any single LLM interaction** — one-shot call, tool use, or
  structured output. `Agent` is the only class you need for
  Basic-tier work; everything else is opt-in via keyword args.
- **Building blocks for composition.** Two agents passed in
  `tools=[...]` of a third agent forms the supervisor pattern. A list
  of agents passed to `Plan` becomes a deterministic pipeline. The
  same `Agent` is the unit at every level.
- **Tier upgrades.** Add `output=` for structured output, `memory=`
  for conversation continuity, `session=` for observability,
  `verify=` for high-stakes outputs, `cache=True` for prompt caching
  — without changing the run-loop you've already written.

## When NOT to use it

- **Pure deterministic logic.** Don't wrap arithmetic, file parsing,
  or HTTP calls in an `Agent` — they go in `tools=[...]` or remain
  plain functions. The agent's job is to decide *when* to call them.
- **Streaming-only callsites where you can drop to `LLMEngine`
  directly.** If you genuinely don't need tools, memory, sessions,
  guards, or any other agent-level feature, `LLMEngine(...).stream(...)`
  works — but in practice this is rare; almost every use grows into
  needing at least one of those features.
- **Cases where you want a graph DSL.** LazyBridge expresses
  composition in plain Python (`Agent`, `Plan`, `Step`). If you need
  a separate graph definition language, you're on the wrong
  framework.

## Example

```python
from lazybridge import Agent, LLMEngine, Session
from pydantic import BaseModel


# 1) Minimal agent.
agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
)
result = agent("hello")
print(result.text())


# 2) Tools — auto-schema from type hints + docstring.
def search(query: str) -> str:
    """Search the web for ``query`` and return the top three hits."""
    return "..."

researcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[search],
    name="research",
)
print(researcher("AI news April 2026").text())


# 3) Structured output — read .payload, not .text().
class Summary(BaseModel):
    title: str
    bullets: list[str]

summariser = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    output=Summary,
)
result = summariser("Summarise LazyBridge in three bullets.")
print(result.payload.title)
print(result.payload.bullets)


# 4) Tool-is-Tool composition (Agents wrap Agents).
editor = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[researcher],            # researcher.name="research" becomes the tool name
    name="editor",
)
print(editor("find papers on bees and write a one-paragraph summary").text())


# 5) output_validator — application invariants on top of Pydantic.
class DateRange(BaseModel):
    start_date: str
    end_date: str


def chronological(payload: DateRange) -> DateRange:
    """Re-prompt up to max_output_retries times if start > end."""
    if payload.start_date > payload.end_date:
        raise ValueError(
            f"start_date ({payload.start_date}) must precede end_date ({payload.end_date})"
        )
    return payload


extractor = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    output=DateRange,
    output_validator=chronological,
    max_output_retries=2,
)


# 6) Streaming — same Agent, drop down to .stream() for partial output.
import asyncio
from lazybridge import Agent, LLMEngine

async def stream_brief() -> None:
    agent = Agent(engine=LLMEngine("claude-opus-4-7"))
    async for chunk in agent.stream("Outline LazyBridge in five bullets."):
        print(chunk, end="", flush=True)

asyncio.run(stream_brief())


# 7) Cache — explicit TTL on Anthropic.
from lazybridge import CacheConfig

cached = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    cache=CacheConfig(ttl="1h"),       # "5m" (default) or "1h" on Anthropic
)


# 8) Production-shape: timeout + cache + provider fallback + tracing.
fallback_agent = Agent(
    engine=LLMEngine("gpt-5"),
    tools=[search],
    name="fallback",
)
prod = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[search],
    timeout=30.0,
    cache=True,
    fallback=fallback_agent,
    session=Session(db="events.sqlite"),
)
prod("draft a one-pager on the LazyBridge audit findings")
```

## Pitfalls

- **`output=SomeModel` + `.text()`** — calling `.text()` on a
  structured envelope returns the JSON dump of the payload, which is
  rarely what you want. Read `.payload` instead.
- **`verify=` semantics** — the judge must return a verdict starting
  with `"approved"` (case-insensitive) to accept. Anything else is
  treated as rejection plus feedback for the next attempt. Bound the
  loop with `max_verify=...`.
- **`guard=` blocks the engine.** A blocked input or output produces
  an error `Envelope` without invoking the engine — `result.ok` is
  `False`, `result.error.type` is `"GuardError"`. Don't expect the
  agent to "see" the rejected text and self-correct; guards are
  hard gates.
- **`timeout=None`** (default) leaves the run unbounded. Tool calls
  inside a runaway agent can block forever. Pick a deadline that
  matches your SLO.
- **`fallback=` runs the fallback's full pipeline** — its tools,
  memory, and guards — on the same envelope, with the primary's
  error threaded into `context`. Configure compatible `output=` and
  `tools=` on both agents, or the fallback may fail differently.
- **`output_validator=`** is a callable applied to the payload
  *after* Pydantic validation passes (or directly when `output=str`).
  Receives the payload, returns the validated payload (may
  transform). Raise to reject — the framework re-prompts up to
  `max_output_retries` times with the validator's error message
  threaded back into the prompt. Useful for application-level
  invariants that aren't expressible in the Pydantic schema (e.g.
  "the `start_date` field must come before `end_date`").
- **`cache=True`** enables prompt caching where the provider supports
  it (Anthropic explicit, OpenAI / DeepSeek auto). Pass
  `CacheConfig(ttl="1h")` for the longer Anthropic TTL.
- **Nested agents inherit the caller's `session=`** when they have
  none of their own. This is what gives you transitive cost rollup
  and a single graph view of the whole tree — pass an explicit
  `session=None` on a sub-agent only when you genuinely want it
  invisible.
- **Fleet config via dict spread** — the 0.7-era ``runtime`` /
  ``resilience`` / ``observability`` configs were deleted in 0.8.0 (they
  carried a ``flat kwarg > config object > default`` precedence game
  with a private ``_UNSET`` sentinel — an LLM trap).  Share kwargs
  across a fleet via a Python dict::

      PROD_DEFAULTS = dict(timeout=60, max_retries=5, cache=True, session=sess)
      Agent(**PROD_DEFAULTS, engine=LLMEngine("model"), name="agent-X")

## See also

- [Tool](tool.md) — how plain Python functions become tools the
  agent can call.
- [Envelope](envelope.md) — the typed result every agent returns.
- [Native tools](native-tools.md) — provider-hosted alternatives
  via `native_tools=[...]`.
- [Mental model](../../concepts/mental-model.md) — the
  Engine + Tools + State decomposition.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) — every
  factory and shortcut, with its canonical equivalent.
