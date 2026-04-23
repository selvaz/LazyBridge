# Agent

`Agent` is the only public abstraction in LazyBridge. Every pipeline —
one-shot LLM calls, REPL supervisors, multi-step Plans — is an `Agent`
with a different engine plugged into it. The call surface does not change.

The key contract to internalise is **tool is tool**. When you pass things
to `tools=[...]`, it doesn't matter whether each entry is a plain function,
another Agent, or an Agent that itself has Agents as tools. The composition
is closed under `Tool`, so you can build deeply-nested agent trees and the
outer engine treats them identically to a one-line function.

The second contract is **parallelism is free, not configured**. If the
engine decides to call five tools in one step, LazyBridge fans them out
with `asyncio.gather`. You do not flip a switch. When you want the
orchestration to be deterministic — the fan-out shape is fixed by you,
not by the model — you use `Plan` or the `Agent.parallel` sugar.

## Example

The next block runs four increasingly capable Agents against the same
call surface.  Each builds on the previous one: Tier 1 is the minimum
call, Tier 2 adds tools, Tier 3 adds structured output, Tier 4 nests
agents.  Nothing changes about how you *invoke* the agent; the
difference is entirely in how you *construct* it.

```python
from lazybridge import Agent
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    bullets: list[str]

# Tier 1: two lines — smallest viable call. Useful as a smoke test
# that credentials and provider routing work end-to-end.
print(Agent("claude-opus-4-7")("hello").text())

# Tier 2: with tools. Plain functions with type hints + docstring are
# auto-wrapped into Tool definitions; the model sees the docstring as
# the tool description and calls the function when helpful. Multiple
# tool calls in one turn execute concurrently via asyncio.gather.
def search(query: str) -> str:
    """Search the web for ``query`` and return the top 3 hits."""
    return "..."

print(Agent("claude-opus-4-7", tools=[search])("AI news April 2026").text())

# Tier 3: structured output. output=Summary switches the model into
# JSON mode with the Pydantic schema, validates the response, and
# exposes the parsed instance as Envelope.payload. Use .payload, not
# .text(), when you care about fields rather than prose.
resp = Agent("claude-opus-4-7", output=Summary)("summarise LazyBridge")
print(resp.payload.title, resp.payload.bullets)

# Tier 4: nested agent-of-agent. ``tools=[researcher]`` is sugar for
# ``tools=[researcher.as_tool()]``. The outer engine treats researcher
# identically to a function — same schema, same concurrency, same
# observability — and researcher's Session propagates down from
# editor's automatically.
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
editor     = Agent("claude-opus-4-7", tools=[researcher], name="editor")
print(editor("find papers and write a one-paragraph summary").text())
```

What you've seen: one call surface (`agent(task)`), one return type
(`Envelope`), and capabilities added purely via constructor kwargs.
The sections below extend this with reliability knobs (timeouts,
retries, caching), fallback routing, and the full kwarg table.

## Provider fallback routing

Pass `fallback=` to route automatically to a backup agent when the primary
engine returns an error (rate limit, network failure, model refusal):

```python
from lazybridge import Agent
from lazybridge.engines.llm import LLMEngine

# Primary: Anthropic. Fallback: OpenAI.
primary  = Agent(engine=LLMEngine("anthropic"), name="primary")
backup   = Agent(engine=LLMEngine("openai"),    name="backup")

agent = Agent(engine=LLMEngine("anthropic"), fallback=backup, name="router")

result = agent("Summarise the paper.")
# If Anthropic is unavailable, backup is tried automatically.
```

Fallback chains are arbitrarily deep — each fallback can itself have a
`fallback`. The last agent in the chain returns its error envelope if all
options fail.

```python
# Three-tier fallback: primary → secondary → tertiary
tertiary  = Agent(engine=LLMEngine("openai"),    name="tertiary")
secondary = Agent(engine=LLMEngine("anthropic"), fallback=tertiary, name="secondary")
primary   = Agent(engine=LLMEngine("google"),    fallback=secondary, name="primary")
```

## Reliability & performance kwargs

Production code usually needs more than the defaults: deadlines on hung
providers, automatic retries on transient 429/5xx, prompt caching to
cut repeat cost, and post-parse validation on structured output.  These
are plain kwargs on `Agent`; they're not opt-in capabilities behind a
flag.

```python
# What this shows: a single Agent configured for production.  The
# order matters — retry is the innermost loop (per-LLM-call), timeout
# is the outermost (wall-clock deadline on the whole run including
# tools).
# Why each kwarg: hung provider → timeout cancels; transient 429 →
# max_retries recovers without surfacing an error; identical system +
# tools on repeat calls → cache shaves ~90% off input-token cost on
# Anthropic; unsafe output → output_validator raises ValueError and
# triggers up to max_output_retries with the validation error injected
# as feedback.

from lazybridge import Agent, LLMEngine
from lazybridge.core.types import CacheConfig
from pydantic import BaseModel, field_validator

class Hits(BaseModel):
    items: list[str]

    @field_validator("items")
    @classmethod
    def _at_least_three(cls, v: list[str]) -> list[str]:
        if len(v) < 3:
            raise ValueError("must have at least 3 items")
        return v

def my_validator(hits: Hits) -> Hits:
    # Extra post-parse check; raising ValueError retriggers the LLM
    # with the error text appended to the task as feedback.
    if any(len(item) < 10 for item in hits.items):
        raise ValueError("each item must be at least 10 characters")
    return hits

research = Agent(
    # thinking= is NOT an Agent kwarg — pass via engine= when needed.
    engine=LLMEngine(
        "claude-opus-4-7",
        thinking=True,          # extended thinking on Anthropic / o-series.
        max_turns=15,            # tool-call rounds; default 10.
        request_timeout=60.0,    # per-LLM-call deadline.
    ),
    output=Hits,
    output_validator=my_validator,   # raises ValueError to retry
    max_output_retries=2,            # retries specifically for structured-output
                                     # validation failures (default 2)
    timeout=120.0,                   # deadline on the WHOLE agent run (tool loop
                                     # included); returns an error Envelope if hit
    cache=CacheConfig(ttl="1h"),     # 1-hour Anthropic cache (default is "5m"
                                     # when cache=True; cache=False disables)
    max_retries=3,                   # transient-error retries inside LLMEngine
                                     # (429 / 5xx / network). 0 disables.
    retry_delay=1.0,                 # base delay in seconds; exponential with
                                     # ±10% jitter.
    fallback=Agent("gpt-5"),         # on error envelope, run this agent from
                                     # scratch on the same task. Chainable.
)
env = research("find 3 papers about prompt caching")
```

After the call returns, `env.ok` distinguishes success from error,
`env.metadata` always carries token counts and latency, and
`env.metadata.nested_*` aggregates cost across any tool-calls inside
the run.  If both the primary and the fallback fail, the final error
Envelope surfaces to the caller — no exception is raised through.

## Shared config objects (`ResilienceConfig`, `ObservabilityConfig`)

Once you have more than one or two agents in a pipeline, repeating
the same seven resilience kwargs at every call site gets noisy and
drift-prone. The three config dataclasses — `ResilienceConfig`,
`ObservabilityConfig`, and the composite `AgentRuntimeConfig` — let
you define a policy once and inject it everywhere.

```python
# What this shows: a single production-grade resilience policy and a
# single Session shared across three agents. Without the config
# objects, each Agent() call would have to repeat timeout=,
# max_retries=, cache=, and session= by hand — six places to edit
# when the policy changes, six places to get it wrong.
# Why precedence matters: the ``fact_checker`` needs a tighter
# timeout than the rest of the fleet. Passing ``timeout=15.0``
# alongside ``resilience=policy`` overrides just that one field —
# retries, cache, and fallback still come from the shared policy.

from lazybridge import (
    Agent, Session, ResilienceConfig, ObservabilityConfig, CacheConfig,
)

# Fleet-wide policy: 60s wall-clock, 5 retries, 1h Anthropic cache.
policy = ResilienceConfig(
    timeout=60.0,
    max_retries=5,
    cache=CacheConfig(ttl="1h"),
)

# Fleet-wide observability: every agent reports to the same session.
trace = Session(console=True)
obs = ObservabilityConfig(session=trace)

researcher   = Agent("claude-opus-4-7", resilience=policy, observability=obs, name="researcher")
writer       = Agent("claude-opus-4-7", resilience=policy, observability=obs, name="writer")
# One-off override: tighter timeout for a fast-path agent — everything
# else (retries, cache, session) still inherits from the shared configs.
fact_checker = Agent("claude-opus-4-7", resilience=policy, observability=obs,
                     name="fact_checker", timeout=15.0)
```

**Precedence.** Flat kwargs always win over the config:

| Value passed via…           | Wins when…                                      |
| :-------------------------- | :---------------------------------------------- |
| `Agent(..., timeout=30.0)`  | always (explicit flat kwarg beats everything)   |
| `ResilienceConfig(timeout=120.0)` | no flat `timeout=` is passed              |
| documented default (`None`) | neither the flat kwarg nor the config supplied |

The precedence rule is implemented with a private sentinel, so even
passing the *documented default* explicitly counts as user intent:
`Agent(resilience=policy_with_retries_7, max_retries=3)` uses 3, not 7.

**`AgentRuntimeConfig` — one object for the whole fleet.** When you
pass both configs to every agent, wrap them in `AgentRuntimeConfig`
and pass a single object instead:

```python
from lazybridge import AgentRuntimeConfig

runtime = AgentRuntimeConfig(resilience=policy, observability=obs)

researcher = Agent("claude-opus-4-7", runtime=runtime, name="researcher")
writer     = Agent("claude-opus-4-7", runtime=runtime, name="writer")
```

**What's in / what's out.** The config objects cover the *runtime*
kwargs — resilience and observability. The *structural* kwargs
(`tools`, `output`, `memory`, `sources`, `guard`, `verify`, `engine`)
stay flat because they describe what the agent *is*, not how it
behaves under pressure. A shared resilience policy with a custom
tool list is idiomatic; sharing a full `AgentConfig` including tools
would collapse the distinction between agents.

## Three call surfaces: sync, async, streaming

One `Agent` exposes three ways to invoke it.  They return different
shapes but share the same engine pipeline (guard → engine → verify →
fallback); pick based on your host.

```python
# What this shows: the three entry points side-by-side. The sync
# __call__ detects an active event loop and hops to a worker thread
# so it's safe in Jupyter / FastAPI. run() is the async primitive
# that every other entry point ultimately dispatches to. stream()
# yields token chunks as they arrive for low-latency UI streaming.
# Why distinguish: inside a coroutine always prefer await
# agent.run(...), never agent(...). The auto-hop is there for
# convenience at REPL / notebook boundaries, not as the primary path.

import asyncio
from lazybridge import Agent

agent = Agent("claude-opus-4-7", name="writer")

# Sync — from scripts, Jupyter cells, or non-async test code.
env = agent("write a haiku about bees")
print(env.text())

# Async — from a coroutine. The primitive; all others delegate here.
async def use_it():
    env = await agent.run("write a haiku about bees")
    print(env.text())
asyncio.run(use_it())

# Streaming — async generator of text chunks. Useful when the UI
# wants to render tokens as they arrive. The final chunk is the
# complete payload; intermediate chunks are deltas.
async def stream_it():
    async for chunk in agent.stream("write a haiku about bees"):
        print(chunk, end="", flush=True)
asyncio.run(stream_it())
```

A per-chunk timeout is enforced when `timeout=` is set on the Agent
— the first chunk that takes too long raises `TimeoutError` and the
generator stops.  Non-streaming engines (HumanEngine, Plan) satisfy
the streaming protocol by yielding a single chunk with the final
text.

## `sources=` — live-view context injection

`sources=[...]` is LazyBridge's mechanism for injecting context that
is **computed fresh on every call**.  Each source object is asked for
its current text at call time (no snapshotting); the concatenated
text becomes `env.context`, which the engine prepends to the system
prompt when invoking the LLM.

```python
# What this shows: an agent whose system prompt is "What's in the
# blackboard?" and whose answer depends on the current Store state.
# Why live-view: if we snapshotted ``store`` at Agent() construction,
# the monitor would always report the Store contents at startup, not
# at call time. ``sources=[store]`` makes every call re-read it.
# What objects are valid sources: anything with a ``.text()`` method
# returning ``str``. Memory (live conversation), Store, or any user
# class. Plain strings also work — ``sources=["policy: peer-reviewed
# only"]`` injects them verbatim.

from lazybridge import Agent, LLMEngine, Store, Memory

store = Store(db="status.sqlite")
chat_memory = Memory(strategy="sliding", max_tokens=2000)

monitor = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You monitor the blackboard and answer questions about it.",
    ),
    sources=[
        store,              # store.to_text() each call
        chat_memory,        # chat_memory.text() each call (live history)
        "Today is 2026-04-22. Always cite the date when relevant.",
    ],
    name="monitor",
)

store.write("service_a", "healthy")
monitor("what's the state of service_a?")     # sees fresh store contents
store.write("service_a", "degraded")
monitor("what's the state of service_a?")     # sees the new value
```

Use `sources=` for live state (Store, Memory, a config file that can
be edited at runtime).  Use `LLMEngine(system=...)` for the static
system prompt.  Both combine at call time — system first, then
sources joined by `\n\n`, then the task.

## Pitfalls

- Passing `output=SomeModel` without tools and then calling `.text()`
  gives you the JSON dump of the payload, which is rarely what you want.
  Read `.payload` instead.
- `Agent.parallel` is sugar for deterministic fan-out returning
  `list[Envelope]`. It is **not** "a different kind of parallelism" —
  if you want the LLM to decide, put the candidates in `tools=[]`.
- `verify=` expects a judge that returns a verdict string starting
  with `"approved"` (case-insensitive) to accept. Anything else is
  treated as rejection + feedback.
- `fallback=` is tried on *any* error envelope, including errors caused
  by guard blocks or output validation failures — not just network errors.
  Make sure the fallback enforces the same invariants as the primary.
- `timeout=` is wall-clock; it includes tool execution time. A slow
  tool can exhaust the budget even when the LLM responds quickly.
- `cache=True` caches the **static prefix** (system prompt + tool
  definitions).  Adding one more tool or changing the system prompt
  evicts the cache.  OpenAI / DeepSeek auto-cache; Google needs a
  different API so `cache=` is a no-op there.
- `max_turns` on `LLMEngine` defaults to **20** (raised from 10 in the
  2026-04-23 amendments).  A task with many tool-call rounds can still
  hit it and return a `MaxTurnsExceeded` error.  Raise it when you
  expect long loops; lower it during dev to fail fast.
- `Agent("grok-2")` — an unknown model — falls back to Anthropic with
  a `UserWarning` by default.  In production, call
  `LLMEngine.set_default_provider(None)` once at startup to turn that
  into a loud `ValueError` at construction, so unknown-model bugs
  surface before any network round-trip.

!!! note "API reference"

    Agent(
        model_or_engine: str | Engine = "claude-opus-4-7",
        *,
        tools: list[Tool | Callable | Agent] = (),
        output: type = str,
        memory: Memory | None = None,
        sources: list = (),
        guard: Guard | None = None,
        verify: Agent | None = None,
        max_verify: int = 3,
        fallback: Agent | None = None,
        session: Session | None = None,
        verbose: bool = False,
        name: str | None = None,         # label used by Session.graph,
                                         # event logs, usage_summary()["by_agent"],
                                         # and SupervisorEngine's ``retry <name>:``
                                         # command. Defaults to the engine's model.
        description: str | None = None,  # shown to LLMs when this Agent is
                                         # wrapped as a tool (via as_tool() or
                                         # tools=[agent] in another Agent).
        model: str | None = None,     # tier alias when first arg is a provider name
        engine: Engine | None = None, # kwarg alias for the first positional
        # --- Config-object alternatives (see "Shared config objects") ---------
        # Each bundles a subset of the flat kwargs below.  Flat kwargs always
        # win; the config fills in anything omitted.  Use to share a policy
        # across a fleet of agents without copy-pasting seven kwargs.
        runtime: AgentRuntimeConfig | None = None,
        resilience: ResilienceConfig | None = None,
        observability: ObservabilityConfig | None = None,
        # --- Reliability / performance kwargs (see section above) -------------
        native_tools: list[NativeTool | str] | None = None,
        output_validator: Callable[[Any], Any] | None = None,  # raise ValueError to retry
        max_output_retries: int = 2,        # retries for output validation failures
        timeout: float | None = None,       # wall-clock deadline on the whole run
        max_retries: int = 3,               # LLMEngine transient-error retries
        retry_delay: float = 1.0,           # base seconds for exponential backoff
        cache: bool | CacheConfig = False,  # True → 5m Anthropic; CacheConfig(ttl="1h") for 1h
    ) -> Agent

    Sync:   agent(task) -> Envelope
    Async:  await agent.run(task) -> Envelope
    Stream: async for chunk in agent.stream(task): ...

    Factories:
      Agent.from_model(model: str, **kw) -> Agent       # explicit LLM
      Agent.from_engine(engine: Engine, **kw) -> Agent  # explicit Plan / Supervisor / custom
      Agent.from_provider(name: str, *, tier: str = "medium", **kw) -> Agent

    System prompts:
      # ``system=`` is NOT an Agent kwarg. Attach it to the engine:
      Agent(engine=LLMEngine("claude-opus-4-7", system="Be concise."))

    Composition sugar (NOT new paradigms):
      Agent.chain(*agents, **kw)    -> Agent           # sequential
      Agent.parallel(*agents, **kw) -> _ParallelAgent  # deterministic fan-out → list[Envelope]

!!! warning "Rules & invariants"

    - `tools=` accepts functions, Tool instances, Agent instances, and
      Agents-of-Agents. `wrap_tool` normalises everything at construction.
    - When a nested Agent has no `session=`, it inherits the caller's session
      and is registered on the graph with an `as_tool` edge. Observability
      flows through the whole tree.
    - When the engine emits multiple tool invocations in a single step, they
      execute concurrently via `asyncio.gather`. This is a capability, not a
      config knob; there is no serial mode.
    - `output=` defaulting to `str` means `Envelope.payload` is the model's
      text. Passing a Pydantic class sets up structured output and
      `Envelope.payload` becomes an instance of that class.
    - `verify=` wraps the run in a judge/retry loop (max `max_verify`
      attempts). The judge can be an Agent or a plain callable.
    - `guard=` filters both input and output. Blocked runs return an error
      Envelope without invoking the engine.
    - `fallback=` is invoked when `result.error is not None` after the primary
      engine runs. The fallback receives the same original task envelope.

## See also

[tool](tool.md), [envelope](envelope.md),
[chain](chain.md), [agent_parallel](agent-parallel.md),
[as_tool](as-tool.md), [session](session.md),
decision tree: [pick_tier](../decisions/pick-tier.md)
