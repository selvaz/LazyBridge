# LazyBridge — Complete API Quick Reference

## Model tiers (provider-relative)

| tier | `anthropic` / `claude` | `openai` / `chatgpt` / `gpt` | `google` / `gemini` | `deepseek` |
| --- | --- | --- | --- | --- |
| `top` | claude-opus-4-7 | gpt-5.4 | gemini-3.1-pro-preview *(same as expensive)* | deepseek-reasoner *(same as expensive)* |
| `expensive` | claude-opus-4-6 | gpt-5 | gemini-3.1-pro-preview *(same as top)* | deepseek-reasoner *(same as top)* |
| `medium` | claude-sonnet-4-6 | gpt-4o | gemini-3-flash-preview | deepseek-chat *(same as cheap, super_cheap)* |
| `cheap` | claude-haiku-4-5 | gpt-4o-mini | gemini-3.1-flash-lite-preview | deepseek-chat *(same as medium, super_cheap)* |
| `super_cheap` | claude-3-haiku | gpt-3.5-turbo | gemini-2.0-flash | deepseek-chat *(same as medium, cheap)* |

`LazyAgent(provider, model="<tier>")` resolves to the concrete model
above; literal model names pass through; unknown tier strings pass
through as literals.  Canonical matrix: `lazy_wiki/human/agents.md`.

## LazyAgent

```python
LazyAgent(
    provider: str | BaseProvider,          # "anthropic"|"claude"|"openai"|"gpt"|"google"|"gemini"|"deepseek"
    *,
    name: str | None = None,               # defaults to first 8 chars of uuid
    description: str | None = None,        # used when exposed as tool
    model: str | None = None,              # overrides provider default
    system: str | None = None,             # base system prompt
    context: LazyContext | Callable[[], str] | None = None,  # injected into system at execution time
    tools: list[LazyTool | ToolDefinition | dict] | None = None,  # agent-level tools
    native_tools: list[NativeTool | str] | None = None,         # provider-managed tools (web search, etc.)
    output_schema: type | dict | None = None,                   # agent-level structured output schema
    session: LazySession | None = None,    # enables tracking, store, graph
    verbose: bool = False,                 # print events to stdout in real-time (standalone agents)
    max_retries: int = 0,                  # retry on 429/5xx
    api_key: str | None = None,            # overrides env var
    **kwargs,                              # forwarded to provider constructor
)

LazyAgent.chat(
    messages: str | list[Message | dict],
    *,
    memory: Memory | None = None,          # stateful conversation — auto-accumulates history
    system: str | None = None,             # appended to agent-level system
    tools: list | None = None,             # merged with agent-level tools
    native_tools: list[NativeTool | str] | None = None,
    output_schema: type | dict | None = None,   # Pydantic model or JSON Schema dict
    thinking: bool | ThinkingConfig = False,
    skills: list[str] | None = None,       # Anthropic Skills: ["pdf", "excel", ...]
    stream: bool = False,                  # if True, returns Iterator[StreamChunk]
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    tool_choice: str | None = None,        # "auto"|"none"|"required"|"<tool_name>"|"parallel"
    context: LazyContext | Callable[[], str] | None = None,  # overrides agent-level context
    guard: Guard | None = None,            # runs input + output validation; raises GuardError on block
    **kwargs,
) -> CompletionResponse | Iterator[StreamChunk]

LazyAgent.achat(...)  # async version, same signature → CompletionResponse | AsyncIterator[StreamChunk]

LazyAgent.loop(
    messages: str | list,
    *,
    tools: list | None = None,
    native_tools: list[NativeTool | str] | None = None,
    max_steps: int = 8,                    # hard cap; raises ValueError if < 1
    tool_runner: Callable[[str, dict], Any] | None = None,  # fallback for tools not in registry
    on_event: Callable[[str, Any], None] | None = None,  # events: "step"|"tool_call"|"tool_result"|"done"|"verify_rejected"
    verify: Verifier | Callable[[str, str], str] | None = None,  # judge: any object with .text() or a callable
    max_verify: int = 3,                   # max retry attempts when verify is set
    tool_timeout: float | None = None,     # per-tool timeout in seconds; TimeoutError on breach
    guard: Guard | None = None,            # forwarded to each internal chat() call; guards input + output
    **chat_kwargs,                         # forwarded to chat() on each step
                                           # tool_choice="parallel" runs multiple tool calls concurrently
) -> CompletionResponse

LazyAgent.aloop(...)  # async version → CompletionResponse; tool_choice="parallel" uses asyncio.gather()

LazyAgent.chat_stream(messages, ...) -> Iterator[StreamChunk]     # dedicated streaming (preferred over chat(stream=True))
LazyAgent.achat_stream(messages, ...) -> AsyncIterator[StreamChunk]  # async streaming

LazyAgent.text(messages: str | list, **kwargs) -> str
LazyAgent.json(messages: str | list, schema: type | dict, **kwargs) -> Any
LazyAgent.atext(messages: str | list, **kwargs) -> str  # async
LazyAgent.ajson(messages: str | list, schema: type | dict, **kwargs) -> Any  # async

LazyAgent.as_tool(
    name: str | None = None,              # defaults to agent.name
    description: str | None = None,       # defaults to agent.description
    *,
    guidance: str | None = None,          # injected into calling agent's system prompt
    output_schema: type | dict | None = None,
    native_tools: list | None = None,
    system_prompt: str | None = None,     # override agent's system for this tool invocation
    tool_choice: str | None = None,      # "required"|"none"|"<name>" — forwarded to inner loop()
    strict: bool = False,
) -> LazyTool

LazyAgent.id: str           # UUID, set at construction
LazyAgent.name: str         # human-readable name
LazyAgent.description: str | None
LazyAgent.result: Any               # PUBLIC   — canonical last-value accessor: Pydantic if output_schema set, else str, else None. Always use this in pipeline code.
LazyAgent._last_output: str | None  # INTERNAL — plain text read by LazyContext.from_agent(). Do not use directly in user code; use agent.result instead.
LazyAgent._last_response: CompletionResponse | None  # ADVANCED — full response (.parsed, .usage, .tool_calls, .grounding_sources). Use when you need token counts or structured fields.

# GUI — available after `import lazybridge.gui`
LazyAgent.gui(*, open_browser: bool = True) -> str  # opens browser panel; returns URL
```

## HumanAgent

```python
HumanAgent(
    name: str = "human",
    *,
    description: str | None = None,
    input_fn: Callable[[str], str] | None = None,      # default: input()
    ainput_fn: Callable[[str], Awaitable[str]] | None = None,
    mode: Literal["single", "dialogue"] = "single",    # dialogue: multi-turn until end_token
    end_token: str = "done",                            # dialogue exit word
    prompt_template: str = "{task}",
    timeout: float | None = None,                       # seconds; None = forever
    default: str | None = None,                         # response on timeout
    session: LazySession | None = None,
)

# Same interface as LazyAgent — works in chains, parallel, as tools, as verify
HumanAgent.chat(messages, **kw) -> CompletionResponse
HumanAgent.achat(messages, **kw) -> CompletionResponse   # uses asyncio.to_thread
HumanAgent.text(messages, **kw) -> str
HumanAgent.as_tool(name, description, **kw) -> LazyTool
HumanAgent.result: str | None
```

## SupervisorAgent

```python
SupervisorAgent(
    name: str = "supervisor",
    *,
    description: str | None = None,
    input_fn: Callable[[str], str] | None = None,
    ainput_fn: Callable[[str], Awaitable[str]] | None = None,
    tools: list[LazyTool] | None = None,     # tools the human can call interactively
    agents: list | None = None,               # agents the human can retry with feedback
    session: LazySession | None = None,       # for store/events access
    timeout: float | None = None,
    default: str | None = None,
)

# REPL commands: continue | retry <agent>: <feedback> | store <key> | <tool>(<args>)
SupervisorAgent.chat(messages, **kw) -> CompletionResponse  # runs interactive REPL
SupervisorAgent.as_tool(name, description, **kw) -> LazyTool
```

## LazySession

```python
LazySession(
    *,
    db: str | None = None,                 # SQLite path; None = in-memory
    tracking: TrackLevel | str = TrackLevel.BASIC,  # OFF | BASIC | VERBOSE | FULL
    console: bool = False,                 # print events to stdout in real-time
    exporters: list | None = None,         # event exporters (CallbackExporter, …)
    redact: Callable[[str, dict], dict] | None = None,
                                           # mask sensitive tool args/results in
                                           # event payloads before store/export
)

LazySession.id: str
LazySession.store: LazyStore
LazySession.events: EventLog
LazySession.graph: GraphSchema

LazySession.gather(*coros: Awaitable) -> list[Any]  # FALLBACK — low-level async fan-out; prefer as_tool(mode="parallel") unless you need raw CompletionResponse per agent
LazySession.as_tool(                                 # CANONICAL — compose pipeline into a single tool
    name: str,
    description: str,
    *,
    mode: str | None = None,               # "parallel" | "chain" (required unless using entry_agent=)
    participants: list[LazyAgent | LazyTool] | None = None,  # explicit order; defaults to session agents
    combiner: str = "concat",              # parallel only: "concat" | "last"
    native_tools: list | None = None,      # forwarded to agents in chain mode
    entry_agent: LazyAgent | None = None,  # legacy single-agent delegation
    guidance: str | None = None,
    run_id: str | None = None,             # checkpoint isolation for concurrent runs
) -> LazyTool

LazySession.usage_summary() -> dict       # {"total": {...}, "by_agent": {...}} — requires tracking="verbose"
LazySession.add_exporter(exporter) -> None
LazySession.remove_exporter(exporter) -> None
LazySession.to_json() -> str
LazySession.from_json(text: str, **kwargs) -> LazySession  # classmethod
```

### Exporters

```python
CallbackExporter(fn: Callable)                     # wraps any callable
FilteredExporter(inner, event_types={"tool_call"})  # forwards only specified types
JsonFileExporter("events.jsonl")                    # appends JSON lines to file
StructuredLogExporter(logger_name="lazybridge.events")  # JSON via stdlib logging
OTelExporter(service_name="my-app", tracer=None)   # OpenTelemetry spans (pip install lazybridge[otel])
```

### Verifier Protocol

```python
from lazybridge import Verifier

class MyJudge:
    def text(self, messages: str) -> str:
        return "approved" if ok else "retry: fix X"

# Any object with .text() satisfies Verifier — including LazyAgent itself
agent.loop("task", verify=MyJudge())
```

### EventLog (`sess.events`)

```python
EventLog.log(event_type: str, *, agent_id, agent_name, **data) -> None
EventLog.get(*, agent_id=None, event_type=None, limit=200) -> list[dict]
EventLog.agent_log(agent_id: str, agent_name: str | None) -> _AgentLog
```

### TrackLevel (StrEnum)

```python
TrackLevel.OFF      # no events logged
TrackLevel.BASIC    # all events except MESSAGES, SYSTEM_CONTEXT, STREAM_CHUNK
TrackLevel.VERBOSE  # all events
TrackLevel.FULL     # synonym for VERBOSE
```

### Event (StrEnum)

```python
Event.MODEL_REQUEST | MODEL_RESPONSE | TOOL_CALL | TOOL_RESULT | TOOL_ERROR
Event.AGENT_START | AGENT_FINISH | LOOP_STEP
Event.MESSAGES | SYSTEM_CONTEXT | STREAM_CHUNK  # VERBOSE only
```

## LazyTool

```python
LazyTool.from_function(
    func: Callable,
    *,
    name: str | None = None,              # defaults to func.__name__
    description: str | None = None,       # defaults to first line of docstring
    guidance: str | None = None,          # injected into calling agent's system prompt
    schema_mode: ToolSchemaMode = ToolSchemaMode.SIGNATURE,  # SIGNATURE|LLM|HYBRID
    strict: bool = False,
    schema_builder: ToolSchemaBuilder | None = None,  # custom builder (caching, flatten_refs)
    schema_llm: Any | None = None,        # required for LLM/HYBRID modes
) -> LazyTool

LazyTool.compile(schema_llm: Any = None) -> LazyTool  # pre-freeze schema; returns self

LazyTool.from_agent(
    agent: LazyAgent,
    *,
    name: str | None = None,
    description: str | None = None,
    guidance: str | None = None,
    output_schema: type | dict | None = None,
    native_tools: list | None = None,
    system_prompt: str | None = None,
    tool_choice: str | None = None,       # "required"|"none"|"<name>" — forwarded to inner loop()
    strict: bool = False,
) -> LazyTool  # schema always {"task": str}

LazyTool.parallel(
    *participants: LazyAgent | LazyTool,  # at least one required
    name: str,
    description: str,
    combiner: str = "concat",             # "concat" | "last"
    native_tools: list | None = None,
    session: LazySession | None = None,   # validation-only; raises on cross-session conflict
    guidance: str | None = None,
    concurrency_limit: int | None = None, # max simultaneous participants; None = all at once
    step_timeout: float | None = None,    # per-participant timeout (seconds); None = no timeout
) -> LazyTool  # all participants run concurrently; cloned per invocation
               # original agent._last_output is None after run — use return value
               # timed-out participants appear as "[ERROR: TimeoutError: ...]" in concat output

LazyTool.chain(
    *participants: LazyAgent | LazyTool,  # at least one required
    name: str,
    description: str,
    native_tools: list | None = None,
    session: LazySession | None = None,   # validation-only; raises on cross-session conflict
    guidance: str | None = None,
    step_timeout: float | None = None,    # per-step timeout (seconds); asyncio.TimeoutError raised on breach
) -> LazyTool  # participants run sequentially; each receives previous output
               # async-under-the-hood: uses achat/aloop/ajson — never blocks the event loop
               # cloned per invocation — original agent._last_output is None after run

LazyTool.run(arguments: dict, parent: LazyAgent | None = None) -> Any
LazyTool.arun(arguments: dict, parent: LazyAgent | None = None) -> Any  # async

LazyTool.definition() -> ToolDefinition   # compiled JSON Schema ToolDefinition
LazyTool.specialize(
    name: str | None = None,
    description: str | None = None,
    guidance: str | None = None,
    schema_mode: ToolSchemaMode | None = None,
    strict: bool | None = None,
) -> LazyTool  # new LazyTool with overrides, same callable

# Persistence — save to .py file, reload across processes / machines
LazyTool.save(path: str) -> None          # generate human-readable .py file
LazyTool.load(path: str) -> LazyTool      # classmethod; only loads LazyBridge-generated files
# save() works for from_function() and from_agent() tools
# save() raises ValueError for parallel() / chain() tools (_is_pipeline_tool=True)
# load() requires sentinel header — rejects arbitrary .py files (security)
# NEVER expose LazyTool.load as an agent tool — it executes the file
```

### NormalizedToolSet

```python
NormalizedToolSet.from_list(tools: list[LazyTool | ToolDefinition | dict]) -> NormalizedToolSet
NormalizedToolSet.bridges: list[LazyTool]       # LazyTool items (have callables)
NormalizedToolSet.definitions: list[ToolDefinition]  # all items as ToolDefinition
NormalizedToolSet.registry: dict[str, LazyTool] # name → LazyTool lookup
```

## LazyContext

```python
LazyContext.from_text(text: str) -> LazyContext
LazyContext.from_function(fn: Callable[[], str]) -> LazyContext  # called at build() time
LazyContext.from_store(
    store: LazyStore,
    *,
    keys: list[str] | None = None,         # None = all keys
    prefix: str = "[store context]",
) -> LazyContext
LazyContext.from_agent(
    agent: LazyAgent,
    *,
    prefix: str | None = None,             # default: "[{agent.name} output]"
) -> LazyContext  # reads agent._last_output; "" if agent not yet run or returned ""
                  # both cases emit a DEBUG log to aid diagnosis

LazyContext.merge(*contexts: LazyContext) -> LazyContext  # classmethod
LazyContext.__add__(other) -> LazyContext  # ctx1 + ctx2 == merge(ctx1, ctx2)
LazyContext.build() -> str                 # materializes all sources, joined by \n\n
LazyContext.__call__() -> str              # alias for build()
```

## LazyStore

```python
LazyStore(db: str | None = None)           # None = in-memory; str = SQLite path

LazyStore.write(key: str, value: Any, *, agent_id: str | None = None) -> None
LazyStore.read(key: str, default: Any = None) -> Any
LazyStore.read_entry(key: str) -> StoreEntry | None  # includes agent_id, written_at
LazyStore.read_all() -> dict[str, Any]
LazyStore.read_by_agent(agent_id: str) -> dict[str, Any]
LazyStore.keys() -> list[str]
LazyStore.entries() -> list[StoreEntry]
LazyStore.delete(key: str) -> None
LazyStore.clear() -> None
LazyStore.to_text(keys: list[str] | None = None) -> str   # for LazyContext.from_store
LazyStore.__contains__(key) / __getitem__(key) / __setitem__(key, value)

# Async API — non-blocking wrappers for use inside async agent code
LazyStore.awrite(key: str, value: Any, *, agent_id: str | None = None) -> None
LazyStore.aread(key: str, default: Any = None) -> Any
LazyStore.aread_all() -> dict[str, Any]
LazyStore.akeys() -> list[str]

# StoreEntry fields: key, value, agent_id, written_at (datetime, UTC)
```

## LazyRouter

```python
LazyRouter(
    condition: Callable[[Any], str],       # returns a key from routes
    routes: dict[str, LazyAgent],
    name: str = "router",
    default: str | None = None,            # fallback key; raises KeyError if not set and key unknown
)
LazyRouter.route(value: Any) -> LazyAgent
LazyRouter.aroute(value: Any) -> LazyAgent  # async; awaits condition if coroutine
LazyRouter.agent_names -> list[str]
LazyRouter.to_graph_node() -> dict
```

## GraphSchema

```python
GraphSchema(session_id: str)  # auto-created by LazySession

GraphSchema.add_agent(agent: LazyAgent) -> None
GraphSchema.add_router(router: LazyRouter) -> None
GraphSchema.add_edge(src_id: str, dst_id: str, *, kind: EdgeType | str = EdgeType.TOOL, label: str | None = None) -> None
GraphSchema.to_dict() -> dict
GraphSchema.to_json() -> str
GraphSchema.to_yaml() -> str               # requires PyYAML
GraphSchema.save(path: str) -> None        # .json or .yaml extension
GraphSchema.from_dict(data: dict) -> GraphSchema   # classmethod
GraphSchema.from_json(text: str) -> GraphSchema    # classmethod
GraphSchema.from_file(path: str) -> GraphSchema    # classmethod

# NodeType (StrEnum): AGENT | ROUTER
# EdgeType (StrEnum): TOOL | CONTEXT | ROUTER
```

## Core response types

### CompletionResponse

```python
.content: str
.thinking: str | None
.tool_calls: list[ToolCall]
.stop_reason: str              # "end_turn" | "tool_use" | "max_tokens" | ...
.model: str | None
.usage: UsageStats             # .input_tokens, .output_tokens, .thinking_tokens, .cost_usd
.raw: Any                      # original provider response
.parsed: Any                   # populated when output_schema is set
.validated: bool | None        # None if no schema; True on success; False on failure
.validation_error: str | None
.grounding_sources: list[GroundingSource]  # .url, .title, .snippet
.web_search_queries: list[str]
.raise_if_failed() -> None     # raises StructuredOutputParseError (JSON) or StructuredOutputValidationError (schema)
```

### StreamChunk

```python
.delta: str                    # text delta
.thinking_delta: str
.tool_calls: list[ToolCall]
.stop_reason: str | None
.is_final: bool                # True on last chunk; parsed/usage/grounding populated here
.parsed: Any
.usage: UsageStats | None
```

### ToolCall

```python
.id: str
.name: str
.arguments: dict[str, Any]
```

### Memory

```python
Memory(
    *,
    strategy: Literal["full", "rolling", "auto"] = "auto",  # auto: compress when over budget
    max_context_tokens: int = 4000,     # token budget (len/4 estimate)
    window_turns: int = 10,             # recent turns kept raw
    compressor: LazyAgent | None = None, # LLM for semantic compression (optional)
)
Memory.from_history(messages, **kwargs) -> Memory  # restore with strategy
mem.history     # list[dict] — full raw history (never truncated)
mem.summary     # str | None — current compressed block
len(mem)        # total stored messages
mem.clear()     # reset history + compression
```

### NativeTool (StrEnum)

```python
NativeTool.WEB_SEARCH | CODE_EXECUTION | FILE_SEARCH | COMPUTER_USE | GOOGLE_SEARCH | GOOGLE_MAPS
```

### ThinkingConfig

```python
ThinkingConfig(enabled=True, effort="high", display=None, budget_tokens=None)
# effort: "low"|"medium"|"high"|"xhigh"
```

### StructuredOutputConfig (internal — use `output_schema=` shorthand)

```python
StructuredOutputConfig(schema, strict=True, max_retries=1, enable_fallback=True,
                       fallback_provider=None, fallback_model=None)
```

## ToolSchemaBuilder

```python
from lazybridge import ToolSchemaBuilder
from lazybridge.core.tool_schema import (
    InMemoryArtifactStore, ArtifactStore,
    ToolCompileArtifact, ToolSourceStatus,
)

ToolSchemaBuilder(
    artifact_store: ArtifactStore | None = None,  # caches compiled artifacts by fingerprint
    flatten_refs: bool = False,                   # inline $ref/$defs for providers without $ref support
)

ToolSchemaBuilder.build(
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
    mode: ToolSchemaMode = ToolSchemaMode.SIGNATURE,
    schema_llm: Any | None = None,
) -> ToolDefinition

ToolSchemaBuilder.build_artifact(func, ...) -> ToolCompileArtifact  # same args; returns full audit object

# ToolCompileArtifact fields (frozen dataclass):
.fingerprint: str              # 24-char SHA-256 hex of all compile inputs
.compiler_version: str
.prompt_version: str
.mode: ToolSchemaMode
.source_status: ToolSourceStatus   # BASELINE_ONLY | LLM_ENRICHED | LLM_INFERRED | FALLBACK_TO_BASELINE
.definition: ToolDefinition        # provider-ready schema
.baseline_definition: ToolDefinition | None   # SIGNATURE baseline (set for LLM/HYBRID modes)
.llm_enriched_fields: frozenset[str]          # param names whose descriptions the LLM generated
.warnings: tuple[str, ...]
.cache_hit: bool

# ArtifactStore protocol:
.get(fingerprint: str) -> ToolCompileArtifact | None
.put(artifact: ToolCompileArtifact) -> None

# InMemoryArtifactStore().__len__()  # number of cached entries
```

## Type annotation → JSON Schema (SIGNATURE mode)

| Python | JSON Schema |
|--------|-------------|
| `str` | `{"type":"string"}` |
| `int` | `{"type":"integer"}` |
| `float` | `{"type":"number"}` |
| `bool` | `{"type":"boolean"}` |
| `list[X]` | `{"type":"array","items":<X schema>}` |
| `set[X]` / `frozenset[X]` | `{"type":"array","uniqueItems":true,"items":<X>}` |
| `tuple[X,...]` | `{"type":"array","items":<X>}` |
| `tuple[X,Y,Z]` | `{"type":"array","prefixItems":[...],"items":false}` |
| `dict` | `{"type":"object"}` |
| `Optional[X]` / `X \| None` | `{"anyOf":[<X>,{"type":"null"}]}` |
| `X \| Y` | `{"anyOf":[<X>,<Y>]}` |
| `Literal["a","b"]` | `{"enum":["a","b"]}` |
| `MyEnum` (Enum subclass) | `{"enum":[e.value for e in MyEnum]}` |
| `MyModel` (Pydantic BaseModel) | `model_json_schema()` (may contain $defs/$ref) |
| `Annotated[X, "desc"]` | `{<X schema>, "description":"desc"}` |
| `Any` / unannotated | `{}` |

Parameters are **required** when no default; optional when `param=default`.
Pydantic coercion applied before function call; raises `ToolArgumentValidationError` on failure.
