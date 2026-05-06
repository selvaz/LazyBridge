# LazyBridge — API reference

Signature-first index of every public symbol. For usage and context, see the tier pages.


## Agent & tools

### `Agent(engine_or_model: 'str | Any' = 'claude-opus-4-7', tools: 'list[Tool | Callable | Agent] | None' = None, output: 'type' = <class 'str'>, memory: 'Any | None' = None, sources: 'list[Any] | None' = None, guard: 'Any | None' = None, verify: 'Agent | None' = None, max_verify: 'int' = 3, name: 'str | None' = _UNSET, description: 'str | None' = _UNSET, session: 'Any | None' = _UNSET, verbose: 'bool' = _UNSET, model: 'str | None' = None, engine: 'Any | None' = None, native_tools: 'list[Any] | None' = None, allow_dangerous_native_tools: 'bool' = False, runtime: 'AgentRuntimeConfig | None' = None, resilience: 'ResilienceConfig | None' = None, observability: 'ObservabilityConfig | None' = None, output_validator: 'Callable[[Any], Any] | None' = _UNSET, max_output_retries: 'int' = _UNSET, timeout: 'float | None' = _UNSET, max_retries: 'int' = _UNSET, retry_delay: 'float' = _UNSET, fallback: 'Agent | None' = _UNSET, cache: 'bool | Any' = _UNSET) -> 'None'`

Universal agent — ``Container(engine, tools, state)``.

### `Tool(func: 'Callable', *, name: 'str | None' = None, description: 'str | None' = None, mode: "Literal['signature', 'llm', 'hybrid']" = 'signature', schema_llm: 'Any | None' = None, strict: 'bool' = False, returns_envelope: 'bool' = False) -> 'None'`

Wraps any Python callable as an LLM-accessible tool.


## Envelope

### `Envelope(*, task: str | None = None, context: str | None = None, images: list[typing.Any] | None = None, audio: typing.Any | None = None, payload: Optional[~T] = None, metadata: lazybridge.envelope.EnvelopeMetadata = <factory>, error: lazybridge.envelope.ErrorInfo | None = None) -> None`

Typed envelope carrying a payload of type ``T``.

### `from_prev`

Use the Envelope produced by the previous step (default).

### `from_start`

Use the initial task/context Envelope passed to the Plan.

### `from_step(name: 'str') -> '_FromStep'`

*(no docstring)*

### `from_parallel(name: 'str') -> '_FromParallel'`

*(no docstring)*


## Engines

### `LLMEngine(model: 'str', *, provider: 'str | None' = None, thinking: 'bool' = False, max_turns: 'int' = 20, tool_choice: "Literal['auto', 'any']" = 'auto', temperature: 'float | None' = None, system: 'str | None' = None, native_tools: 'list[NativeTool | str] | None' = None, allow_dangerous_native_tools: 'bool' = False, max_retries: 'int' = 3, retry_delay: 'float' = 1.0, request_timeout: 'float | None' = 120.0, max_parallel_tools: 'int | None' = 8, tool_timeout: 'float | None' = None, stream_idle_timeout: 'float | None' = _UNSET, stream_buffer: 'int' = 64, cache: 'bool | Any' = False, strict_multimodal: 'bool' = False) -> 'None'`

Drives the LLM ↔ tool-call loop for a single agent invocation.

### `Plan(*steps: 'Step', max_iterations: 'int' = 100, store: 'Store | None' = None, checkpoint_key: 'str | None' = None, resume: 'bool' = False, on_concurrent: "Literal['fail', 'fork']" = 'fail') -> 'None'`

Structured multi-step execution engine.

### `Step(target: 'Any', task: 'Sentinel | str' = <factory>, context: 'Sentinel | str | list[Sentinel | str] | None' = None, sources: 'list[Any]' = <factory>, writes: 'str | None' = None, input: 'type' = typing.Any, output: 'type' = <class 'str'>, parallel: 'bool' = False, name: 'str | None' = None, routes: 'dict[str, Callable[[Any], bool]] | None' = None, routes_by: 'str | None' = None, after_branches: 'str | None' = None) -> None`

A single node in a Plan.

### `PlanCompileError`

Common base class for all non-exit exceptions.

### `ToolTimeoutError`

Raised when a tool exceeds ``LLMEngine.tool_timeout``.

### `StreamStallError`

Raised when a streaming response goes idle past ``stream_idle_timeout``.


## Memory / Store / Session

### `Memory(*, strategy: "Literal['auto', 'sliding', 'summary', 'none']" = 'auto', max_tokens: 'int | None' = 4000, max_turns: 'int | None' = 1000, store: 'Any | None' = None, summarizer: 'Any | None' = None, summarizer_timeout: 'float | None' = 30.0) -> 'None'`

Conversation memory with configurable compression strategy.

### `Store(db: 'str | None' = None) -> 'None'`

Key-value store for PlanState and shared data.

### `Session(*, db: 'str | None' = None, exporters: 'list[Any] | None' = None, redact: 'Callable[[dict[str, Any]], dict[str, Any]] | None' = None, redact_on_error: "Literal['fallback', 'strict']" = 'strict', console: 'bool' = False, batched: 'bool' = False, batch_size: 'int' = 100, batch_interval: 'float' = 1.0, max_queue_size: 'int' = 10000, on_full: "Literal['drop', 'block', 'hybrid']" = 'hybrid', critical_events: 'frozenset[str] | set[str] | None' = None) -> 'None'`

Container for observability config: exporters, redaction, EventLog.

### `EventLog(session_id: 'str', db: 'str | None' = None, *, batched: 'bool' = False, batch_size: 'int' = 100, batch_interval: 'float' = 1.0, max_queue_size: 'int' = 10000, on_full: "Literal['drop', 'block', 'hybrid']" = 'hybrid', critical_events: 'frozenset[str] | set[str] | None' = None) -> 'None'`

SQLite-backed event log. Thread-safe via thread-local connections.

### `EventType(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)`

Enum where members are also (and must be) strings


## Guards

### `Guard()`

Base guard. Override check_input and/or check_output.

### `GuardAction(allowed: 'bool' = True, message: 'str | None' = None, modified_text: 'str | None' = None, metadata: 'dict[str, Any]' = <factory>) -> None`

GuardAction(allowed: 'bool' = True, message: 'str | None' = None, modified_text: 'str | None' = None, metadata: 'dict[str, Any]' = <factory>)

### `ContentGuard(input_fn: 'Callable[[str], GuardAction] | None' = None, output_fn: 'Callable[[str], GuardAction] | None' = None) -> 'None'`

Function-based guard.

### `GuardChain(*guards: 'Guard') -> 'None'`

Run multiple guards in sequence; first block wins.

### `LLMGuard(agent: 'Any', policy: 'str' = 'block harmful content', *, timeout: 'float | None' = 60.0) -> 'None'`

Use an Agent as a judge. Returns block if the verdict begins with 'block' or 'deny'.


## Exporters

### `CallbackExporter(*, fn: 'Callable[[dict[str, Any]], None]') -> 'None'`

Forward every event to a user-supplied callable.

### `ConsoleExporter(*, stream: 'Any' = None) -> 'None'`

Pretty-print events to stdout for human inspection.

### `FilteredExporter(*, inner: 'Any', event_types: 'set[str]') -> 'None'`

Forward only events whose type is in ``event_types`` to ``inner``.

### `JsonFileExporter(*, path: 'str') -> 'None'`

Append each event as a JSON line to ``path``.

### `StructuredLogExporter(*, logger_name: 'str' = 'lazybridge') -> 'None'`

Emit each event via Python's ``logging`` module.


## Graph

### `GraphSchema(session_id: 'str' = '') -> 'None'`

Directed graph of agents, routers, and their connections.


## Core types

### `from_parallel_all(name: 'str') -> '_FromParallelAll'`

Aggregate every consecutive parallel sibling starting at ``name``.

### `ToolProvider(*args, **kwargs)`

A ``tools=[...]`` entry that expands itself into one or more Tools.

### `NativeTool(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)`

Provider-native server-side tools (run on provider infrastructure).

### `when`

Entry point for the predicate DSL.

### `GuardError`

Raised when a Guard blocks execution.

### `EventExporter(*args, **kwargs)`

Protocol satisfied by all exporter classes.

### `BaseProvider(api_key: 'str | None' = None, model: 'str | None' = None, *, strict_native_tools: 'bool | None' = None, **kwargs)`

Stable abstract base class for all LLM providers.

### `AgentRuntimeConfig(resilience: 'ResilienceConfig | None' = None, observability: 'ObservabilityConfig | None' = None) -> None`

Composite — carries both resilience and observability.

### `CacheConfig(enabled: 'bool' = True, ttl: 'str' = '5m') -> None`

Mark the static prefix of a request (system prompt + tool

### `ObservabilityConfig(verbose: 'bool' = False, session: 'Any' = None, name: 'str | None' = None, description: 'str | None' = None) -> None`

Bundle of identity / tracing knobs shareable across Agents.

### `ResilienceConfig(timeout: 'float | None' = None, max_retries: 'int' = 3, retry_delay: 'float' = 1.0, cache: 'bool | CacheConfig' = False, max_output_retries: 'int' = 2, output_validator: 'Any' = None, fallback: 'Any' = None) -> None`

Bundle of reliability / performance knobs shareable across Agents.

### `MockAgent(responses: 'Any', *, name: 'str' = 'mock_agent', description: 'str | None' = None, output: 'type' = <class 'str'>, cycle: 'bool' = False, delay_ms: 'float' = 0.0, default_input_tokens: 'int' = 10, default_output_tokens: 'int' = 20, default_cost_usd: 'float' = 0.0, default_latency_ms: 'float | None' = None, default_model: 'str' = 'mock', default_provider: 'str' = 'mock') -> 'None'`

Deterministic test double that quacks like :class:`lazybridge.Agent`.
