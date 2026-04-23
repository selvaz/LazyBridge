# LazyBridge — API reference

Signature-first index of every public symbol. For usage and context, see the tier pages.


## Agent & tools

### `Agent(engine_or_model: 'str | Any' = 'claude-opus-4-7', tools: 'list[Tool | Callable | Agent] | None' = None, output: 'type' = <class 'str'>, memory: 'Any | None' = None, sources: 'list[Any] | None' = None, guard: 'Any | None' = None, verify: 'Agent | None' = None, max_verify: 'int' = 3, name: 'str | None' = <object object at 0x7fd2b3f24f30>, description: 'str | None' = <object object at 0x7fd2b3f24f30>, session: 'Any | None' = <object object at 0x7fd2b3f24f30>, verbose: 'bool' = <object object at 0x7fd2b3f24f30>, model: 'str | None' = None, engine: 'Any | None' = None, native_tools: 'list[Any] | None' = None, runtime: 'AgentRuntimeConfig | None' = None, resilience: 'ResilienceConfig | None' = None, observability: 'ObservabilityConfig | None' = None, output_validator: 'Callable[[Any], Any] | None' = <object object at 0x7fd2b3f24f30>, max_output_retries: 'int' = <object object at 0x7fd2b3f24f30>, timeout: 'float | None' = <object object at 0x7fd2b3f24f30>, max_retries: 'int' = <object object at 0x7fd2b3f24f30>, retry_delay: 'float' = <object object at 0x7fd2b3f24f30>, fallback: "'Agent | None'" = <object object at 0x7fd2b3f24f30>, cache: 'bool | Any' = <object object at 0x7fd2b3f24f30>) -> 'None'`

Universal agent — delegates execution to a swappable Engine.

### `Tool(func: 'Callable', *, name: 'str | None' = None, description: 'str | None' = None, guidance: 'str | None' = None, mode: "Literal['signature', 'llm', 'hybrid']" = 'signature', schema_llm: 'Any | None' = None, strict: 'bool' = False, returns_envelope: 'bool' = False) -> 'None'`

Wraps any Python callable as an LLM-accessible tool.


## Envelope

### `Envelope(*, task: str | None = None, context: str | None = None, payload: Optional[~T] = None, metadata: lazybridge.envelope.EnvelopeMetadata = <factory>, error: lazybridge.envelope.ErrorInfo | None = None) -> None`

Typed envelope carrying a payload of type ``T``.

### `EnvelopeMetadata(*, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0, latency_ms: float = 0.0, model: str | None = None, provider: str | None = None, run_id: str | None = None, nested_input_tokens: int = 0, nested_output_tokens: int = 0, nested_cost_usd: float = 0.0) -> None`

!!! abstract "Usage Documentation"

### `ErrorInfo(*, type: str, message: str, retryable: bool = False) -> None`

!!! abstract "Usage Documentation"

### `from_prev`

Use the Envelope produced by the previous step (default).

### `from_start`

Use the initial task/context Envelope passed to the Plan.

### `from_step(name: 'str') -> '_FromStep'`

*(no docstring)*

### `from_parallel(name: 'str') -> '_FromParallel'`

*(no docstring)*


## Engines

### `Engine(*args, **kwargs)`

Contract every engine must satisfy.

### `LLMEngine(model: 'str', *, provider: 'str | None' = None, thinking: 'bool' = False, max_turns: 'int' = 20, tool_choice: "Literal['auto', 'any']" = 'auto', temperature: 'float | None' = None, system: 'str | None' = None, native_tools: 'list[NativeTool | str] | None' = None, max_retries: 'int' = 3, retry_delay: 'float' = 1.0, request_timeout: 'float | None' = 120.0, cache: 'bool | Any' = False) -> 'None'`

Drives the LLM ↔ tool-call loop for a single agent invocation.

### `HumanEngine(*, timeout: 'float | None' = None, ui: "Literal['terminal', 'web'] | _UIProtocol" = 'terminal', default: 'str | None' = None) -> 'None'`

Presents the task to a human and returns their response as an Envelope.

### `SupervisorEngine(*, tools: 'list[Tool | Callable | Any] | None' = None, agents: 'list[Any] | None' = None, store: 'Store | None' = None, input_fn: 'Callable[[str], str] | None' = None, ainput_fn: 'Callable[[str], Awaitable[str]] | None' = None, timeout: 'float | None' = None, default: 'str | None' = None) -> 'None'`

Human-in-the-loop engine with tool-calling and agent retry.

### `Plan(*steps: 'Step', max_iterations: 'int' = 100, store: 'Store | None' = None, checkpoint_key: 'str | None' = None, resume: 'bool' = False, on_concurrent: "Literal['fail', 'fork']" = 'fail') -> 'None'`

Structured multi-step execution engine.

### `Step(target: 'Any', task: 'Sentinel | str' = <factory>, context: 'Sentinel | str | None' = None, sources: 'list[Any]' = <factory>, writes: 'str | None' = None, input: 'type' = typing.Any, output: 'type' = <class 'str'>, parallel: 'bool' = False, name: 'str | None' = None) -> None`

A single node in a Plan.

### `PlanState(plan_id: 'str', current_step: 'str', next_step: 'str | None', store: 'dict[str, Any]', history: 'list[StepResult]', status: "Literal['running', 'paused', 'done', 'failed']") -> None`

PlanState(plan_id: 'str', current_step: 'str', next_step: 'str | None', store: 'dict[str, Any]', history: 'list[StepResult]', status: "Literal['running', 'paused', 'done', 'failed']")

### `StepResult(step_name: 'str', envelope: 'Envelope', ts: 'float' = <factory>) -> None`

StepResult(step_name: 'str', envelope: 'Envelope', ts: 'float' = <factory>)

### `PlanCompileError`

Common base class for all non-exit exceptions.


## Memory / Store / Session

### `Memory(*, strategy: "Literal['auto', 'sliding', 'summary', 'none']" = 'auto', max_tokens: 'int | None' = 4000, max_turns: 'int | None' = 1000, store: 'Any | None' = None, summarizer: 'Any | None' = None) -> 'None'`

Conversation memory with configurable compression strategy.

### `Store(db: 'str | None' = None) -> 'None'`

Key-value store for PlanState and shared data.

### `StoreEntry(key: 'str', value: 'Any', written_at: 'float' = <factory>, agent_id: 'str | None' = None) -> None`

StoreEntry(key: 'str', value: 'Any', written_at: 'float' = <factory>, agent_id: 'str | None' = None)

### `Session(*, db: 'str | None' = None, exporters: 'list[Any] | None' = None, redact: 'Callable[[dict[str, Any]], dict[str, Any]] | None' = None, redact_on_error: "Literal['fallback', 'strict']" = 'fallback', console: 'bool' = False) -> 'None'`

Container for observability config: exporters, redaction, EventLog.

### `EventLog(session_id: 'str', db: 'str | None' = None) -> 'None'`

SQLite-backed event log. Thread-safe via thread-local connections.

### `EventType(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)`

Enum where members are also (and must be) strings


## Guards & evals

### `Guard()`

Base guard. Override check_input and/or check_output.

### `GuardAction(allowed: 'bool' = True, message: 'str | None' = None, modified_text: 'str | None' = None, metadata: 'dict[str, Any]' = <factory>) -> None`

GuardAction(allowed: 'bool' = True, message: 'str | None' = None, modified_text: 'str | None' = None, metadata: 'dict[str, Any]' = <factory>)

### `ContentGuard(input_fn: 'Callable[[str], GuardAction] | None' = None, output_fn: 'Callable[[str], GuardAction] | None' = None) -> 'None'`

Function-based guard.

### `GuardChain(*guards: 'Guard') -> 'None'`

Run multiple guards in sequence; first block wins.

### `LLMGuard(agent: 'Any', policy: 'str' = 'block harmful content') -> 'None'`

Use an Agent as a judge. Returns block if the verdict begins with 'block' or 'deny'.

### `EvalCase(input: 'str', check: 'Callable[..., bool]', expected: 'Any' = None, description: 'str' = '') -> None`

EvalCase(input: 'str', check: 'Callable[..., bool]', expected: 'Any' = None, description: 'str' = '')

### `EvalReport(results: 'list[EvalResult]' = <factory>) -> None`

EvalReport(results: 'list[EvalResult]' = <factory>)

### `EvalSuite(*cases: 'EvalCase') -> 'None'`

Run a set of EvalCases against any agent callable.

### `exact_match(expected: 'str') -> 'Callable[[str, str], bool]'`

*(no docstring)*

### `contains(substring: 'str') -> 'Callable[[str], bool]'`

*(no docstring)*

### `llm_judge(agent: 'Any', criteria: 'str') -> 'Callable[[str], bool]'`

Returns a judge function using an agent to evaluate output.


## Exporters

### `CallbackExporter(fn: 'Callable[[dict[str, Any]], None]') -> 'None'`

Forward every event to a user-supplied callable.

### `ConsoleExporter(*, stream: 'Any' = None) -> 'None'`

Pretty-print events to stdout for human inspection.

### `FilteredExporter(inner: 'Any', *, event_types: 'set[str]') -> 'None'`

Forward only events whose type is in ``event_types`` to ``inner``.

### `JsonFileExporter(path: 'str') -> 'None'`

Append each event as a JSON line to ``path``.

### `OTelExporter(endpoint: 'str | None' = None, *, exporter: 'Any | None' = None) -> 'None'`

Export events as OpenTelemetry spans (requires opentelemetry-sdk).

### `StructuredLogExporter(logger_name: 'str' = 'lazybridge') -> 'None'`

Emit each event via Python's ``logging`` module.


## Graph

### `GraphSchema(session_id: 'str' = '') -> 'None'`

Directed graph of agents, routers, and their connections.

### `NodeType(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)`

Enum where members are also (and must be) strings

### `EdgeType(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)`

Enum where members are also (and must be) strings


## Core types

### `GuardError`

Raised when a Guard blocks execution.

### `not_contains(substring: 'str') -> 'Callable[[str], bool]'`

*(no docstring)*

### `max_length(n: 'int') -> 'Callable[[str], bool]'`

*(no docstring)*

### `min_length(n: 'int') -> 'Callable[[str], bool]'`

*(no docstring)*

### `EventExporter(*args, **kwargs)`

Protocol satisfied by all exporter classes.

### `BaseProvider(api_key: 'str | None' = None, model: 'str | None' = None, **kwargs)`

Stable abstract base class for all LLM providers.

### `CompletionRequest(messages: 'list[Message]', model: 'str | None' = None, system: 'str | None' = None, max_tokens: 'int' = 4096, temperature: 'float | None' = None, tools: 'list[ToolDefinition]' = <factory>, tool_choice: 'str | None' = None, native_tools: 'list[NativeTool]' = <factory>, structured_output: 'StructuredOutputConfig | None' = None, thinking: 'ThinkingConfig | None' = None, skills: 'SkillsConfig | None' = None, cache: 'CacheConfig | None' = None, stream: 'bool' = False, extra: 'dict[str, Any]' = <factory>) -> None`

Unified request object passed to any provider.

### `CompletionResponse(content: 'str', thinking: 'str | None' = None, tool_calls: 'list[ToolCall]' = <factory>, stop_reason: 'str' = 'end_turn', model: 'str | None' = None, usage: 'UsageStats' = <factory>, raw: 'Any' = None, parsed: 'Any' = None, validation_error: 'str | None' = None, validated: 'bool | None' = None, grounding_sources: 'list[GroundingSource]' = <factory>, web_search_queries: 'list[str]' = <factory>, search_entry_point: 'str | None' = None, verify_log: 'list[str]' = <factory>) -> None`

Unified response from any provider.

### `Message(role: 'Role', content: 'str | list[ContentBlock]') -> None`

Message(role: 'Role', content: 'str | list[ContentBlock]')

### `NativeTool(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)`

Provider-native server-side tools (run on provider infrastructure).

### `Role(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)`

Enum where members are also (and must be) strings

### `StreamChunk(delta: 'str' = '', thinking_delta: 'str' = '', tool_calls: 'list[ToolCall]' = <factory>, stop_reason: 'str | None' = None, usage: 'UsageStats | None' = None, is_final: 'bool' = False, parsed: 'Any' = None, validation_error: 'str | None' = None, validated: 'bool | None' = None, grounding_sources: 'list[GroundingSource]' = <factory>, web_search_queries: 'list[str]' = <factory>, search_entry_point: 'str | None' = None) -> None`

A single chunk from a streaming response.

### `StructuredOutputConfig(schema: 'type | dict[str, Any]', strict: 'bool' = True) -> None`

Config for constrained JSON output.

### `ThinkingConfig(enabled: 'bool' = True, display: 'str | None' = None, effort: 'str' = 'high', budget_tokens: 'int | None' = None) -> None`

Reasoning/thinking configuration.

### `ToolCall(id: 'str', name: 'str', arguments: 'dict[str, Any]', thought_signature: 'Any' = None) -> None`

ToolCall(id: 'str', name: 'str', arguments: 'dict[str, Any]', thought_signature: 'Any' = None)

### `ToolDefinition(name: 'str', description: 'str', parameters: 'dict[str, Any]', strict: 'bool' = False) -> None`

Unified tool/function definition (JSON Schema based).

### `UsageStats(input_tokens: 'int' = 0, output_tokens: 'int' = 0, thinking_tokens: 'int' = 0, cost_usd: 'float | None' = None) -> None`

UsageStats(input_tokens: 'int' = 0, output_tokens: 'int' = 0, thinking_tokens: 'int' = 0, cost_usd: 'float | None' = None)

### `AgentRuntimeConfig(resilience: 'ResilienceConfig | None' = None, observability: 'ObservabilityConfig | None' = None) -> None`

Composite — carries both resilience and observability.

### `CacheConfig(enabled: 'bool' = True, ttl: 'str' = '5m') -> None`

Mark the static prefix of a request (system prompt + tool

### `ObservabilityConfig(verbose: 'bool' = False, session: 'Any' = None, name: 'str | None' = None, description: 'str | None' = None) -> None`

Bundle of identity / tracing knobs shareable across Agents.

### `ResilienceConfig(timeout: 'float | None' = None, max_retries: 'int' = 3, retry_delay: 'float' = 1.0, cache: 'bool | CacheConfig' = False, max_output_retries: 'int' = 2, output_validator: 'Any' = None, fallback: 'Any' = None) -> None`

Bundle of reliability / performance knobs shareable across Agents.
