# Errors

Two ways LazyBridge surfaces failures:

1. **Error envelopes.** When an agent run fails *but the framework
   catches the exception*, the agent returns an `Envelope` with
   `result.ok == False` and `result.error` populated. The agent
   call itself does not raise — read `result.error.type` /
   `result.error.message`.
2. **Raised exceptions.** When the failure is structural (broken
   plan DAG, concurrent-run collision, schema-build issue), the
   framework raises a typed exception you catch with `try/except`.

The convention is "raise on construction, return error on
runtime" — but there are exceptions (see `PlanCompileError` notes
below).

## Error envelopes (`result.error.type` values)

These are the `type` strings you see when checking `result.ok` is
`False`. The agent call itself never raises.

| `type=` | Cause | Diagnosis | Fix |
|---|---|---|---|
| `MaxTurnsExceeded` | `LLMEngine` ran the tool-calling loop for `max_turns` rounds without producing a final answer | Likely an infinite tool-call loop (the model keeps re-asking for the same tool). Inspect the session events for the last few `TOOL_CALL` payloads | Bump `LLMEngine(max_turns=N)` for genuinely long tasks; tighten the system prompt; or add a `verify=` judge that rejects non-final responses |
| `MaxIterationsExceeded` | `Plan` exceeded `max_iterations` while routing | Routing cycle (`A → B → A` self-correction loop with no termination predicate) or under-sized cap | Lower the cap during dev to fail fast; add a counter via `writes=` and a predicate that breaks the loop; raise `Plan(max_iterations=N)` when the loop is legitimate |
| `GuardBlocked` | A `Guard` returned `allowed=False` on input or output | `result.error.message` carries the guard's verdict text | Either fix the input/output to satisfy the guard, or relax the guard if it's over-broad |
| `ToolArgumentParseError` | The LLM emitted tool arguments that don't match the tool's JSON schema | Provider-side strict mode rejected the call. Inspect the tool's `definition().parameters` and the model's emitted arguments in the session event | Loosen the tool's `strict=` flag; clarify the docstring / type hints so the model emits valid args; or add an LLM-fixed retry |
| `TimeoutError` | `Agent(timeout=N)` deadline expired | The whole run exceeded the budget | Raise the timeout, or cap individual tool calls with `LLMEngine(tool_timeout=N)` |

## Raised exceptions (`try/except` surface)

These propagate as Python exceptions. Catch them when constructing
plans, registering providers, or hitting strict-mode features.

| Exception | Raised when | Where | Fix |
|---|---|---|---|
| `PlanCompileError` | DAG validation fails: duplicate step names, unknown `routes=` targets, malformed `routes_by` Literal type, dangling `from_step` / `from_parallel_all` references, mid-band parallel target, `after_branches` referencing a step that doesn't come after | At `Agent(engine=Plan(...), tools=[...])` construction | Fix the offending step. The error message names both the offending step and the violation. **Also raised at runtime** when a `routes={}` predicate raises an exception ([known issue #20](https://github.com/selvaz/LazyBridge/issues/20)) — wrap risky predicates with their own try/except returning `False` |
| `ConcurrentPlanRunError` (subclasses `RuntimeError`) | Two `Plan` runs share a `checkpoint_key` with `on_concurrent="fail"` (default) | Runtime CAS collision in `_save_checkpoint` / `_claim_checkpoint` | Use a unique `checkpoint_key` per concurrent run, or `on_concurrent="fork"` for fan-out workflows (incompatible with `resume=True`) |
| `ToolTimeoutError` | A tool exceeded `LLMEngine(tool_timeout=N)` | Runtime, inside `LLMEngine` tool dispatch | The engine catches this internally and reports it to the model as `[TOOL_TIMEOUT] …` in the tool result, so the model can recover; the agent run does not abort. Catch only if you wrap the engine yourself |
| `StreamStallError` | A streaming response went idle longer than `LLMEngine(stream_idle_timeout=N, default 90s)` | Runtime, during `agent.stream(...)` or the engine's stream consumer | Pair with `agent.run(...)` instead for non-interactive use; bump `stream_idle_timeout` only if you trust the upstream provider (passing `None` disables it — emits a one-shot `UserWarning`) |
| `GuardError` | Some `Guard` integrations raise this for hard policy failures | Runtime | Catch and surface to the user; or replace the guard with one that returns `GuardAction(allowed=False, message=...)` for graceful rejection |
| `UnsupportedNativeToolError` (subclasses `ValueError`) | The provider doesn't implement a requested `NativeTool` AND `strict_native_tools=True` | At provider time, when the request includes the unsupported tool | Either remove the native tool from `native_tools=[...]`, switch to a provider that supports it (the message lists supported alternatives), or accept the warning-and-drop default by leaving `strict_native_tools=False` |
| `UnsupportedFeatureError` (subclasses `ValueError`) | The model doesn't support the multimodal modality the request includes (vision / audio) AND `strict_multimodal=True` | At provider time | Drop the attachment, switch to a multimodal model, or accept the warning-and-drop default |
| `ToolArgumentValidationError` (subclasses `ValueError`) | A tool's args fail the auto-generated Pydantic model's validation | At tool dispatch | Loosen the type hint on the tool function, fix the model's emitted args via prompt engineering, or pass `strict=False` on the `Tool` |
| `ToolSchemaBuildError` (subclasses `RuntimeError`) | The schema builder couldn't infer a JSON schema for the tool function | At `Tool(...)` construction or first `Agent(tools=[...])` use | Add type hints to all parameters; switch to `mode="llm"` or `mode="hybrid"` for legacy callables; or pass a pre-built schema via `Tool.from_schema(...)` |
| `StructuredOutputError` / `StructuredOutputParseError` / `StructuredOutputValidationError` | The LLM produced output that failed `output=PydanticModel` validation, exhausting `max_output_retries` | Runtime, after retries | Tighten the system prompt; relax the model (less strict types or `Optional` fields); raise `Agent(max_output_retries=N)`; or accept errors via `Envelope.error` checking |

## Common diagnosis flow

1. **`result.ok` is `False`?** Read `result.error.type` and
   `result.error.message`. The type maps to one row in the
   first table above.
2. **Got an exception, not an envelope?** It's one of the rows
   in the second table. Check the message — every framework
   exception names the offending step / tool / model in the
   message body.
3. **Hit a `MaxIterationsExceeded` or `MaxTurnsExceeded`?** Pull
   the session's tool-call events for the last few rounds —
   `session.events.query(event_type=EventType.TOOL_CALL)` (or
   filter by run id). Loops nearly always reveal themselves as
   the same tool name calling repeatedly.
4. **Hit a `PlanCompileError`?** The error message names the
   offending step. If the message says "predicate raised …",
   you've hit the runtime branch of issue #20 — check the
   predicate code, not the DAG shape.
5. **Hit a `ConcurrentPlanRunError`?** Your `checkpoint_key` is
   shared between two runs. Either pass a unique key per run, or
   switch to `on_concurrent="fork"` (giving up `resume=True`).
6. **Hit a provider-native exception?** Anthropic / OpenAI /
   Google SDK exceptions propagate as-is — LazyBridge does not
   wrap them. The `Executor` retries them when
   `provider.is_retryable(exc)` returns `True`; otherwise they
   reach you verbatim.

## Best practices

- **`result.ok` first, `.payload` second.** Production code
  should always check `result.ok` before reading the payload.
  An error envelope's payload is whatever was last produced
  (often `None` or a partial result).
- **Bound everything.** `Agent(timeout=N)`,
  `LLMEngine(max_turns=N, tool_timeout=N, stream_idle_timeout=N)`,
  `Plan(max_iterations=N)`, `verify=judge` with `max_verify=N`.
  Every loop in the framework has a budget; pick defensible
  defaults rather than relying on `None`.
- **Fail loud at construction.** Plan validation and provider
  registration mistakes should surface at `Agent(...)` /
  `Plan(...)` time. If you're catching `PlanCompileError`
  routinely, your construction code is probably wrong, not your
  inputs.
- **Use `Session` for forensics.** A failing run is opaque
  without an event log. Even in development, pair an
  `Agent(verbose=True)` or `Session(console=True)` with the run;
  in production, `JsonFileExporter` or `OTelExporter` give you
  the same per-step trace post-mortem.

## See also

- [Envelope](guides/basic/envelope.md) — `Envelope.ok` /
  `Envelope.error` semantics, `ErrorInfo` shape, retryable flag.
- [Session](guides/mid/session.md) — `events.query(...)` for
  pulling the trace of a failing run.
- [Exporters](guides/full/exporters.md) — `EventType.TOOL_TIMEOUT`
  vs `EventType.TOOL_ERROR`; per-event-type exporter wiring.
- [Checkpoint & resume](guides/full/checkpoint.md) —
  `ConcurrentPlanRunError` and the `on_concurrent` policy table.
- [Plan](guides/full/plan.md) — `PlanCompileError` taxonomy and
  the `max_iterations` safety net.
- [BaseProvider](guides/advanced/base-provider.md) —
  `UnsupportedNativeToolError` / `UnsupportedFeatureError`
  strict-mode behaviour.
