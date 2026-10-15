# Sessions, memory, store and guards

Read when adding conversation memory, durable state, observability (events,
exporters, OpenTelemetry), or input/output guards. Source of truth:
`lazybridge/memory.py`, `lazybridge/store/`, `lazybridge/session.py`,
`lazybridge/exporters.py`, `lazybridge/guardrails.py`,
`lazybridge/dedup_guard.py`; docs `docs/guides/mid/{memory,store,session,guards,dedup-guard}.md`,
`docs/guides/full/exporters.md`, `docs/guides/advanced/otel.md`.

## Three different "sessions" — keep them apart

- **LazyBridge `Session`**: observability events plus in-process approval
  grants and runtime slots.
- **Native Claude session / Codex thread**: the coding runtime's own model
  history (see `coding-engines.md`).
- **`SessionRegistry`**: a JSON file mapping human aliases to native ids.

`Memory`, `Store` and `Session` objects can be shared between agents with
different engines; native history never migrates between engines.

## Memory (conversation history)

`Memory(strategy="auto" | "sliding" | "summary" | "none", max_tokens=4000,
max_turns=1000, store=None, key=None, session_key="default", summarizer=None,
summarizer_timeout=30.0)`. Attach with `Agent(memory=Memory())`. With
`store=` + `key=` the history persists in the Store's dedicated memory table.
Methods: `add`, `messages`, `text`, `clear`, `amend_last`. When a coding
engine resumes a durable native conversation, `Memory` is no longer prepended
(the runtime holds the history) but it keeps recording completed answers.

## Store (durable key/value + CAS)

```python
from lazybridge import Store

store = Store(db="state.sqlite")        # Store() = in-memory, lost on exit
store.write("job:2026-10-15", {"status": "done"}, agent_id="ingest")
entry = store.read_entry("job:2026-10-15")   # value, written_at, agent_id
ok = store.compare_and_swap("lock:report", None, "worker-1")
```

Methods: `write`, `read`, `read_entry`, `read_all`, `items(prefix=)`, `keys`,
`delete`, `clear`, `to_text` (render for `sources=`), `compare_and_swap`,
`write_memory` / `read_memory` / `delete_memory`, `close`. The SQLite backend
is WAL-mode and thread-safe, and several processes may share the file
(cross-process coordination goes through `compare_and_swap`); since 1.8.0 concurrent first opens
retry instead of failing "database is locked", and since 1.5.0 failed writes
roll back. Use the Store for checkpoints, cross-run registries, handles to
large data and idempotency reservations; do not open new SQLite/DuckDB files
for the same purpose.

## Session, events and exporters

```python
from lazybridge import Agent, JsonFileExporter, LLMEngine, Session

session = Session(exporters=[JsonFileExporter("events.jsonl")])
agent = Agent(engine=LLMEngine("claude-opus-5-5"), session=session, name="observed")
```

- `Session(db=, exporters=[...], redact=, redact_on_error="strict",
  unsafe_log_payloads=False, console=False, batched=False, ...)`. Events are
  recorded in an `EventLog` and typed by `EventType` (agent start/finish,
  tool call/result, tool timeout, ...). Coding engines emit native tool calls
  as `TOOL_CALL` / `TOOL_RESULT` with `native: True` (1.4.0).
- Exporters: `JsonFileExporter`, `ConsoleExporter`, `CallbackExporter`,
  `FilteredExporter`, `StructuredLogExporter`, or a custom `EventExporter`.
  OpenTelemetry: `from lazybridge.ext.otel import OTelExporter`
  (`pip install "lazybridge[otel]"`).
- `GraphSchema` describes the agent/plan topology for viewers and the
  visualizer.

**Inheritance (verified):** an agent passed via `tools=[...]`, `fallback=` or
`verify=` inherits the parent's session automatically. An agent used
**directly as a `Plan` `Step` target does not**: pass `session=` when
building it, or its events (and those of everything it calls) never reach
the log.

## Guards (hard gates)

```python
from lazybridge import ContentGuard, GuardAction, GuardChain, LLMGuard


def no_secrets(text: str) -> GuardAction:
    return GuardAction(allowed="sk-live-" not in text, message="secret detected")


guard = GuardChain(ContentGuard(input_fn=no_secrets, output_fn=no_secrets),
                   LLMGuard(judge_agent, "Reject requests for personal data."))
agent = Agent(engine=LLMEngine("claude-opus-5-5"), guard=guard, name="guarded")
```

- `Guard` protocol: `acheck_input(text)` / `acheck_output(text)` returning
  `GuardAction(allowed, message, modified_text, metadata)`.
- A blocked input never reaches the engine; a blocked output becomes an error
  envelope. `GuardError` is raised by some integrations on hard failure.
- `DeduplicateGuard` strips repeated text blocks from multi-agent inputs
  before they reach the model.
- Guards block; `verify=` re-asks (see `agents-and-engines.md`).
