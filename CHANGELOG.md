# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **`lazybridge.gui` umbrella** — core, stdlib-only package for per-object GUIs. Importing it installs a ``.gui()`` method on ``LazyAgent``, ``LazyTool``, and ``LazySession``; calling it spins up (or reuses) a single shared HTTP server and opens a browser tab. Each panel has an **Inspect/Edit** tab and a live **Test** tab:
  - `AgentPanel` — read provider/model/name/description; live-edit `system` prompt; toggle which session-scoped tools are enabled on the agent; run live `chat` / `loop` / `text` against the real provider (usage + cost surfaced inline).
  - `ToolPanel` — schema-driven form (one typed input per parameter) that invokes `tool.run(args)` on submit; works for function-backed tools and for pipeline tools (`LazyTool.chain` / `.parallel`).
  - `SessionPanel` — session id, tracking level, registered agents, current store keys. `session.gui()` also auto-registers a panel for every agent and tool already in the session so they appear in the sidebar immediately.
  - Shared primitives: `GuiServer`, `Panel` (ABC), `get_server`, `close_server`, `is_running` — exposed so third parties can add their own panels (and so the planned whole-pipeline GUI can embed existing panels as sub-views).
  - Security: 127.0.0.1-bound, 24-byte urlsafe token required on every `/api/*` request; `ThreadingHTTPServer` spawns a thread per request so long live test calls don't block sidebar polling.
  - 47 new tests in `tests/unit/gui/` covering the server + panel plumbing, each panel kind, the monkey-patch installer, and the end-to-end HTTP round-trip for `tool.gui()`.
- `examples/gui_demo.py` — researcher + writer + shared search tool, opens `session.gui()`.
- `lazybridge.gui.human` — optional, **stdlib-only** browser UI for `HumanAgent` and `SupervisorAgent`. Exposes `WebInputServer` and a one-call `web_input_fn()` factory that returns a drop-in `input_fn`. Each REPL prompt renders on a local `127.0.0.1:<ephemeral>` page with the previous agent output, optional quick-command chips, and a submit textarea; Ctrl/⌘-Enter submits. Token-gated (random 24-byte urlsafe) and localhost-bound. Covered by 14 unit tests in `tests/unit/gui/human/test_web_input.py`, including an end-to-end integration with `SupervisorAgent`.
- `examples/human_gui_demo.py` — runnable `researcher → SupervisorAgent → writer` pipeline using the browser UI.

### Documentation
- `README.md` — new **Human-in-the-loop** section showcasing `SupervisorAgent` in a `researcher → supervisor → writer` chain, plus an entry in the Documentation table.
- `lazy_wiki/bot/13_supervisor.md` — new LLM-oriented reference for `HumanAgent` / `SupervisorAgent` (constructors, methods, REPL commands, scripted-input patterns for tests).
- `lazy_wiki/bot/INDEX.md` — `HumanAgent` and `SupervisorAgent` added to the class table, import block, and reading order.
- `lazy_wiki/human/quickstart.md` — cross-reference rows for human-in-the-loop supervision in both the "Next steps" and "Choosing the right pattern" tables.
- `mkdocs.yml` — new Guide nav entry "Human-in-the-Loop" pointing at the existing module 13 walkthrough.
- `examples/supervised_pipeline.py` — runnable, non-interactive demo wiring `researcher → SupervisorAgent → writer` via scripted `input_fn`, exercising `continue`, `retry`, and tool-call commands.
- `lazybridge/gui/human/README.md` — full API reference for the new browser UI extension; cross-linked from `README.md`, `lazy_wiki/human/agents.md`, and `lazy_wiki/bot/13_supervisor.md`.

### Added (prior, now surfaced)
- `HumanAgent` and `SupervisorAgent` (`lazybridge/human.py`, `lazybridge/supervisor.py`) — human-in-the-loop participants that slot into chains, parallel tools, `as_tool()`, and `verify=`. Exported from `lazybridge` top-level.

---

## [0.6.0] — Claude Opus 4.7 support + version bump

See commit `20f7130` for details.

---

## [0.5.0] — 2026-04-14

### Breaking Changes

- **Core/extension separation**: domain-specific modules moved under `lazybridge.ext/`:
  - `lazybridge.stat_runtime` → `lazybridge.ext.stat_runtime`
  - `lazybridge.data_downloader` → `lazybridge.ext.data_downloader`
  - `lazybridge.quant_agent` → `lazybridge.ext.quant_agent`
  - `lazybridge.tools.doc_skills` → `lazybridge.ext.doc_skills`
  - `lazybridge.tools.read_docs` → `lazybridge.ext.read_docs`
- `quant_agent` is no longer exported from the top-level `lazybridge` package.
  Use `from lazybridge.ext.quant_agent import quant_agent` instead.

### Added
- `LazyTool.parallel(*participants, name, description, combiner="concat", native_tools=None, session=None, guidance=None)` — session-free fan-out pipeline tool. All participants run concurrently on the same task; results are concatenated or the last result is returned depending on `combiner`.
- `LazyTool.chain(*participants, name, description, native_tools=None, session=None, guidance=None)` — session-free sequential pipeline tool. Each participant receives the previous output as its task (tool→agent) or as injected context (agent→agent).
- `lazybridge/pipeline_builders.py` — neutral module extracted from `lazy_session.py`. Contains `build_parallel_func`, `build_chain_func`, `_ChainState`, `_clone_for_invocation`, `_resolve_participant`, `_validate_session_compatibility`. No circular imports — all cross-module imports are deferred inside function bodies.
- `LazyTool._is_pipeline_tool: bool` — dataclass field set to `True` by `parallel()` and `chain()`. Prevents accidental `save()` calls on runtime-only pipeline tools.
- `_clone_delegate_tool_for_invocation()` in `lazy_tool.py` — friend-module helper for cloning `LazyTool.from_agent()` participants in pipeline execution.

### Changed
- `LazySession.as_tool()` parallel and chain paths now delegate to `pipeline_builders.build_parallel_func` / `build_chain_func`. Behavior is identical; code is no longer duplicated.
- `_ChainState` moved to `pipeline_builders.py`; re-exported from `lazy_session.py` for backward compatibility.

### Fixed
- `LazyTool.save()` now raises `ValueError` for pipeline tools (chain/parallel) that cannot be serialized to disk.

---

## [0.3.1] — 2026-04-02

### Added
- `verbose: bool = False` on `LazyAgent` — prints all tracked events to stdout in real-time. For standalone agents (no session), creates a private `EventLog(console=True)`. For session agents, enables `console=True` on the shared `EventLog`.
- `console: bool = False` on `LazySession` — prints events to stdout from the session's `EventLog`.
- `TrackLevel.FULL` — synonym for `TrackLevel.VERBOSE`.
- `LazyStore.awrite()` / `aread()` / `aread_all()` / `akeys()` — async wrappers that offload SQLite I/O to the thread-pool executor, preventing event loop blocking in async agent contexts.
- `Memory.from_history(messages: list[dict])` — classmethod to restore a `Memory` instance from a serialised history list.
- `LazySession.as_tool()` — new `mode="parallel"` and `mode="chain"` paths with optional `participants=` list. All registered session agents are used by default. Participants can mix `LazyAgent` and `LazyTool` instances (nested pipelines). The `entry_agent=` path is retained for backward compatibility.
- `_JSON_SYSTEM_SUFFIX` injected into system prompt by `json()` / `ajson()` — belt-and-suspenders JSON enforcement alongside native structured output API.

### Fixed
- `MODEL_REQUEST` tracking: `model` field used `request.model` (often `None` for default-model calls); now falls back to `self._model_name`.
- `MODEL_RESPONSE` tracking: added `model` and `content` fields to the event data.
- `_run_suppressed()` in `lazy_run.py`: custom event loop runner that installs an exception handler suppressing "Event loop is closed" errors from httpx GC-time cleanup tasks, then cancels pending tasks and calls `shutdown_asyncgens()` before closing.
- Asyncio logging filter in test `conftest.py` — suppresses "Task exception was never retrieved: Event loop is closed" cosmetic warnings from the asyncio logger.
- `investment_research_platform.py` example: removed invalid `.loop()` call on `LazyTool`; added `SectorAllocation` Pydantic model replacing unsupported `dict[str, str]` for OpenAI structured output; added explicit JSON-only instructions to all schema agents; updated orchestrator system prompt to prevent redundant parallel tool calls.

---

## [0.3.0] — 2026-03-31

### Added
- `ModelBridge.__init__()` — three new production-resilience parameters:
  - `max_retries: int = 0` — retry count on transient API errors (429 / 5xx) with exponential backoff and ±10 % jitter
  - `retry_delay: float = 1.0` — base delay in seconds for backoff (`delay * 2^attempt`)
  - `max_stream_buffer_chars: int = 500_000` — cap on stream-repair buffer; emits `ResourceWarning` when exceeded
- `UsageStats.cost_usd: float | None` — estimated cost in USD computed at parse time from each provider's price table. `None` when the model is not in the table.
- `AnthropicProvider` — two new constructor keyword arguments (passed via `ModelBridge(**kwargs)`):
  - `force_stream_threshold: int` — token count above which sync API silently auto-switches to streaming (default: 20 000); now configurable and emits a `DEBUG` log
  - `beta_overrides: dict[str, str]` — pin or override specific Anthropic beta header versions (e.g. `{"web_search": "web-search-2026-xx-xx"}`) without subclassing
- `ToolSchemaBuilder(flatten_refs=True)` — opt-in post-compilation step that recursively inlines `$ref` / `$defs` from Pydantic-generated JSON Schema. Use for providers that do not resolve `$ref` natively.
- `_flatten_refs(schema)` — public utility function that inlines `$ref`/`$defs` in any JSON Schema dict.

### Fixed
- `AgentTrack.log()` exceptions (SQLite lock, disk-full) are now swallowed in an internal `_track()` helper — a logging failure never crashes the primary LLM request.
- `ToolSchemaMode.LLM` and `HYBRID` modes: transient LLM failures during schema compilation now fall back to `SIGNATURE` mode with a `UserWarning` instead of propagating `ToolSchemaBuildError`.
- Tool names longer than 64 characters raise `ValueError` before the API call, with a clear message (Anthropic and OpenAI both reject names over 64 chars with a cryptic 400).
- `_make_arg_model()` cache changed from `@lru_cache` to `WeakKeyDictionary` keyed on `(func, id(func.__annotations__))`. Entries are evicted automatically when functions are garbage-collected, and stale schemas caused by annotation mutation are detected and rebuilt.
- `model.model_json_schema()` now called with `mode="serialization"` — produces schemas that match actual serialisation output for Pydantic models with computed fields or custom serializers.
- `AgentMemory._get_lineage()` and `_is_child_readable()` recursive CTEs are depth-capped at 100 levels to prevent runaway recursion in cyclic or pathologically deep hierarchies.
- `_retry_complete` / `_aretry_complete` and stream-buffer guards use `getattr(..., default)` for all instance attributes added in `__init__`, so `LazyLayer.__new__()`-based test construction no longer crashes.

---

## [0.2.0] — 2026-03-28

### Added
- `AgentMemory` — SQLite-backed hierarchical memory with modes: off / private / parent / grandparent / family
- `AgentTrack` — SQLite event log with modes: off / basic / superverbose
- `LazySession` — shared tracking session for multi-agent pipelines (`LazyLayer.session()`)
- `LazyLayer.child_layer()` / `finish_child()` / `get_child_context()` — parent-child agent hierarchy
- `LazyLayer.as_tool()` — wrap a layer as `ToolBridge` in one call
- `LazyLayer.session()` now accepts `memory` parameter to control `MemoryMode` for all session agents
- Root-level `_root_track` on `LazySession` — logs `CHILD_CREATED` events for complete run tree visibility
- Composite SQLite index `(agent_id, scope)` on `memory_items` for faster child context queries
- `UserWarning` when `temperature` is passed alongside `thinking=True` (silently ignored before)
- GitHub Actions CI: `.github/workflows/test.yml`, `.github/workflows/lint.yml`
- Pre-commit config: `.pre-commit-config.yaml` (ruff, ruff-format, detect-private-key)
- Ruff linting config in both `pyproject.toml` files
- Root `README.md` with quick start, architecture overview, and development guide
- `CHANGELOG.md` (this file)

### Fixed
- `SQLiteMemoryStore._connect()` converted to `@contextmanager` — eliminates file handle leak
- `LazySession.agent()` no longer creates `AgentMemory` when `memory_mode=OFF` — avoids spurious DB writes
- `AgentMemory._get_lineage()` replaced N+1 loop with single recursive CTE — better performance and no open connection across loop iterations
- SQLite busy timeout increased to 30s (`timeout=30` + `PRAGMA busy_timeout=30000`) in `AgentMemory` and `AgentTrack` — handles lock contention in multi-process pipelines
- `test_toolbridge.py` — all `_make_ai()` helpers now set `_agent_memory=None` and `_agent_track=None` (bypassed by `__new__`, caused 10+ test failures)
- `test_lazylayer_providers.py` — removed hardcoded Windows path `D:/LazyBridge`
- `modelbridge/pyproject.toml` — legacy `setuptools.backends.legacy:build` replaced with `setuptools.build_meta`
- Google provider: `thought_signature` preserved as raw Part objects to avoid 400 INVALID_ARGUMENT on follow-up requests
- `thinking-mode` detection uses `"-4-6"` prefix to avoid false match on `claude-sonnet-4-5`
- `tool_loop` / `atool_loop`: detect awaitable result from sync `tool_runner` and raise clear `RuntimeError`
- `tool_loop` / `atool_loop`: `None` result serializes to `""` instead of `"None"`
- `child_layer()` raises `ValueError` when parent has no `agent_memory`
- `_remember_response` uses `isinstance(blk, TextContent)` — correct content block detection
- `agent_id` defaults to `uuid4()` instead of `""`

### Removed
- Dead `telemetry: dict[str, Any]` field from `CompletionResponse` and `StreamChunk` (never populated)

### Changed
- `max_tokens` default changed from `4096` → `None` across all chat/stream/atext signatures
- `setup_paths.py` moved to `dev/setup_paths.py` with dynamic path detection (cross-platform)
- `website_examples.py` and `website_examples_guide.html` moved to `docs/examples/`

---

## [0.1.0] — 2026-03-27

### Added
- Initial import: `modelbridge`, `lazybridge`, `modelbridge_updater`
- Multi-provider support: Anthropic, OpenAI, Google Gemini, DeepSeek
- Core types: `Message`, `CompletionRequest`, `CompletionResponse`, `StreamChunk`, `ToolDefinition`, `NativeTool`, `ThinkingConfig`
- `ToolBridge`, `ToolBridgeGuidance`, `ToolSchemaBuilder` in lazybridge
- Multi-provider integration tests (28 live tests)
