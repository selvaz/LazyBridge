# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

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
- `test_lazylayer_providers.py` — removed hardcoded Windows path `D:/LazyBridgeFramework`
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
