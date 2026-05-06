# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — 2026-05-05 — bug-fix and routing hardening

### Breaking

- **`LLMEngine.stream_idle_timeout` default changed from `None` to
  `90.0` s.**  Old default left provider streams unbounded — a
  half-open HTTP/2 connection (TCP RST never delivered, PING dropped)
  would pin a worker indefinitely.  New default raises
  ``StreamStallError`` after 90 s of inter-chunk silence, which is
  large enough to absorb provider-side thinking pauses on
  Opus / Gemini Pro.  Pass ``stream_idle_timeout=None`` to opt out
  explicitly — a one-shot ``UserWarning`` flags the choice because
  the failure mode (worker pinned forever) is silent and hard to
  diagnose.  The class-level default exposed for ``__new__`` /
  subclass paths shifts the same way.  No code change is required for
  callers that already pass an explicit value.

### Features

- **`Step.after_branches`** — exclusive-branch rejoin point.  Set
  alongside `routes` / `routes_by` to route to exactly one branch and
  skip all sibling steps; execution resumes at the named step after the
  branch completes.  Without `after_branches`, routing is a *detour*
  (linear progression resumes from the routed-to step's declared
  position).  See `Step` docstring for the full example.

### Bug Fixes (Critical)

- **Store SQLite CAS: open transaction on `JSONDecodeError`** — the
  `except sqlite3.Error` clause in `compare_and_swap` did not catch
  `json.loads` failures on corrupt rows, leaving `BEGIN IMMEDIATE`
  open on the thread-local connection and poisoning every subsequent
  call on that thread.  Widened to `except (sqlite3.Error, ValueError)`
  (``json.JSONDecodeError`` is a ``ValueError`` subclass).
- **Store in-memory: mutable references break CAS invariants** —
  `read()` and `write()` returned / stored the live Python object,
  so callers mutating the result could silently alter the stored value
  and defeat `compare_and_swap`.  Both paths now go through
  `_deep_copy_safe` (deep-copy with a non-copyable fallback).
- **`LLMEngine` `tool_choice="any"` infinite loop** — after the model
  satisfied the "must call at least one tool" contract on the first
  turn, `provider_tc` stayed `"any"`, forcing every subsequent turn
  to also call a tool.  The loop never exited until `max_turns` was
  exhausted.  Fixed: `provider_tc` is reset to `"auto"` immediately
  after the first tool-result turn.
- **`LLMEngine` `tool_choice="any"` sent as literal tool name** —
  Anthropic and OpenAI reject `tool_choice="any"` as an unknown tool
  name.  The framework now maps `"any"` → `"required"` when building
  the provider request so the wire value is always a recognised
  constant.
- **Plan: parallel-band failure checkpoint pointed at failing step** —
  when a step inside a parallel band failed, the checkpoint recorded
  `current_step` as the failing step rather than the band-start, so
  `resume=True` re-entered mid-band in an inconsistent state.  Now
  points at the band-start step.
- **Agent: failed structured-output retries contaminated memory** —
  correction retries in `_validate_and_retry` were called with the
  live `memory` object, so each failed attempt added a garbage turn to
  the agent's conversation history.  Retries now pass `memory=None`;
  only the final accepted result reaches memory.
- **`Agent.stream()`: input guard bypassed** — `acheck_input` was not
  called in the streaming path, so `guard=` had no effect when callers
  used `async for token in agent.stream(...)`.  The guard check now
  runs before the first token is emitted.

### Bug Fixes (High)

- **`LLMGuard` sync path: `timeout=` ignored** — `_judge` invoked
  `self._agent(prompt)` directly on the calling thread without any
  deadline.  Fixed by running the judge in a daemon thread and calling
  `thread.join(timeout=self._timeout)`.
- **Memory `strategy="sliding"` silently disabled with
  `max_tokens=None`** — `_plan_compression` gated all compression on
  `bool(self.max_tokens)`, so `strategy="sliding"` with the default
  `max_tokens=None` never triggered.  Fixed: only `"auto"` requires a
  token budget; `"sliding"` and `"summary"` compress by turn count
  independently of `max_tokens`.
- **Memory: `_overflow_warned` flag shared between turn-cap and
  summarizer timeout** — a summariser timeout silenced the turn-cap
  warning (or vice versa) because both used the same flag.  Split into
  `_overflow_warned` (turn cap) and `_summarizer_warned` (summariser
  timeout).
- **Predicates: `empty()` / `not_empty()` treated `0` / `False` as
  empty** — only `None` and zero-length containers (`str`, `list`,
  `dict`, `tuple`, `set`, `frozenset`) are now considered empty.
  Numerics and booleans are always non-empty; use `eq(0)` / `eq(False)`
  for those cases.
- **Google provider: `finish_reason` never mapped to `"max_tokens"`** —
  the `MAX_TOKENS` stop reason from the Google API was not translated,
  so callers inspecting `stop_reason` always saw `None` instead of
  `"max_tokens"`.
- **Tool schema: `model_dump()` destroyed Pydantic model args** —
  `Tool.definition()` called `model_dump()` on the entire arguments
  dict, collapsing Pydantic model instances to plain dicts before the
  schema was built.  Fixed: `getattr` per field preserves the original
  objects.

---

## [0.7.0] — pre-1.0 reset, simplified namespace layout

**Major reorganization.** The framework is dropping back to pre-1.0 and
reshaping its namespace boundaries before stabilizing. Everything is
`alpha`.

### Breaking
- **Version downgrade**: `1.0.0 → 0.7.0`. The 1.0 release was premature
  given the API churn since; 0.7.x is the honest baseline.
- **Single stability tier**: every surface is `alpha`. The
  `stable / beta / alpha / domain` 4-tier taxonomy is removed. Per-module
  `__stability__` and `__lazybridge_min__` markers are removed; only
  `lazybridge.__stability__ = "alpha"` remains.
- **Namespace reorganization** — domain modules moved out of
  `lazybridge.ext.*`:
  - `lazybridge.ext.read_docs`        → `lazybridge.external_tools.read_docs`
  - `lazybridge.ext.doc_skills`       → `lazybridge.external_tools.doc_skills`
  - `lazybridge.ext.data_downloader`  → `lazybridge.external_tools.data_downloader`
  - `lazybridge.ext.stat_runtime`     → `lazybridge.external_tools.stat_runtime`
  - `lazybridge.ext.report_builder`   → `lazybridge.external_tools.report_builder`
  - `lazybridge.ext.external_tools`   → `lazybridge.ext.gateway` (file rename to free the namespace)
- `lazybridge.ext.*` is now reserved for **framework extensions** that
  augment the agent runtime (`mcp`, `otel`, `hil`, `evals`, `gateway`,
  `planners`, `viz`).
- New namespace `lazybridge.external_tools.*` — domain tool packages
  (returns `list[Tool]`).

### Removed
- `lazybridge.ext.veo` and `lazybridge.ext.quant_agent` — neither was
  ready for use and they only created confusion. Re-introduce later if
  the underlying integrations stabilize.
- `lazybridge.external_tools.stat_runtime` (statistical / econometrics
  sandbox) and `lazybridge.external_tools.data_downloader` (Yahoo /
  FRED / ECB market-data ingestion) — same rationale: scope-creep
  domain examples that distract from the framework's actual surface.
  The matching `[stats]` and `[downloader]` optional-deps extras are
  also removed from `pyproject.toml`.

### Tool factory shape (breaking)
- All surviving `external_tools/*` factories standardize on
  `def X_tools(*, ...) -> list[Tool]` — keyword-only arguments, always
  returning a list. Single-tool cases return a 1-element list.
  - `report_tools(*, output_dir=...)`
  - `fragment_tools(*, bus, default_section=None, step_name=None)`
  - `skill_tools(*, skill_dir, ...)` (was `skill_tool(skill_dir, ...) -> Tool`)
  - `skill_builder_tools(*, ...)` (was `skill_builder_tool(...) -> Tool`)
  - `read_docs_tools(*, base_dir=None)` — new factory wrapping
    `read_folder_docs` as a Tool.

### Boundary
- New CI check (`tools/check_ext_imports.py`): `ext/` and
  `external_tools/` may only import from public `lazybridge.*`, never
  from internal `lazybridge.core.*` or other private submodules.

### Migration
```python
# before
from lazybridge.ext.read_docs import read_docs_tools
from lazybridge.ext.external_tools import ExternalToolGateway

# after
from lazybridge.external_tools.read_docs import read_docs_tools
from lazybridge.ext.gateway import ExternalToolGateway
```

---

## [0.7.0 — short-term audit hardening] — bundled into the 0.7.0 cut

Closes the high-severity findings from the deep architecture audit
(plan §5.1).  All changes are additive; defaults shift only on
`Session(batched=True)` (`on_full="hybrid"` instead of `"drop"`) which
strictly improves the safety of the existing path — critical events
that previously could be dropped under saturation now block the
producer.  Pass `on_full="drop"` to opt back into the legacy policy.

### Hardening

- **OTel GenAI conventions** (audit H-D).  `OTelExporter` now emits
  `gen_ai.system` / `gen_ai.request.model` / `gen_ai.usage.*` /
  `gen_ai.tool.*` attributes per the OpenTelemetry Semantic
  Conventions for Generative AI, and constructs a real parent-child
  span hierarchy (`invoke_agent → chat`, `invoke_agent → execute_tool`)
  with cross-agent context propagation through OTel contextvars.
  Tool spans correlate via `tool_use_id` so N parallel invocations
  of the same tool no longer collide.  Span registry is per-instance
  so multiple `OTelExporter`s in a process don't fight over the
  global tracer provider.
- **`Memory.summarizer_timeout=`** (audit H-B).  Default 30 s.  An LLM
  summariser that hangs no longer blocks `add()` — the keyword
  fallback runs and a one-shot warning surfaces.  Compression also
  computes the summary OUTSIDE `Memory._lock`, so concurrent `add()`
  calls progress while a slow summariser is in flight.
- **Per-event-type back-pressure in `EventLog`** (audit H-A).  New
  default `on_full="hybrid"` — the writer queue blocks the producer
  for audit-critical events (`AGENT_*` / `TOOL_*` / `HIL_DECISION`)
  but drops cheap telemetry (`LOOP_STEP` / `MODEL_REQUEST` /
  `MODEL_RESPONSE`) under saturation.  Override the set via
  `Session(critical_events=...)`.  `"block"` and `"drop"` policies
  remain available unchanged.
- **MCP `_tools_cache` TTL + invalidation** (audit H-E).  New
  `cache_tools_ttl` parameter on `MCPServer` / `MCP.stdio` / `MCP.http`
  (default 60 s) and an `invalidate_tools_cache()` method.  An MCP
  server that hot-loads or unloads tools is eventually reflected in
  the agent's tool list instead of forever-stale.
- **Loud surfacing of malformed tool-call arguments** (audit M-A).
  Provider `_safe_json_loads` helpers (OpenAI, LiteLLM) now tag the
  raw argument blob with `_parse_error` on JSON decode failure or
  non-object payload.  `LLMEngine._exec_tool` short-circuits on the
  tag and emits a structured `TOOL_ERROR` (`type:
  "ToolArgumentParseError"`, `parse_error`, `raw_arguments`) instead
  of letting the tool fail downstream with a misleading
  "missing required field" message.  Tool events also carry
  `tool_use_id` for downstream correlation.

### Tests / CI

- New `tests/unit/test_audit_short_term.py` (17 tests) covering each
  of the above plus the streaming + tool-call accumulation regression
  for Gemini / DeepSeek shape (audit M-B).
- Coverage policy widened (audit M-I): `lazybridge/ext/{otel,mcp,hil,
  planners,evals}` are now in scope for the gate (previously omitted
  wholesale).  Domain extensions (`stat_runtime`, `data_downloader`,
  `doc_skills`, `veo`, `quant_agent`, `read_docs`, `external_tools`)
  remain omitted because their dedicated test suites live under
  `tests/unit/ext/` and are skipped by the default run.  Gate stays
  at 70 with broader coverage; target for 1.1 is 80.
- New CI workflows (audit M-J): `release.yml` (PyPI Trusted Publishing
  on `v*.*.*` tags), `codeql.yml` (weekly scheduled SAST + per-PR),
  `dependabot.yml` (weekly action + pip updates with major SDK pins
  preserved).  Pre-commit hooks now run as a CI job.

---

## [1.0.0] — 2026-04-26 — initial public release

> **Historical**: this entry describes the deleted 4-tier stability
> taxonomy (`stable / beta / alpha / domain`) and the original namespace
> layout. Both were removed in 0.7.0; entries below are kept for
> historical accuracy only.

### Core

- `Agent` — universal façade with swappable engines (`LLMEngine`, `Plan`,
  plus `HumanEngine` / `SupervisorEngine` from `lazybridge.ext.hil`).
  One call surface (`agent.run` / `agent(...)` / `agent.stream`)
  regardless of engine.
- **Tool-is-Tool**: plain functions, `Agent` instances, `Agent.as_tool()`
  results, and tool providers (e.g. an MCP server) all plug into
  `tools=[...]` with the same dispatch contract. Nested agents inherit
  the outer session for end-to-end observability.
- **Compile-time DAG validation**: `PlanCompiler` rejects duplicate
  step names, forward references, broken `from_step` / `from_parallel`
  / `from_parallel_all` sentinels, and parallel-band misuse before
  any LLM call.
- **Crash-resume**: `Plan` checkpoints to `Store` via `compare_and_swap`.
  Concurrent runs on the same `checkpoint_key` collide at claim time;
  `resume=True` adopts an in-flight checkpoint.
- **Parallel tool dispatch**: when an LLM emits multiple tool calls in
  one turn, the engine runs them concurrently via `asyncio.gather`.
- **Structured output**: `output=SomeBaseModel` flips the engine into
  schema-validated mode with retry-with-feedback up to
  `max_output_retries`.
- **Sources / Memory / Store / Session / Guard**: composable
  observability and state primitives. `Session` is SQLite-backed with
  thread-local connections and a batched event-log writer.
- **Sync façade**: `agent("…")` works inside or outside a running
  event loop. When invoked from inside one, the worker-thread loop
  inherits the caller's `contextvars` context so OTel spans / request
  IDs / structured-logging context flow through.

### Providers

- `AnthropicProvider`, `OpenAIProvider`, `GoogleProvider`,
  `DeepSeekProvider`, `LiteLLMProvider`, `LMStudioProvider`.
- Provider-tier aliasing (`super_cheap` → `cheap` → `medium` →
  `expensive` → `top`) keeps preview / date-pinned model strings in
  one place per provider.
- LMStudio adapter is a thin `OpenAIProvider` subclass that targets
  the local server (`http://localhost:1234/v1` by default), pinned to
  Chat Completions, with zero cost and no native tools.
- Native server-side tools per provider:
  `WEB_SEARCH` / `CODE_EXECUTION` / `FILE_SEARCH` / `COMPUTER_USE` /
  `GOOGLE_SEARCH` / `GOOGLE_MAPS`.
- Prompt-caching forwarded via `cache=True` (Anthropic explicit,
  OpenAI / DeepSeek auto, Google via separate API).

### Extensions (`lazybridge.ext`)

Extensions ship at `__stability__ = "alpha"` by default and may break
between minor releases. Promotion: alpha → beta → stable → core. See
`docs/guides/core-vs-ext.md` for the policy.

- `ext.hil` — `HumanEngine` (approval gate) and `SupervisorEngine`
  (full REPL with tool calls, retry-with-feedback, store access).
- `ext.planners` — `make_planner` (DAG builder) and
  `make_blackboard_planner` (todo-list).
- `ext.mcp` — Model Context Protocol integration at the tool boundary.
  `MCP.stdio` / `MCP.http` / `MCP.from_transport` build a tool
  provider that drops into `Agent(tools=[server])`.
- `ext.otel` — OpenTelemetry exporter for `Session`.
- `ext.evals` — `EvalSuite` / `EvalCase` / `llm_judge` / built-in
  matchers.
- `ext.stat_runtime` — sandboxed DuckDB query engine with
  AST-validated SQL (sqlglot DuckDB dialect; defence-in-depth regex
  layer for environments without sqlglot).
- `ext.data_downloader`, `ext.quant_agent`, `ext.doc_skills`,
  `ext.read_docs`, `ext.veo` — domain extensions.

### Documentation

- Per-tier guides (`docs/tiers/{basic,mid,full,advanced}.md`).
- Decision trees (`docs/decisions/`) — when to use which primitive.
- Recipes (`docs/recipes/`) — tool calling, structured output,
  pipeline with resume, human-in-the-loop, MCP, orchestration tools.
- LLM-assistant skill (`lazybridge/skill_docs/`) ships with the
  package; same content as the site, signature-first for LLM
  consumption. Single-source via
  `python -m lazybridge.skill_docs._build`.
