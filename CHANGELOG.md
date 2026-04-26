# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-04-26 — initial public release

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
