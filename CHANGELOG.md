# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-04-21 — **v1 complete rewrite**

v1 is a complete, breaking rewrite. Every public class from 0.x
(`LazyAgent`, `LazySession`, `LazyTool`, `LazyContext`, `LazyStore`,
`LazyRouter`, `SupervisorAgent`, `HumanAgent`) is removed. The API
collapses to a single `Agent` with swappable engines and a uniform
`Tool` contract. ~4700 → ~1300 LOC of core runtime; 217 tests green.

### Added — architecture

- **`Agent`** — universal agent. Delegates to a swappable `Engine`
  (`LLMEngine`, `HumanEngine`, `SupervisorEngine`, `Plan`). One call
  surface regardless of engine: `agent(task) -> Envelope`,
  `await agent.run(task)`, `async for ... in agent.stream(task)`.
- **`Envelope[T]`** — single data type flowing between engines.
  Generic over its payload. `payload` / `metadata` / `error`; `.ok`,
  `.text()`. Replaces the scattered `_last_output` / `_last_response`
  contract of 0.x.
- **`Engine` protocol** — runtime-checkable contract (`run` + `stream`).
  Used by `LLMEngine`, `HumanEngine`, `SupervisorEngine`, and `Plan`.
  Subclass to ship custom engines (ReAct, router-based, caching, …).
- **`Tool` + Tool-is-Tool contract** — `tools=[...]` accepts plain
  functions, `Tool` instances, `Agent` instances, and Agents-of-Agents
  uniformly. `wrap_tool` normalises everything; nested Agents inherit
  the outer `Session` and register `as_tool` edges on the graph.
- **`Tool` schema modes** — `"signature"` (default, no LLM, inferred
  from type hints + docstring), `"llm"` (cheap Agent infers schema),
  `"hybrid"` (signature first, LLM fills gaps).
- **`NativeTool`** — `WEB_SEARCH`, `CODE_EXECUTION`, `FILE_SEARCH`,
  `COMPUTER_USE`, `GOOGLE_SEARCH`, `GOOGLE_MAPS`. Passed via
  `Agent(..., native_tools=[...])` shortcut.
- **`Plan` + `Step` + sentinels** — declared multi-step workflow with
  compile-time validation (`PlanCompileError`). Sentinels
  (`from_prev`, `from_start`, `from_step`, `from_parallel`) declare
  input flow without hand-threading. Typed hand-offs via `output=`.
  Conditional routing via `out.next: Literal[...]`. Parallel branches
  via `Step(parallel=True)`.
- **Plan checkpoint / resume** — `Plan(..., store=Store(db="…"),
  checkpoint_key="…", resume=True)` persists minimal state after every
  step. Failed runs resume from the failing step; done plans
  short-circuit to the cached `writes` bucket.
- **`Plan.to_dict` / `Plan.from_dict(registry=...)`** — topology
  round-trips through JSON; callables / Agents rebind via explicit
  registry on load.
- **`SupervisorEngine`** — human-in-the-loop REPL with tools, agent
  retry, and store access. Commands: `continue`, `retry <agent>:
  <feedback>`, `store <key>`, `<tool>(<args>)`. Accepts plain
  callables / Agents in `tools=` (same contract as `Agent`).
- **`HumanEngine`** — minimal HIL (prompt/answer, Pydantic field
  forms).
- **`Memory`** — four strategies (`auto` / `sliding` / `summary` /
  `none`), live-view semantics (`.text()` re-reads on every call),
  shareable across agents via `sources=`.
- **`Store`** — key-value blackboard, in-memory or SQLite WAL.
  Thread-safe. `to_text()` for `sources=` injection.
- **`Session`** — observability container with `EventLog` (8 event
  types), exporter fan-out, redaction hook, auto-populated
  `GraphSchema`, `usage_summary()`. Shortcut: `Session(console=True)`
  or `Agent(verbose=True)` installs a `ConsoleExporter`.
- **Exporters** — `CallbackExporter`, `ConsoleExporter`,
  `FilteredExporter`, `JsonFileExporter`, `StructuredLogExporter`,
  `OTelExporter`.
- **`GraphSchema`** — serialisable topology (JSON / YAML); auto-built
  from session-registered agents + `as_tool` edges.
- **Guards** — `Guard`, `ContentGuard`, `GuardChain`, `LLMGuard`.
  Input/output filtering returning `GuardAction` (allow / block /
  modify).
- **Evals** — `EvalSuite`, `EvalCase`, `EvalReport`; built-in checks
  `exact_match`, `contains`, `not_contains`, `min_length`,
  `max_length`, `llm_judge`.
- **`verify=`** — LLM-as-judge retry loop, placeable at Agent level
  (final output gate), tool level (`as_tool(verify=…)` — "Option B"),
  or Plan-step level.
- **Provider registry** — `LLMEngine.register_provider_alias` +
  `register_provider_rule` — extend routing at runtime (e.g. ship a
  new model family without editing framework code).
- **Parallelism as capability** — when the engine emits multiple tool
  calls in a turn, they execute concurrently via `asyncio.gather`. No
  `tool_choice="parallel"` knob. Declared parallel branches in `Plan`
  via `Step(parallel=True)`.
- **Factories** — `Agent.from_model(str, **kw)`,
  `Agent.from_engine(engine, **kw)`,
  `Agent.from_provider(name, tier="medium", **kw)`.

### Added — documentation

- **Claude Skill** at `lazybridge/skill_docs/` — ships with the
  package. 8 files (SKILL manifest + overview + 4 tiers + decision
  trees + reference + errors). Loadable by Claude Code / LLM
  assistants on demand.
- **Human MkDocs site** under `docs/` — 4 tier landing pages,
  per-topic guides, 9 decision trees ("when to use which"), quickstart,
  API reference, errors table.
- **Single-source documentation** — every topic authored once in
  `lazybridge/skill_docs/fragments/`; rendered to both skill and site
  by `python -m lazybridge.skill_docs._build`. CI drift-checked.
- **`llms.txt`** — minimal index pointing at the skill + site for
  LLM assistants that don't auto-load skills.

### Changed

- `LLMEngine.tool_choice` now accepts only `"auto"` / `"any"`. Legacy
  `"parallel"` raises `DeprecationWarning` and collapses to `"auto"`.
- `Envelope` is generic (`Envelope[T]`) so mypy / pyright narrow
  `payload` when `output=Model` is set. Untyped `Envelope` stays
  equivalent to `Envelope[Any]`.
- `Agent(verbose=True)` without a session creates a private
  `Session(console=True)` automatically.
- Nested Agent-as-tool: when Agent A with `session=s` has Agent B in
  `tools=[...]`, B inherits `s` and is registered on the graph. Events
  from nested agents flow to the outer `EventLog`; usage aggregates
  across the tree.

### Removed

- `LazyAgent`, `LazySession`, `LazyTool`, `LazyContext`, `LazyStore`,
  `LazyRouter`, `SupervisorAgent`, `HumanAgent` (superseded by
  `Agent` + engines).
- `lazybridge/gui/` — the GUI subsystem monkey-patched the removed
  classes; gone until a v1-native UI is rebuilt.
- `docs/course/*`, `lazy_wiki/*`, old `docs/*.md` — all described the
  pre-v1 API. Replaced by the new tier-organised docs.
- Pre-v1 examples (`framework_monitor.py`, 11 files under
  `examples/`). A v1-native research pipeline + framework_monitor port
  is pending.

### Fixed

- `Tool.run_sync` now drives async `func` to completion instead of
  returning a raw coroutine.
- `tool_choice="parallel"` no longer leaks to provider APIs as an
  unsupported value; `provider_tc` is always a valid enum.

---

## [0.6.0] and earlier

See git history for pre-v1 changelog entries. v1 is a breaking rewrite
and supersedes the 0.x series.
