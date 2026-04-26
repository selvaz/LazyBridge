# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.1] — unreleased — **structural split + MCP integration + HIL & evals to ext**

### Fixed — second-opinion audit (ChatGPT, merged into the same sweep)

- **Z1 — `Any` undefined in DeepSeek provider**
  (``lazybridge/core/providers/deepseek.py``).  ``Any`` was used at
  ``_ensure_json_word_in_prompt(... schema: Any = None)`` without
  being imported.  Worked at runtime under ``from __future__ import
  annotations`` (deferred evaluation) but broke ``typing.get_type_hints()``
  and tripped ``ruff F821``.  One-line ``from typing import Any`` fix.
- **Z2 — SQL guard upgraded from regex to AST validation**
  (``lazybridge/ext/stat_runtime/query.py``).  The existing
  ``_MUTATION_RE`` / ``_FILE_READER_RE`` / ``_PATH_LITERAL_RE`` regex
  pipeline was bypassable (multi-statement smuggling, schema-qualified
  ``main.read_parquet(...)``, comment injection across the keyword,
  unsupported-syntax fallthrough) and false-positive prone (forbidden
  tokens inside string literals).  New ``_validate_with_sqlglot``
  parses with the DuckDB dialect, rejects multi-statement and
  unparseable SQL, walks the tree for forbidden node types
  (``Insert`` / ``Drop`` / ``Pragma`` / ``Command`` / …), forbidden
  function calls (handles typed ``ReadParquet`` and bare
  ``Anonymous``, plus ``httpfs_*`` / ``s3_*`` / ``gcs_*`` /
  ``azure_*`` prefixes), and path-literal replacement scans
  (``FROM '/etc/passwd'`` and URI schemes).  Regex layer kept as
  defence-in-depth and as a graceful fallback when sqlglot is
  missing (with a one-time warning).  27 new red-team tests in
  ``tests/unit/ext/stat_runtime/test_query_red_team.py`` cover
  multi-statement, comment bypass, schema-qualified readers, nested
  CTE mutations, ``Command``-fallback rejections (``LOAD``, ``VACUUM``,
  ``CALL``), forbidden function-name prefixes, and URI-scheme paths.

### Fixed — high-severity audit findings (`claude/audit-architecture-competitors-TzBly`)

- **E1 — Plan parallel-band write atomicity**
  (``lazybridge/engines/plan.py``).  When one branch in a ``parallel=True``
  band failed, branches earlier in the declared order had already
  committed their ``writes=`` to ``kv`` and the external Store before
  the loop reached the failure.  Resume re-ran the whole band and
  partially-double-applied side-effects.  Post-fix the engine scans
  every branch for failure first and returns WITHOUT applying any
  writes when any branch errored — clean re-run on resume.
- **E2 — Plan checkpoint claim race**
  (``lazybridge/engines/plan.py:_claim_checkpoint``).  Two concurrent
  fresh runs on the same ``checkpoint_key`` could both pass the claim
  step (which only read the key without CAS) and both execute step 0
  before the loser saw a CAS failure on its first save.  Post-fix the
  claim now writes a ``status="claimed"`` placeholder via CAS up-front,
  so concurrent fresh runs collide BEFORE either has executed any
  step.  ``resume=True`` against a ``status="done"`` checkpoint still
  short-circuits to the cached ``kv`` (documented behaviour preserved).
- **E4 — Retryable exception classification**
  (``lazybridge/core/executor.py:_is_retryable``).  Previously fell
  back to a string scan over the exception message after the
  status-code check, so transient errors with empty / non-English /
  SDK-mangled messages (``RateLimitError("")``, ``APITimeoutError``
  in a Spanish locale) classified as non-retryable.  Post-fix the
  classifier walks the exception MRO and matches well-known transient
  class names from a frozenset (``RateLimitError``,
  ``APITimeoutError``, ``APIConnectionError``, ``ConnectionResetError``,
  …) before falling back to the string scan.
- **A1 — Sync ``__call__`` contextvars propagation**
  (``lazybridge/agent.py``, ``lazybridge/tools.py``).  When
  ``Agent.__call__`` (or ``Tool.run_sync``) was invoked inside an
  already-running event loop (FastAPI / Starlette / Jupyter), it
  spawned a worker-thread loop via ``asyncio.run`` with no context
  bridge — :mod:`contextvars` set in the outer loop (OTel spans,
  request IDs, structured-logging context) were invisible inside the
  agent.  Post-fix every sync façade copies the caller's context via
  ``contextvars.copy_context().run`` so observability state crosses
  the loop boundary.
- **A2 — ``fallback=`` / ``verify=`` agents inherit outer session**
  (``lazybridge/agent.py``).  Tool-list Agents were registered on the
  outer session and added to the graph; ``fallback=`` and ``verify=``
  Agents were skipped, so events they emitted (errors handled by the
  fallback, judge verdicts) recorded nowhere.  Post-fix both inherit
  the outer session and appear on the graph with ``fallback`` /
  ``verify`` edge labels (Agents that already carry their own session
  are not stomped).
- **Tests** — ``tests/unit/test_audit_high_findings.py`` (13 new
  tests, one per finding plus regression coverage).

### Added — extensions

- **`lazybridge.ext` regime** — formalised core-vs-ext split documented
  in `docs/guides/core-vs-ext.md`. Core is `__stability__ = "beta"`
  (public API stable in spirit); extensions ship at
  `__stability__ = "alpha"` by default and may break between minor
  releases. Promotion: alpha → beta → stable → core, after N minor
  releases without breakage.
- **`lazybridge.ext.planners`** — moved from `lazybridge.planners`
  (clean break, no shim). Exposes `make_planner` (DAG builder) and
  `make_blackboard_planner` (todo-list).
- **`lazybridge.ext.otel`** — moved `OTelExporter` here (clean break,
  no shim). Install with `pip install lazybridge[otel]`.
- **`lazybridge.ext.mcp`** *(new)* — Model Context Protocol integration
  at the tool boundary. `MCP.stdio` / `MCP.http` / `MCP.from_transport`
  build an `MCPServer` that drops into `Agent(tools=[server])` and
  expands to one `Tool` per MCP tool. Auto-namespacing
  (`{server}.{tool}`); `allow` / `deny` glob patterns; lazy connect;
  async-context-manager for explicit lifecycle. Tools-only phase 1;
  resources and prompts in a later phase. Install with
  `pip install lazybridge[mcp]`.

### Added — core

- **`from_parallel_all`** sentinel and `_FromParallelAll` aggregator —
  N-branch parallel-band synthesis. The join step receives a single
  envelope whose `task` and `payload` are a labelled-text join of every
  branch's output. PlanCompiler enforces: target step exists and is
  earlier in the plan, target is a `parallel=True` step, target is the
  FIRST member of its parallel band. Closes the architectural footgun
  where `from_parallel("name")` only ever forwarded one branch.
- **`Tool.from_schema(name, description, parameters, func)`** —
  construct a Tool with a pre-built JSON Schema instead of inferring
  from the Python signature. General-purpose; useful for OpenAPI /
  third-party tool registries beyond MCP.
- **`build_tool_map` provider expansion** — items in `tools=[...]` with
  `_is_lazy_tool_provider = True` and an `as_tools() -> list[Tool]`
  method are expanded into their constituent tools. Same dispatch
  semantics for plain functions, Agents, and tool collections.

### Removed (clean break)

- `lazybridge.planners` (moved to `lazybridge.ext.planners`; no shim).
- `lazybridge.OTelExporter` and `lazybridge.exporters.OTelExporter`
  (moved to `lazybridge.ext.otel.OTelExporter`; no shim).
- `OTelExporter` from the top-level `lazybridge.__all__`.
- `lazybridge.HumanEngine` and `lazybridge.SupervisorEngine` (moved to
  `lazybridge.ext.hil`; no shim). Internal modules
  `lazybridge.engines.human` / `lazybridge.engines.supervisor` are gone —
  use `lazybridge.ext.hil.{human,supervisor}` for their internal symbols
  (e.g. `_TerminalUI` in tests).
- `lazybridge.EvalSuite` / `EvalCase` / `EvalReport` / `llm_judge` /
  `contains` / `exact_match` / `min_length` / `max_length` /
  `not_contains` (moved to `lazybridge.ext.evals`; no shim). The runtime
  ``verify_with_retry`` helper used by ``Agent(verify=...)`` is now
  the private `lazybridge._verify` module — same contract, different
  import path.

### Fixed — audit-driven

- **`MCPServer._lock` lazy-init** — `asyncio.Lock()` is no longer
  constructed in `__init__` (sync context). Deferred to first async use
  to avoid the deprecation / runtime-loop coupling on Python 3.12+.
- **`PlanCompiler` first-member check** for `from_parallel_all` — pre-
  audit version validated only "is parallel"; post-audit version also
  validates that the named step is the FIRST member of its band, so
  pointing at a mid-band step now fails at compile time instead of
  silently truncating the aggregation.
- **`build_tool_map` collision warning** — fires once per collided
  name (was firing on every duplicate after the first); `stacklevel`
  raised from 2 to 4 so the warning points at the user's
  `Agent(tools=[...])` call instead of the internal loop.
- **`MCPServer` reuse contract** — class docstring now states
  explicitly that closure is terminal: a closed server cannot be
  reconnected. Behaviour was already correct; only documentation drifted.

### Fixed — review-driven (post-Sprint 2)

- ``docs/guides/core-vs-ext.md`` no longer contradicts itself —
  ``HumanEngine`` / ``SupervisorEngine`` are listed only under Ext
  (they were duplicated under Core, a leftover from before the move).
  LiteLLM bridge now annotated as "core provider; optional install".
- Version pinned to **1.0.1** in ``pyproject.toml`` and
  ``lazybridge.__version__`` (was 1.0.0; the CHANGELOG already
  documented this release as 1.0.1).
- ``lazybridge/ext/__init__.py`` registry now lists every shipped
  extension, including ``mcp``, ``hil``, ``evals``, ``otel``, and ``veo``.
- ``pyproject.toml [all]`` extra now includes the OTel and MCP
  dependencies (was missing them despite each having its own
  ``[otel]`` / ``[mcp]`` extra).
- ``Agent`` docstring rewritten so it doesn't imply ``HumanEngine`` /
  ``SupervisorEngine`` are top-level core imports; adds the
  ``from lazybridge.ext.hil import …`` path explicitly.
- ``Agent.as_tool`` docstring example fixed: previous version used
  ``Agent("…", system="…")`` which is not a valid constructor signature
  — replaced with ``Agent(engine=LLMEngine("…", system="…"))``.
- ``docs/recipes/mcp.md`` lazy-connect description tightened: connection
  happens at ``Agent(tools=[server])`` construction time (when
  ``build_tool_map`` calls ``as_tools``), not at first user query —
  fail-fast semantics are explicit now.

### Added — review-driven

- All six remaining extensions (``stat_runtime``, ``data_downloader``,
  ``quant_agent``, ``doc_skills``, ``read_docs``, ``veo``) now declare
  ``__stability__ = "alpha"`` and ``__lazybridge_min__``, matching the
  policy's "every extension declares maturity" rule.
- ``tests/unit/test_core_ext_boundary.py`` — two new architectural
  guards that fail CI on policy violation:
    1. **Core never imports from ``lazybridge.ext``.** AST-walks every
       ``.py`` under ``lazybridge/`` (excluding ``lazybridge/ext/``)
       and reports any ``import``/``from`` statement targeting the ext
       namespace.  No third-party import-linter dependency.
    2. **Every ext module declares ``__stability__`` and
       ``__lazybridge_min__``** with one of {alpha, beta, stable}.
       Imports each ext module and validates programmatically.

### Tests

- 21 audit-driven cases + 2 architectural guards = 23 new test
  cases this release. Total suite: 786 passed (was 746 before this
  version), 3 skipped, 0 regressions.

### Documentation

- `docs/guides/core-vs-ext.md` — new contributor / maintenance guide.
- `docs/recipes/mcp.md` — full guide with quickstart, multi-server,
  namespacing, allow/deny (clarified as fnmatch glob, not regex),
  lifecycle, testing without an SDK, pitfalls, API reference.
- `docs/recipes/orchestration-tools.md` — updated with
  `from_parallel_all` worked example, builder API workflow, and links
  to the in-box `lazybridge.ext.planners`.

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
  Supports provider fallback via `fallback=Agent(...)`: if the primary
  engine returns an error (rate limit, quota, outage), the request is
  transparently retried on the fallback agent.
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
- **`HumanEngine`** — HIL engine with terminal (`ui="terminal"`, default)
  and browser (`ui="web"`) modes. Terminal prompts inline; web mode
  launches a local HTTP server, opens a browser tab, and awaits
  form submission. Supports Pydantic field forms and `output=` schema.
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
