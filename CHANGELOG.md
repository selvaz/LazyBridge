# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed
- **`ext.planners` DAG builder — `add_step` now exposes `from_parallel_all`.**
  The incremental builder tool's `task_kind` annotation was
  `Literal["literal", "from_prev", "from_step", "from_parallel"]`, omitting
  `from_parallel_all` even though `StepSpec`, the step validator, `_resolve_task`,
  and `PLANNER_GUIDANCE` all already supported it — so the value the guidance
  steers the planner toward was not selectable through the generated tool schema.
  Added `"from_parallel_all"` to the `add_step` Literal and documented it in the
  tool docstring. Additive; no existing behaviour changes.

---

## [0.9.1] — 2026-05-28 — Store.items(prefix=) range scan

### Added
- **`Store.items(prefix=)`** — returns `(key, value)` pairs restricted to keys
  starting with `prefix` via a single indexed B-tree range scan
  (`WHERE key >= ? AND key < ?`). Sub-linear in total keyspace size;
  O(M) in the number of matching keys. The in-memory path filters under the
  store lock using `str.startswith`. Pass `prefix=None` (default) or `prefix=""`
  to iterate the full store. The `EncryptedStoreAdapter` delegates to the inner
  store and decrypts each returned value.
- **`_prefix_upper_bound(prefix)`** — private helper that computes the exclusive
  upper bound for the B-tree scan. Handles the U+10FFFF edge case by falling back
  to a Python-level `startswith` filter.

### Compatibility
Additive — no existing behaviour changed. LazyPulse 0.2.0 uses this method
to replace its O(N+1) `_scan_records` implementation.

---

## [0.9.0] — 2026-05-24 — lazytoolkit extraction (Phase 3: shims removed)

### Removed (breaking)
The lazy deprecation shims left behind by the 0.8 extraction are gone.
Import from `lazytools` directly.

- `lazybridge.ext.mcp` → use `lazytools.connectors.mcp` (`pip install 'lazytoolkit[mcp]'`).
- `lazybridge.ext.gateway` → use `lazytools.connectors.gateway`.
- `lazybridge.external_tools.read_docs` → use `lazytools.documents` (`pip install 'lazytoolkit[docs]'`).
- `lazybridge.external_tools.doc_skills` → use `lazytools.skills`.
- The whole `lazybridge.external_tools` namespace is deleted.

The old paths now raise `ModuleNotFoundError` instead of emitting a
`DeprecationWarning`. `lazybridge` still has no runtime dependency on
`lazytools`.

---

## [0.8.0] — 2026-05-24 — lazytoolkit extraction (Phases 0–2)

The concrete, dependency-carrying tools moved to the new sibling package
**`lazytoolkit`** (repo: `selvaz/LazyTools`). LazyBridge keeps only the minimal
runtime + framework extensions.

### Moved (lazy deprecation shims left behind; removed in 0.9)
- `lazybridge.ext.mcp` → `lazytools.connectors.mcp` (`pip install 'lazytoolkit[mcp]'`).
- `lazybridge.ext.gateway` → `lazytools.connectors.gateway`.
- `lazybridge.external_tools.read_docs` → `lazytools.documents`
  (`pip install 'lazytoolkit[docs]'`).
- `lazybridge.external_tools.doc_skills` → `lazytools.skills`.

Old import paths still work and emit a `DeprecationWarning` pointing at the new
location. The shims are lazy (PEP 562 `__getattr__`) so `import lazybridge`
never imports `lazytools` — `lazybridge` has no runtime dependency on the
toolkit. The `mcp` and `tools` extras were removed (use `lazytoolkit[mcp]` /
`lazytoolkit[docs]`).

---

## [0.7.9] — 2026-05-10 — simplification release

The headline change: **deletion-led simplification**.  The framework
had no users yet, so we ship breaking changes without deprecation
paths or shims.  Net public surface change: −1 in
``lazybridge.__all__`` (50 → 49), 5 deleted ``Agent.from_*`` class
methods, 9 silent-fallback paths converted to explicit errors, and
the entire ``report_builder`` subsystem extracted to its own repo.
Zero new public concept.

The single LLM-friendliness lever is consistency: one canonical
form per concept, errors always raise, no opt-in modes.

See ``docs/migrations/0.7-to-0.79.md`` for per-deletion before/after
codemod snippets.

### Breaking — extraction

- **``lazybridge.external_tools.report_builder`` extracted** to the
  sibling repo
  [``selvaz/LazyReport``](https://github.com/selvaz/LazyReport)
  (PyPI: ``lazybridge-reports``).  Every import path
  ``lazybridge.external_tools.report_builder.*`` is gone — replace
  with ``lazybridge_reports.*`` after installing the new package.
  The five optional extras ``[report]``, ``[report-charts]``,
  ``[report-citations]``, ``[report-fallback]``, ``[pdf]`` are
  gone from ``lazybridge``'s ``pyproject.toml``; their replacements
  live as ``lazybridge-reports[charts,citations,fallback,pdf]``.
  No shim — there is no fallback import path.
- **New optional extra**: ``[encryption]`` →
  ``cryptography>=42,<46`` for the new
  ``lazybridge.store.encryption.EncryptedStoreAdapter`` (Fernet
  at-rest encryption for ``Store`` values, with ``MultiFernet``
  key rotation).

### Breaking — deletions

- **5 ``Agent.from_*`` factories deleted**: ``from_model``,
  ``from_engine``, ``from_chain``, ``from_plan``, ``from_parallel``.
  All five were pure-alias forwarders (verified by audit).  Use the
  canonical ``Agent(engine=...)`` ctor or the kept-because-non-trivial
  factories (``Agent.chain``, ``Agent.parallel``,
  ``Agent.from_provider``).
- **3 config dataclasses deleted**: ``AgentRuntimeConfig``,
  ``ResilienceConfig``, ``ObservabilityConfig``.  These were
  wrapper-of-flat-kwargs configs whose only behaviour was a
  ``flat kwarg > config object > default`` precedence merge that
  required a private ``_UNSET`` sentinel on every kwarg.  The
  precedence game and ``_UNSET`` are gone with them.  ``CacheConfig``
  is **kept** — it carries real semantic value (``enabled`` /
  ``ttl``) consumed by ``LLMEngine``.
- **``mode="auto"`` graceful-fallback ladder removed** from
  ``Tool`` / ``tool()``.  Both default to ``mode="signature"`` now;
  pass ``mode="hybrid"`` or ``mode="llm"`` plus ``schema_llm=`` to
  opt into LLM-driven schema generation.  Passing
  ``mode="auto"`` raises ``ValueError``.
- **``_ParallelAgent`` renamed to ``ParallelAgent``** and its return
  contract changed.  ``ParallelAgent.__call__`` and ``run()`` now
  return ONE ``Envelope`` whose ``.payload`` is the labelled-text
  join across every branch (with transitive cost rollup in
  ``metadata.nested_*`` and first-error short-circuit in
  ``.error``) — restoring the framework invariant that every Agent
  returns ``Envelope``.  For typed per-branch ``list[Envelope]``,
  call the new ``run_branches(task)`` async helper.
- **``wrap_tool`` made private** (``_wrap_tool``).  Use the public
  ``tool(...)`` factory instead.
- **``LLMEngine(tool_choice="parallel")`` raises ``ValueError``**
  (was a 0.7-era ``DeprecationWarning`` that downgraded to ``"auto"``).\n  Concurrent tool execution is the default and not configurable.
- **``Old doc/`` directory deleted** (1.2 MB, zero references).
- **``pythonpath = ["lazybridge"]`` removed** from ``pyproject.toml``
  (unused).

### Breaking — silent fallbacks → explicit errors

- ``from_step("typo")`` / ``from_parallel("typo")`` no longer warn +
  fall back to the start envelope — they raise ``PlanRuntimeError``
  with the actual step history and a typo-aware "Did you mean?" hint.
- ``LLMEngine`` with an unknown model raises ``ValueError`` instead of
  silently routing to Anthropic.  Set
  ``LLMEngine.set_default_provider("...")`` for the legacy behaviour.
- MCP server emitting a non-``object`` ``inputSchema`` raises
  ``ValueError`` (was silently coerced to an empty parameter set).
- ``Envelope.text()`` on a non-JSON-serialisable payload raises
  ``TypeError`` (was ``str(payload)`` fallback).
- ``Memory(summarizer_timeout < 5.0)`` warns at construction
  (timeout almost always fires for typical summariser shapes).
- ``BaseProvider._resolve_model`` raises ``ValueError`` when nothing
  is configured (was empty string fallback).
- ``Agent(engine=<non-LLM>)`` requires an explicit ``name=`` (was
  silently named ``"agent"`` and collided when used as a tool).
- ``Agent(model=..., engine=<non-LLM>)`` raises ``ValueError`` (was
  silently dropped).

### Added

- **``lazybridge.matrix``** — declarative provider-capability lookup.
  ``provider_capabilities()`` and ``native_tool_support()`` aggregate
  the per-provider ``ClassVar`` flags into a single typed dict for
  docs / introspection / capability-aware error messages.
- **``BaseProvider`` capability ``ClassVar`` flags**:
  ``supports_streaming`` / ``supports_structured_output`` /
  ``supports_thinking``.  Subclasses override when a backend doesn't.
- **Standard error-message format** — every ``PlanCompileError`` /
  ``PlanRuntimeError`` / ``UnsupportedFeatureError`` follows::

      Step '<name>' (#<pos>) — <slot>=<sentinel> <reason>.
        <context, e.g. ``Defined steps: [...].``>
        Did you mean '<close>'?  (when applicable)
        Fix: <concrete action>.

- **OTel ``gen_ai.agent.nesting_level`` attribute** on agent spans —
  dashboards filtering on ``=0`` get clean root-only views.
- **``Session.emit`` exporter-exception dedup** — warn-once per
  ``(exporter class, exception class)`` pair with a count of
  suppressed identical failures.
- **``test_public_api_snapshot.py``** — pins ``lazybridge.__all__``
  and locks the deleted-in-0.7.9 names as permanently gone.

### Fixed (bug fixes from Phase 1)

- B1: DeepSeek provider — defensively rebuild ``params['messages']``
  rather than mutating in place.
- B2: Anthropic ``_compute_cost`` accepts ``cached_input_tokens=0``
  and applies the standard 10% cache-read rate; cost telemetry was
  over-counted on cached calls pre-fix.
- B4: Anthropic provider warns on URL-source ``AudioContent`` (was
  silently dropped).
- B6: Plan compiler typo-aware "Did you mean?" suggestions on
  ``from_step`` / ``from_agent`` / ``from_memory`` unknown-target
  errors.
- B7: Plan serialisation raises ``ValueError`` on unknown sentinel
  ``kind`` (was silently fallback to ``from_prev``).
- B8: OTel exporter logs SDK exceptions at WARNING (was swallowed).
- B9: Blackboard planner closure state resets per ``run()`` call
  (was leaking between successive runs).
- B10: Plan compiler rejects sentinels referencing auto-named
  ``_anon_<id>`` steps (LLMs cannot meaningfully produce that name).
- B11: Plan resume replays Store sidecar writes from the checkpoint
  ``kv`` so external consumers see complete state after a crash in
  the checkpoint→Store-write window.
- I5: Anthropic adaptive-thinking warning corrected for Opus 4.7
  (pre-fix said "use 'effort'" but Opus 4.7 only accepts ``display``).

### Documentation

- ``SKILL.md`` rewritten for the 0.7.9 surface; canonical Plan
  block, default-model fallback advice via ``from_provider(tier=...)``,
  anti-pattern entries for every deletion.
- ``docs/migrations/0.7-to-0.79.md`` (new) — per-deletion before/after
  snippets and a TL;DR table.
- ``docs/reference/configs.md`` rewritten — only ``CacheConfig``
  remains documented; rest of file explains the deletion.
- ``docs/reference/engines.md`` adds the ``thinking=`` knob.
- ``docs/reference/providers.md`` adds the ``stop_reason``
  normalisation table (Google ``MAX_TOKENS`` mapping fix).
- ``docs/guides/mid/parallel.md`` rewritten for the new
  single-Envelope return contract.
- the MCP guide example now shows ``allow=`` filtering as best
  practice.
- ``examples/verify_judge_loop.py`` (new), ``examples/guardrails_demo.py``
  (new), env preflight in ``examples/daily_news_report.py``.

### Tooling

- ``lazybridge.skill_docs._build`` recovered + wired into the
  ``test.yml`` typecheck job (drift gate that asserts every
  public symbol in ``__all__`` is mentioned in SKILL.md).
- ``docs.yml`` asserts ``site/llms.txt`` and ``site/llms-full.txt``
  are non-empty (≥1 KB) after ``mkdocs build --strict``.
- Top-level ``permissions: contents: read`` added to
  ``.github/workflows/test.yml`` (least-privilege baseline).

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

### Hardening

- **`MCP.stdio()` now warns on unrestricted tool surface.**  When
  both ``allow=`` and ``deny=`` are omitted a one-shot ``UserWarning``
  reminds the caller that every tool the subprocess advertises will
  reach the LLM.  Trust model is unchanged (stdio is still
  audit-on-init, not deny-by-default like ``MCP.http``); silence the
  warning by passing ``allow=["*"]`` once you've audited the surface,
  or restrict it with a glob.  See ``SECURITY.md`` for the full
  guidance.
- **CI now enforces skill-doc drift.**  ``test.yml`` runs
  ``python -m lazybridge.skill_docs._build --check`` in the typecheck
  job: a fragment edit without a re-render now fails the PR instead
  of slipping through (this was the documented contract; CI just
  hadn't been wired).
- **`mkdocs build --strict`.**  A missing nav target or unresolved
  cross-reference in ``docs/`` now fails the docs workflow instead of
  shipping an empty page.

### Documentation

- **README** — added an "alpha (0.7.x)" status callout up top, and
  expanded the *Full* tier sentinel list to cover all five exports
  (``from_prev`` / ``from_start`` / ``from_step`` / ``from_parallel``
  / ``from_parallel_all``).
- **SECURITY.md** — new "MCP Servers — Tool Surface Audit" section
  documenting the deny-by-default contract on ``MCP.http`` and the
  audit-on-init warning on ``MCP.stdio``.
- **API reference** — ``_UNSET`` sentinels in generated signatures
  now have an explicit explanation at the top of the reference page,
  including the ``LLMEngine.stream_idle_timeout`` semantics.
- **Skill docs reference grouping fixed** —
  ``from_parallel_all``, ``GuardError``, and ``EventExporter`` are
  now classified under their proper categories instead of falling
  through to "Core types".
- **CHANGELOG hygiene** — the second ``[Unreleased]`` section was
  renamed to ``[0.7.0 — short-term audit hardening]`` so the file no
  longer carries two unreleased headers.

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
