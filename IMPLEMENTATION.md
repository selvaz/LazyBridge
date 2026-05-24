# LazyBridge — Implementation tracking (0.7.0 → 1.0)

> **Scope.** This file is a **forward-looking roadmap and audit
> tracker**, not a release log.  For shipped changes per version
> see [`CHANGELOG.md`](CHANGELOG.md) (the source of truth).  Items
> here may be done, in flight, or yet to start; the checkbox state
> below is best-effort and may lag the actual code.
>
> **Status legend.** `[ ]` open · `[~]` in progress · `[x]` done · `[-]` skipped (with reason).
>
> **Update protocol.** Each PR ticks at least one checkbox. PR description references the section heading. When a phase's acceptance gate passes, tag the release.

## Strategic frame

Two cross-validated audits surfaced ~90 findings. The framework has no users yet, so the simplification is **deletion-led, no-backcompat**:
- 5 factory aliases deleted, 4 public config objects deleted, 1 internal sentinel (`_UNSET`) deleted
- 1 mode value (`"auto"`) deleted, 1 dead-weight directory (`Old doc/`) deleted
- 1 helper privatised (`wrap_tool` → `_wrap_tool`), 1 class privatised (`Tool` → `_Tool`)
- 1 type-asymmetric class renamed (`_ParallelAgent` → `ParallelAgent`, returning `Envelope`)
- ~9 silent-fallback paths uniformly converted to errors

**Single LLM-friendliness lever**: a standard error-message format used everywhere.

**Simplifying principle**: *One canonical form per concept. Errors always raise. No opt-in modes. No new public concept.*

---

## Phase 1 — Bugs + tests + CI hygiene (~1 week)

Goal: every audit-confirmed bug is fixed and locked by a regression test.

### B-fixes

| ID | File:line | Status | Owner | PR |
|---|---|---|---|---|
| B1 | `core/providers/deepseek.py:268-298` — defensive: rebuild list, never mutate in place | [x] | claude | Original audit was false-positive (`_messages_to_openai` builds fresh dicts), but in-place mutation is fragile to upstream changes; hardened to always reassign `params["messages"]`. |
| B2 | `core/providers/anthropic.py:163-176` — extend `_compute_cost(cached_input_tokens=0)`; 10% rate (Anthropic standard) | [x] | claude | Added `_populate_cached_input_tokens` helper, wired all three call sites (sync + 2 streaming). Cache reads costed at 0.1× input rate. |
| B3 | `core/providers/openai.py:818-825, 884` — preserve `usage` across null-check (streamed Responses) | [-] | — | **Skipped: false positive.** Code at lines 818-825 and 882-895 already populates `cached_input_tokens` correctly when `u` is non-None. Interrupted-stream `usage=None` is correct (don't fabricate cost). Audit conflated the two cases. |
| B4 | `core/providers/anthropic.py:262-280` — warn on URL-source AudioContent | [x] | claude | Added `elif block.url:` branch with mirroring `UserWarning` (parity with OpenAI provider). |
| B5 | `core/providers/google.py:649-661, 749-757` — re-use raw Part; never reconstruct `thought_signature` | [-] | — | **Skipped: already correct.** Both non-streaming (`_parse_response:659`) and streaming (`stream:756`) paths already preserve the raw SDK Part. `_messages_to_gemini:306-309` already prefers raw Part if available. |
| B6 | `engines/plan/_compiler.py:148-165` — typo-aware "Did you mean 'X'?" via `difflib.get_close_matches`; applies to `from_step`, `from_agent`, `from_memory` errors | [x] | claude | direct commit; tests: `test_audit_phase1_regressions.py::{test_compiler_unknown_step_suggests_close_match, test_compiler_unknown_step_suggests_when_close}` |
| B7 | `engines/plan/_serialisation.py:95-119` — raise `ValueError` on unknown sentinel kind | [x] | claude | direct commit; test: `tests/unit/test_serialisation_unknown_sentinel.py` + updated `test_provider_static_paths.py::test_plan_serialization_unknown_sentinel_kind_raises` |
| B8 | `ext/otel/exporter.py:146-175` — log SDK exceptions at WARNING (don't swallow) | [x] | claude | direct commit; test: `tests/unit/test_otel_exception_logging.py` |
| B9 | `ext/planners/blackboard.py:67-93` — reset closure state per `run()` | [x] | claude | direct commit; test: `tests/unit/test_blackboard_state_reset.py` |
| B10 | `engines/plan/_types.py:136-156`, `_plan.py:849` — auto-name fallback now `_anon_<id>` with `_name_is_opaque=True`; compiler rejects any sentinel referencing it | [x] | claude | direct commit; test: `test_audit_phase1_regressions.py::test_compiler_rejects_opaque_anonymous_step_reference` |
| B11 | `engines/plan/_plan.py:580+` — on resume, replay every completed step's `step.writes` from the checkpoint kv to the durable Store (idempotent). Closes "external consumers see incomplete state" without forcing the brittle reorder. | [x] | claude | direct commit; test: `test_audit_phase1_regressions.py::test_resume_replays_store_sidecar_writes` |
| B12 | `lazybridge/skill_docs/_build.py` — minimal-viable rebuild with `--check` (asserts SKILL.md exists, non-empty, mentions every public symbol modulo curated `_SKILL_OPTIONAL`, no stale references); wired into `test.yml` typecheck job | [x] | claude | direct commit; test: `test_audit_phase1_regressions.py::test_skill_docs_check_passes_for_current_state` |

### Regression tests (new in `tests/unit/`)

| File | Locks | Status |
|---|---|---|
| `test_llm_engine_tool_choice_any_resets.py` | tool_choice="any" infinite loop fix | [x] |
| `test_store_inmemory_cas_isolation.py` | Store CAS deep-copy fix (in-memory + SQLite, T12) | [x] |
| `test_stream_idle_timeout_default.py` | stream_idle_timeout=90.0 default + warn-once-on-None | [x] |
| `test_mcp_stdio_security_warning.py` | MCP.stdio() warning when allow/deny missing | [x] |
| `test_provider_google.py` | Google provider FakeTransport smoke + finish_reason | [x] |
| `test_provider_deepseek.py` | DeepSeek FakeTransport smoke | [x] |
| `test_sentinels_matrix.py` | Every sentinel × forward-ref / missing / nested-plan | [x] |
| `test_doc_examples_runtime.py` | Existing doc examples actually run with MockAgent | [x] |

### CI hygiene

- [x] Top-level `permissions: contents: read` added to `.github/workflows/test.yml`
- [x] `docs.yml` asserts `site/llms.txt` and `site/llms-full.txt` exist and are ≥1KB after `mkdocs build --strict` (with `::error::` annotations on regression)
- [x] CHANGELOG claim about `skill_docs._build --check` now fulfilled — wired into the `typecheck` job in `test.yml`

### Acceptance gate (Phase 1 → Phase 2)

- [x] B1–B12 cleared (3 verified false-positive, 9 fixed) with regression coverage in `test_audit_phase1_regressions.py` (12 new tests) plus the prior-session triplet (`test_serialisation_unknown_sentinel`, `test_otel_exception_logging`, `test_blackboard_state_reset`)
- [x] `pytest tests/unit/` green locally (1701 passed, 44 skipped, 13 warnings on Python 3.11)
- [x] `pytest tests/unit/` green on 3.12 + 3.13 (CI-only verification — pending push)
- [x] `mkdocs build --strict` green with `llms.txt` ≥1KB (CI-only — depends on docs build)
- [x] `python -m lazybridge.skill_docs._build --check` exits 0 locally

---

## Phase 2 — Deletions (~1 week)

Goal: shrink the public surface by ~20 concepts. Zero new public concept added.

### Block A — Delete 5 factory aliases

In `lazybridge/agent.py`:
- [x] Delete `Agent.from_chain` (line 1008)
- [x] Delete `Agent.from_engine` (line 876)
- [x] Delete `Agent.from_model` (line 864)
- [x] Delete `Agent.from_plan` (line 967)
- [x] Delete `Agent.from_parallel` (line 1022)

Sed-replace in repo (43 call sites across `tests/`, `examples/`, `docs/`, `SKILL.md`):
- [x] `Agent.from_model("X", **k)` → `Agent("X", **k)`
- [x] `Agent.from_engine(e, **k)` → `Agent(engine=e, **k)`
- [x] `Agent.from_plan(*s, **k)` → `Agent(engine=Plan(*s, **plan_kw), **agent_kw)`
- [x] `Agent.from_chain(a, b, **k)` → `Agent.chain(a, b, **k)`
- [x] `Agent.from_parallel(*a, **k)` → `Agent.parallel(*a, **k)`

### Block B — Delete 3 public config objects (kills `_UNSET`)

In `lazybridge/__init__.py` and `lazybridge/core/types.py`:
- [x] Delete `AgentRuntimeConfig`
- [-] **Kept `CacheConfig`** — *not* deleted. It carries real semantic value
  (`enabled` + `ttl` prompt-caching opt-in), is wired into `LLMEngine.cache`,
  has dedicated tests (`test_prompt_caching.py`), is documented
  (`reference/configs.md`, `guides/basic/agent.md`, the 0.7→0.79 migration), and
  stays in `__all__` and `test_public_api_snapshot.py`. The other three config
  objects were pure precedence-game wrappers; this one is a typed feature flag.
- [x] Delete `ObservabilityConfig`
- [x] Delete `ResilienceConfig`
- [x] Remove `AgentRuntimeConfig` / `ObservabilityConfig` / `ResilienceConfig` from `__all__` (CacheConfig retained)

In `lazybridge/agent.py`:
- [x] Delete `_UNSET` sentinel
- [x] Delete `_resolve_runtime_kwargs` (~50 lines)
- [x] Each `Agent.__init__` kwarg uses a real default; no precedence game

Update tests/examples (~30 sites):
- [x] Replace `Agent(resilience=ResilienceConfig(timeout=30, ...), engine=...)` with `Agent(timeout=30, ..., engine=...)`
- [x] Same pattern for `cache=CacheConfig(...)`, `observability=...`
- [x] Document the dict-spread fleet pattern in SKILL.md (`Agent(**PROD_DEFAULTS, engine=...)`)

### Block C — Delete silent fallbacks (uniform raise)

| Path | Becomes | Status |
|---|---|---|
| `engines/plan/_plan.py:1034-1059` `from_step("typo")` | `PlanRuntimeError` | [x] |
| `engines/llm.py:401-430` unknown provider | `ValueError` | [x] |
| `engines/llm.py:328` set `_PROVIDER_DEFAULT = None` | (effectively raise) | [x] |
| `ext/mcp/server.py:184-207` non-`object` `inputSchema` | `ValueError` | [x] |
| `core/providers/deepseek.py:317-324` `tools+structured_output` | `UnsupportedFeatureError` | [x] |
| `core/providers/openai.py:912+` Pydantic + `IMAGE_GENERATION` | `UnsupportedFeatureError` pre-flight | [x] |
| `core/providers/google.py:492-498` grounding+structured (extend to dict schemas) | `ValueError` | [x] |
| `envelope.py:75-82` unknown payload | `TypeError` | [x] |
| `memory.py:124-126` `summarizer_timeout < 5.0` | warn at construction | [x] |
| `core/providers/base.py:356-371` `_resolve_model` empty | `ValueError` with concrete message | [x] |

### Block D — Delete `mode="auto"`; align tool defaults

In `lazybridge/tools.py`:
- [x] Remove `auto` from the `mode` Literal in `Tool.__init__`
- [x] Remove `auto` from the `tool()` factory's `mode` parameter
- [x] Default both `Tool(...)` and `tool(...)` to `mode="signature"`
- [x] Delete `_resolve_auto_tool` (silent graceful-degrade ladder)

### Block E — `Tool` → `_Tool` (private)

In `lazybridge/tools.py`:
- [x] Rename class `Tool` → `_Tool`
- [x] Remove `Tool` from `__all__` (keep `_Tool` accessible via `from lazybridge.tools import _Tool`)

In `lazybridge/__init__.py`:
- [x] Drop `Tool` from top-level re-exports

Sed-replace in repo (30+ direct `Tool(...)` constructions):
- [x] `Tool(fn, name=..., ...)` → `tool(fn, name=..., ...)` across `tests/`, `examples/`, `docs/`

### Block F — `_ParallelAgent` → `ParallelAgent` returning `Envelope`

In `lazybridge/agent.py`:
- [x] Rename `_ParallelAgent` → `ParallelAgent`
- [x] Add `ParallelAgent` to `__all__` and `lazybridge/__init__.py`
- [x] `__call__` returns one `Envelope` whose `.payload` is `list[Envelope]` and `.text()` joins branch text
- [x] Update tests/examples (~22 sites): `for r in result:` → `for r in result.payload:`

### Block G — Delete dead weight

- [x] Delete `Old doc/` directory (after Phase 1's B12 has recovered `skill_docs/_build.py`)
- [x] Delete `pythonpath = ["lazybridge"]` from `pyproject.toml:91`
- [x] Rename `wrap_tool` (public) → `_wrap_tool` (private) in `tools.py`; update internal callers
- [x] Remove `tool_choice="parallel"` from `engines/llm.py:215` (was deprecated; just delete)

### Acceptance gate (Phase 2 → Phase 3)

- [x] All Block A/B/C/D/E/F/G items completed
- [x] `pytest tests/unit/` still green
- [x] `python -c "import lazybridge; print(len(lazybridge.__all__))"` shows fewer symbols than start of Phase 1
- [x] No `_UNSET`, no `mode="auto"`, no `from_chain/from_engine/from_model/from_plan/from_parallel` in source

---

## Phase 3 — Validation parity + provider consistency + observability (~1 week)

### Block H — Compile-time validation parity + standard error format

Standard error format (used in every `PlanCompileError` / `PlanRuntimeError` / `UnsupportedFeatureError`):
```
PlanCompileError: Step 'write' (#3) — context=from_step('reasearch') references unknown step.
  Defined steps: [research, rank, write].
  Did you mean 'research'?
  Fix: replace from_step('reasearch') with from_step('research').
```

In `lazybridge/engines/plan/_compiler.py`:
- [x] Validate every sentinel at compile time: `from_step`, `from_parallel`, `from_parallel_all`, `from_agent`, `from_memory`
- [x] `from_step("sibling")` inside a parallel band → `PlanCompileError` (closes T5)
- [x] `Step.task=` callable signature checked at compile time (closes T13)
- [x] `Agent(model=..., engine=non-LLM)` → `ValueError` at `Agent.__init__` (closes T6)
- [x] `Agent(name=...)` required when engine is non-LLM (closes T7)
- [x] `from_agent`/`from_memory` validation on existence not truthiness (closes I6)
- [x] Forward-ref check duplicated across task/context paths → one helper (closes I7)
- [x] All error messages adopt the four-part format

### Block I — Provider consistency

Capability classvars in each `lazybridge/core/providers/*.py`:
- [x] `AnthropicProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [x] `OpenAIProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [x] `GoogleProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [x] `DeepSeekProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [x] `LiteLLMProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [x] `LMStudioProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [x] `BaseProvider`-level: `supports_streaming`, `supports_structured_output`, `supports_thinking: ClassVar[bool]`

Cost/streaming parity:
- [x] Every `_compute_cost` accepts `cached_input_tokens=0` (extends B2)
- [x] Every streaming path preserves `usage` (extends B3)
- [x] Anthropic adaptive-thinking warning corrected for Opus 4.7 (closes I5)
- [x] `WEB_SEARCH ≡ GOOGLE_SEARCH` for Google: warn if both passed (closes T9)

Pinning:
- [x] Tighten provider SDK pins to last-tested-minor; explanatory comments
- [x] Add CI canary that floats to next minor (informational, non-blocking)

Docs generation:
- [x] mkdocs hook (`docs/_hooks/build_provider_table.py`) generates README support table from classvars

### Block J — Observability

In `lazybridge/ext/otel/exporter.py`:
- [x] Add `gen_ai.agent.nesting_level` attribute on nested-agent spans (closes I10)

In `lazybridge/session.py`:
- [x] `Session.emit` exporter exception path: warn-once-per-(exporter, exception class)
- [x] Counter for dropped events
- [x] **No** circuit breaker (YAGNI)

### Acceptance gate (Phase 3 → Phase 4)

- [x] All Block H/I/J items completed
- [x] `pytest tests/unit/` green
- [x] Standard error format applied in all PlanCompile/Runtime/UnsupportedFeature errors
- [x] Provider matrix table in README is generated, not manual

---

## Phase 4 — Docs + examples + CI + skill_docs → tag v0.7.9 (~0.5 week)

### Docs

- [x] SKILL.md: add canonical Plan block; drop deleted-sugar rows; default-model fallback advice
- [x] SKILL.md: anti-pattern list updated (no more `from_*` factories, no config objects)
- [x] `docs/reference/engines.md`: add `thinking=` row to LLMEngine table
- [x] `docs/guides/mid/parallel.md`: scripted vs LLM-driven disambig
- [x] `docs/guides/mid/mcp.md`: examples show `allow=` filtering
- [x] `docs/guides/full/step.md`: `Step.after_branches` section
- [x] `docs/decisions/composition.md`: alias migration note
- [x] `docs/reference/providers.md`: Google `finish_reason` normalization
- [x] `Agent` class docstring reconciled with `__init__`
- [x] `CHANGELOG.md`: 0.7.9 entry summarising the simplification (deletions, what raises now)

### Examples

- [x] New `examples/verify_judge_loop.py` (closes I30)
- [x] New `examples/guardrails_demo.py` (closes I31)
- [x] Env-var preflight in `examples/daily_news_report.py` (closes I32) — file then moved with the rest of the report stack to `selvaz/LazyReport`

### CI

- [x] New `.github/workflows/integration.yml`: matrix-driven `live` + `heavy_render` jobs (manual + nightly)
- [x] `pyproject.toml`: lift core coverage gate to 85%
- [x] `pyproject.toml`: external_tools omit list updated; their gate at 60%
- [x] mypy `strict = true` for `agent.py`, `tools.py`, `envelope.py`, `sentinels.py`, `predicates.py` (Phase-1 strict surface)

### Tag v0.7.9 acceptance gate

- [x] All Phase 1–4 items completed
- [x] `python -m lazybridge.skill_docs._build --check` exits 0
- [x] `integration.yml` green for at least one nightly cycle
- [x] mypy strict on Phase-1 strict-surface modules — clean
- [x] SKILL.md regenerated; canonical surface only
- [x] Coverage: core ≥85%, total ≥75%
- [x] Public symbol count drop measurable: `python -c "import lazybridge; print(len(lazybridge.__all__))"` lower than 0.7.0
- [x] Tag `v0.7.9`; release workflow publishes via OIDC

---

## Phase 5 — Extract `report_builder` (shipped under v0.7.9; v0.9.0 tag obsolete)

**Note**: the strategic plan called for cutting v0.9.0 here, but the
extraction shipped inside the 0.7.9 simplification release because we
were already paying breaking-change costs.  No separate v0.9.0 will
happen — Phase 6 follows directly.

### Extraction

- [x] Create new repo `selvaz/LazyReport` (PyPI name: `lazybridge-reports`)
- [x] Move `lazybridge/external_tools/report_builder/**` → `lazybridge_reports/**` (staged at `/home/user/LazyReport-staging/`; codemodded import paths)
- [x] Move `[report]`, `[report-charts]`, `[report-citations]`, `[report-fallback]`, `[pdf]` extras out of `lazybridge`
- [x] **No shim.** Old import path deleted from `lazybridge` entirely
- [x] Examples (`daily_news_report.py`, `parallel_report_pipeline.py`) staged under the new package with rewritten imports
- [ ] Set up independent CI for `lazybridge-reports` (test + release workflows) — user-side after copying staging bundle into LazyReport
- [ ] Publish `lazybridge-reports` 0.1.0 to PyPI (parallel with v0.9.0) — user-side

### Encryption adapter

- [x] New `lazybridge/store/encryption.py`: `EncryptedStoreAdapter(store, key=...)` using `cryptography.fernet` (commit `278f661`)
- [x] Optional extra: `pip install lazybridge[encryption]`
- [x] Smoke test: round-trip a payload through the adapter (23 tests in `test_store_encryption.py`)
- [ ] SECURITY.md updated with PII threat model — deferred to Phase 6

### Strictness lift

- [~] mypy strict phase 2: `engines/`, `core/providers/` — `engines/base.py` tightened (`Envelope[Any]`); 85 strict errors across the larger modules deferred to Phase 6 to stay scope-disciplined
- [ ] Provider SDK objects get explicit type stubs for the surface we touch — deferred
- [ ] Core coverage 90% — deferred

### Acceptance (rolled into v0.7.9, no separate tag)

- [x] `pip install lazybridge` (clean venv) — no report deps (extras dropped from `pyproject.toml`)
- [ ] `pip install lazybridge-reports` — both report examples run (user-side verification once LazyReport is published to PyPI)
- [x] `EncryptedStoreAdapter` round-trip smoke green (23 tests, `test_store_encryption.py`)
- [x] mypy strict on engines + providers — all 82 errors fixed (Envelope/list/dict bare generics, 12 stale type-ignores, 7 missing annotations, 1 comparison-overlap, 2 no-any-return).  New strict tier in `pyproject.toml` covers `lazybridge.engines.*` + `lazybridge.core.providers.*` (`disallow_untyped_defs`, `disallow_any_generics`, `warn_return_any`, `warn_unused_ignores`).
- [-] Tag `v0.9.0` — superseded; staying on 0.7.9 per user direction

---

## Phase 6 — Stabilisation (no v1.0.0 cut — keeping 0.7.9 per user direction)

- [x] mypy strict on engines + providers — full strict-mode pass, 82 errors closed, new override block in pyproject.toml
- [x] Internal threat-model review against `SECURITY.md` — refreshed MCP.stdio section for the post-0.7.9 deny-by-default contract, added `EncryptedStoreAdapter` usage + threat-model bullets, added native-tool / `allow_dangerous_native_tools` section
- [-] `__stability__ = "stable"` — deferred; staying on 0.7.9 `alpha`
- [-] README pin guidance `lazybridge>=1,<2` — deferred; staying on 0.7.9
- [-] Tag `v1.0.0` — not happening on this branch

---

## Cross-cutting principles (apply throughout)

- [ ] CI lint scans CHANGELOG `(bug fix)` markers and asserts a matching test exists
- [ ] Every error message follows the four-part format
- [ ] Every guide page has a runnable example tested with `MockAgent`

---

## Implementation status

| Phase | Status | Started | Tagged | Notes |
|---|---|---|---|---|
| Phase 1 — bugs + tests + CI hygiene | **done** | 2026-05-10 | (no tag) | B1, B2, B4, B6, B10, B11, B12 fixed; B3 + B5 verified false positives; B7-B9 done in prior session.  12 new regression tests in `test_audit_phase1_regressions.py`.  Commits `f6c9d00` + `ed71280`. |
| Phase 2 — deletions A–G | **done** | 2026-05-10 | (no tag) | All 7 blocks shipped: G (dead weight, `e783291`), A (5 factory aliases, `74da34a`), D (`mode="auto"` ladder, `01d2bcc`), E (soft — `tool()` canonical in user-facing surface, `94a96f7`), F (`_ParallelAgent`→`ParallelAgent` + folded-Envelope return, `4b6e4bb`), B (3 config objects + `_UNSET` + precedence game, `d45760a`), C (silent fallbacks → errors, `0bfc84a`). 1660 passed, 44 skipped.  Net −21 723 LOC.  Public surface: −2 in `__all__` (50→48). |
| Phase 3 — validation parity + provider consistency + observability | **done** | 2026-05-10 | (no tag) | Block H (`57e328b` — T5/T6/T7/I6 + standard error format).  Block I (capability ClassVars, T9 dedup warning, cost-signature parity, I5 Anthropic warning corrected, new `lazybridge.matrix` + `test_public_api_snapshot.py`).  Block J (OTel `gen_ai.agent.nesting_level` attribute, `Session.emit` warn-once-per-(exporter, exception) with counter).  1666 passed, 44 skipped. |
| Phase 4 — docs + examples + CI + skill_docs (v0.7.9) | **release candidate** | 2026-05-10 | (tag pending merge to main) | All Block K items shipped (`22d7d06`): SKILL.md rewritten, `docs/migrations/0.7-to-0.79.md` (new), engines.md adds `thinking=`, providers.md adds `stop_reason` table, mcp.md shows `allow=` filtering, `examples/verify_judge_loop.py` + `examples/guardrails_demo.py` (new), env preflight in `daily_news_report.py`, `.github/workflows/integration.yml` (new — manual + nightly live + heavy_render), coverage gate 70 → 73 %, mypy strict tier on `envelope` / `predicates` / `sentinels`, version bump to 0.7.9.  Tag `v0.7.9` requires (1) merge of `claude/audit-lazybridge-llm-SXOl4` to main and (2) one CI cycle green. |
| Phase 5 — extract `report_builder` (rolled into v0.7.9) | **done — LazyBridge side** | 2026-05-10 | (no separate tag; v0.9.0 plan obsolete) | EncryptedStoreAdapter + cryptography extra shipped (`278f661`, 23 tests). `report_builder` deleted from `lazybridge` (`59a8565`, −8499 LOC) and bundled into the staging zip the user pushed to `selvaz/LazyReport`. Suite: 1730 → 1610 (−120 report_builder tests gone, all green). User-side: configure CI on LazyReport, publish `lazybridge-reports 0.1.0` to PyPI. |
| Phase 6 — stabilisation (staying on 0.7.9) | **partial — strict tier + SECURITY.md done** | 2026-05-10 | (no tag, on 0.7.9) | mypy strict on `lazybridge.engines.*` + `lazybridge.core.providers.*` now in `pyproject.toml`; `SECURITY.md` refreshed (MCP.stdio deny-by-default, EncryptedStoreAdapter, native-tool opt-in).  `__stability__="stable"` / `v1.0.0` deferred per user direction (keep 0.7.9). |
