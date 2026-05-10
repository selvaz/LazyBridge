# LazyBridge — Implementation tracking (0.7.0 → 1.0)

> **Status legend.** `[ ]` open · `[~]` in progress · `[x]` done · `[-]` skipped (with reason).
>
> **Update protocol.** Each PR ticks at least one checkbox. PR description references the section heading. When a phase's acceptance gate passes, tag the release.
>
> **Strategic plan**: see `/root/.claude/plans/puoi-verificare-anche-questo-partitioned-marshmallow.md` (planning artifact).

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
| B1 | `core/providers/deepseek.py:268-298` — copy `params["messages"]` before mutating | [-] | — | **Skipped: false positive.** `params["messages"]` is freshly built by `_messages_to_openai` per call. `messages[0] = {**messages[0], ...}` assigns a fresh dict to the freshly-built list slot. `request.messages` is never touched. |
| B2 | `core/providers/anthropic.py:163-176` — extend `_compute_cost(cached_input_tokens=0)`; 25% rate | [ ] | | |
| B3 | `core/providers/openai.py:818-825, 884` — preserve `usage` across null-check (streamed Responses) | [ ] | | |
| B4 | `core/providers/anthropic.py:262-280` — warn on URL-source AudioContent | [ ] | | |
| B5 | `core/providers/google.py:649-661, 749-757` — re-use raw Part; never reconstruct `thought_signature` | [ ] | | |
| B6 | `engines/plan/_compiler.py:148-165` — validate existence first; typo-aware error | [ ] | | |
| B7 | `engines/plan/_serialisation.py:95-119` — raise `ValueError` on unknown sentinel kind | [x] | claude | direct commit; test: `tests/unit/test_serialisation_unknown_sentinel.py` + updated `test_provider_static_paths.py::test_plan_serialization_unknown_sentinel_kind_raises` |
| B8 | `ext/otel/exporter.py:146-175` — log SDK exceptions at WARNING (don't swallow) | [x] | claude | direct commit; test: `tests/unit/test_otel_exception_logging.py` |
| B9 | `ext/planners/blackboard.py:67-93` — reset closure state per `run()` | [x] | claude | direct commit; test: `tests/unit/test_blackboard_state_reset.py` |
| B10 | `engines/plan/_types.py:145`, `_plan.py:849` — forbid auto-name when step referenced | [ ] | | |
| B11 | `engines/plan/_plan.py:858-869` — reorder: `store.write(step.writes)` before checkpoint commit | [ ] | | |
| B12 | `lazybridge/skill_docs/_build.py` — recover from `Old doc/`; add `--check`; wire into CI | [ ] | | |

### Regression tests (new in `tests/unit/`)

| File | Locks | Status |
|---|---|---|
| `test_llm_engine_tool_choice_any_resets.py` | tool_choice="any" infinite loop fix | [ ] |
| `test_store_inmemory_cas_isolation.py` | Store CAS deep-copy fix (in-memory + SQLite, T12) | [ ] |
| `test_stream_idle_timeout_default.py` | stream_idle_timeout=90.0 default + warn-once-on-None | [ ] |
| `test_mcp_stdio_security_warning.py` | MCP.stdio() warning when allow/deny missing | [ ] |
| `test_provider_google.py` | Google provider FakeTransport smoke + finish_reason | [ ] |
| `test_provider_deepseek.py` | DeepSeek FakeTransport smoke | [ ] |
| `test_sentinels_matrix.py` | Every sentinel × forward-ref / missing / nested-plan | [ ] |
| `test_doc_examples_runtime.py` | Existing doc examples actually run with MockAgent | [ ] |

### CI hygiene

- [ ] Add `permissions: contents: read` to `.github/workflows/test.yml` (top-level)
- [ ] `docs.yml` asserts `site/llms.txt` and `site/llms-full.txt` are >1KB after `mkdocs build --strict`
- [ ] Remove false CHANGELOG line about `skill_docs._build --check` (or fulfill via B12)

### Acceptance gate (Phase 1 → Phase 2)

- [ ] B1–B12 fixed; each has a regression test that passes
- [ ] `pytest tests/unit/` green on Python 3.11 / 3.12 / 3.13
- [ ] `mkdocs build --strict` green; `llms.txt` non-empty
- [ ] `python -m lazybridge.skill_docs._build --check` exits 0

---

## Phase 2 — Deletions (~1 week)

Goal: shrink the public surface by ~20 concepts. Zero new public concept added.

### Block A — Delete 5 factory aliases

In `lazybridge/agent.py`:
- [ ] Delete `Agent.from_chain` (line 1008)
- [ ] Delete `Agent.from_engine` (line 876)
- [ ] Delete `Agent.from_model` (line 864)
- [ ] Delete `Agent.from_plan` (line 967)
- [ ] Delete `Agent.from_parallel` (line 1022)

Sed-replace in repo (43 call sites across `tests/`, `examples/`, `docs/`, `SKILL.md`):
- [ ] `Agent.from_model("X", **k)` → `Agent("X", **k)`
- [ ] `Agent.from_engine(e, **k)` → `Agent(engine=e, **k)`
- [ ] `Agent.from_plan(*s, **k)` → `Agent(engine=Plan(*s, **plan_kw), **agent_kw)`
- [ ] `Agent.from_chain(a, b, **k)` → `Agent.chain(a, b, **k)`
- [ ] `Agent.from_parallel(*a, **k)` → `Agent.parallel(*a, **k)`

### Block B — Delete 4 public config objects (kills `_UNSET`)

In `lazybridge/__init__.py` and `lazybridge/core/types.py`:
- [ ] Delete `AgentRuntimeConfig`
- [ ] Delete `CacheConfig`
- [ ] Delete `ObservabilityConfig`
- [ ] Delete `ResilienceConfig`
- [ ] Remove these names from `__all__`

In `lazybridge/agent.py`:
- [ ] Delete `_UNSET` sentinel
- [ ] Delete `_resolve_runtime_kwargs` (~50 lines)
- [ ] Each `Agent.__init__` kwarg uses a real default; no precedence game

Update tests/examples (~30 sites):
- [ ] Replace `Agent(resilience=ResilienceConfig(timeout=30, ...), engine=...)` with `Agent(timeout=30, ..., engine=...)`
- [ ] Same pattern for `cache=CacheConfig(...)`, `observability=...`
- [ ] Document the dict-spread fleet pattern in SKILL.md (`Agent(**PROD_DEFAULTS, engine=...)`)

### Block C — Delete silent fallbacks (uniform raise)

| Path | Becomes | Status |
|---|---|---|
| `engines/plan/_plan.py:1034-1059` `from_step("typo")` | `PlanRuntimeError` | [ ] |
| `engines/llm.py:401-430` unknown provider | `ValueError` | [ ] |
| `engines/llm.py:328` set `_PROVIDER_DEFAULT = None` | (effectively raise) | [ ] |
| `ext/mcp/server.py:184-207` non-`object` `inputSchema` | `ValueError` | [ ] |
| `core/providers/deepseek.py:317-324` `tools+structured_output` | `UnsupportedFeatureError` | [ ] |
| `core/providers/openai.py:912+` Pydantic + `IMAGE_GENERATION` | `UnsupportedFeatureError` pre-flight | [ ] |
| `core/providers/google.py:492-498` grounding+structured (extend to dict schemas) | `ValueError` | [ ] |
| `envelope.py:75-82` unknown payload | `TypeError` | [ ] |
| `memory.py:124-126` `summarizer_timeout < 5.0` | warn at construction | [ ] |
| `core/providers/base.py:356-371` `_resolve_model` empty | `ValueError` with concrete message | [ ] |

### Block D — Delete `mode="auto"`; align tool defaults

In `lazybridge/tools.py`:
- [ ] Remove `auto` from the `mode` Literal in `Tool.__init__`
- [ ] Remove `auto` from the `tool()` factory's `mode` parameter
- [ ] Default both `Tool(...)` and `tool(...)` to `mode="signature"`
- [ ] Delete `_resolve_auto_tool` (silent graceful-degrade ladder)

### Block E — `Tool` → `_Tool` (private)

In `lazybridge/tools.py`:
- [ ] Rename class `Tool` → `_Tool`
- [ ] Remove `Tool` from `__all__` (keep `_Tool` accessible via `from lazybridge.tools import _Tool`)

In `lazybridge/__init__.py`:
- [ ] Drop `Tool` from top-level re-exports

Sed-replace in repo (30+ direct `Tool(...)` constructions):
- [ ] `Tool(fn, name=..., ...)` → `tool(fn, name=..., ...)` across `tests/`, `examples/`, `docs/`

### Block F — `_ParallelAgent` → `ParallelAgent` returning `Envelope`

In `lazybridge/agent.py`:
- [ ] Rename `_ParallelAgent` → `ParallelAgent`
- [ ] Add `ParallelAgent` to `__all__` and `lazybridge/__init__.py`
- [ ] `__call__` returns one `Envelope` whose `.payload` is `list[Envelope]` and `.text()` joins branch text
- [ ] Update tests/examples (~22 sites): `for r in result:` → `for r in result.payload:`

### Block G — Delete dead weight

- [ ] Delete `Old doc/` directory (after Phase 1's B12 has recovered `skill_docs/_build.py`)
- [ ] Delete `pythonpath = ["lazybridge"]` from `pyproject.toml:91`
- [ ] Rename `wrap_tool` (public) → `_wrap_tool` (private) in `tools.py`; update internal callers
- [ ] Remove `tool_choice="parallel"` from `engines/llm.py:215` (was deprecated; just delete)

### Acceptance gate (Phase 2 → Phase 3)

- [ ] All Block A/B/C/D/E/F/G items completed
- [ ] `pytest tests/unit/` still green
- [ ] `python -c "import lazybridge; print(len(lazybridge.__all__))"` shows fewer symbols than start of Phase 1
- [ ] No `_UNSET`, no `mode="auto"`, no `from_chain/from_engine/from_model/from_plan/from_parallel` in source

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
- [ ] Validate every sentinel at compile time: `from_step`, `from_parallel`, `from_parallel_all`, `from_agent`, `from_memory`
- [ ] `from_step("sibling")` inside a parallel band → `PlanCompileError` (closes T5)
- [ ] `Step.task=` callable signature checked at compile time (closes T13)
- [ ] `Agent(model=..., engine=non-LLM)` → `ValueError` at `Agent.__init__` (closes T6)
- [ ] `Agent(name=...)` required when engine is non-LLM (closes T7)
- [ ] `from_agent`/`from_memory` validation on existence not truthiness (closes I6)
- [ ] Forward-ref check duplicated across task/context paths → one helper (closes I7)
- [ ] All error messages adopt the four-part format

### Block I — Provider consistency

Capability classvars in each `lazybridge/core/providers/*.py`:
- [ ] `AnthropicProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [ ] `OpenAIProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [ ] `GoogleProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [ ] `DeepSeekProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [ ] `LiteLLMProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [ ] `LMStudioProvider.capabilities: ClassVar[frozenset[NativeTool]]`
- [ ] `BaseProvider`-level: `supports_streaming`, `supports_structured_output`, `supports_thinking: ClassVar[bool]`

Cost/streaming parity:
- [ ] Every `_compute_cost` accepts `cached_input_tokens=0` (extends B2)
- [ ] Every streaming path preserves `usage` (extends B3)
- [ ] Anthropic adaptive-thinking warning corrected for Opus 4.7 (closes I5)
- [ ] `WEB_SEARCH ≡ GOOGLE_SEARCH` for Google: warn if both passed (closes T9)

Pinning:
- [ ] Tighten provider SDK pins to last-tested-minor; explanatory comments
- [ ] Add CI canary that floats to next minor (informational, non-blocking)

Docs generation:
- [ ] mkdocs hook (`docs/_hooks/build_provider_table.py`) generates README support table from classvars

### Block J — Observability

In `lazybridge/ext/otel/exporter.py`:
- [ ] Add `gen_ai.agent.nesting_level` attribute on nested-agent spans (closes I10)

In `lazybridge/session.py`:
- [ ] `Session.emit` exporter exception path: warn-once-per-(exporter, exception class)
- [ ] Counter for dropped events
- [ ] **No** circuit breaker (YAGNI)

### Acceptance gate (Phase 3 → Phase 4)

- [ ] All Block H/I/J items completed
- [ ] `pytest tests/unit/` green
- [ ] Standard error format applied in all PlanCompile/Runtime/UnsupportedFeature errors
- [ ] Provider matrix table in README is generated, not manual

---

## Phase 4 — Docs + examples + CI + skill_docs → tag v0.8.0 (~0.5 week)

### Docs

- [ ] SKILL.md: add canonical Plan block; drop deleted-sugar rows; default-model fallback advice
- [ ] SKILL.md: anti-pattern list updated (no more `from_*` factories, no config objects)
- [ ] `docs/reference/engines.md`: add `thinking=` row to LLMEngine table
- [ ] `docs/guides/mid/parallel.md`: scripted vs LLM-driven disambig
- [ ] `docs/guides/mid/mcp.md`: examples show `allow=` filtering
- [ ] `docs/guides/full/step.md`: `Step.after_branches` section
- [ ] `docs/decisions/composition.md`: alias migration note
- [ ] `docs/reference/providers.md`: Google `finish_reason` normalization
- [ ] `Agent` class docstring reconciled with `__init__`
- [ ] `CHANGELOG.md`: 0.8.0 entry summarising the simplification (deletions, what raises now)

### Examples

- [ ] New `examples/verify_judge_loop.py` (closes I30)
- [ ] New `examples/guardrails_demo.py` (closes I31)
- [ ] Env-var preflight in `examples/daily_news_report.py` (closes I32)

### CI

- [ ] New `.github/workflows/integration.yml`: matrix-driven `live` + `heavy_render` jobs (manual + nightly)
- [ ] `pyproject.toml`: lift core coverage gate to 85%
- [ ] `pyproject.toml`: external_tools omit list updated; their gate at 60%
- [ ] mypy `strict = true` for `agent.py`, `tools.py`, `envelope.py`, `sentinels.py`, `predicates.py` (Phase-1 strict surface)

### Tag v0.8.0 acceptance gate

- [ ] All Phase 1–4 items completed
- [ ] `python -m lazybridge.skill_docs._build --check` exits 0
- [ ] `integration.yml` green for at least one nightly cycle
- [ ] mypy strict on Phase-1 strict-surface modules — clean
- [ ] SKILL.md regenerated; canonical surface only
- [ ] Coverage: core ≥85%, total ≥75%
- [ ] Public symbol count drop measurable: `python -c "import lazybridge; print(len(lazybridge.__all__))"` lower than 0.7.0
- [ ] Tag `v0.8.0`; release workflow publishes via OIDC

---

## Phase 5 — Extract `report_builder` → tag v0.9.0 (~1 week)

### Extraction

- [ ] Create new repo `selvaz/lazybridge-reports`
- [ ] Move `lazybridge/external_tools/report_builder/**` → `lazybridge_reports/**`
- [ ] Move `[report]`, `[report-charts]`, `[report-citations]`, `[report-fallback]`, `[pdf]` extras out of `lazybridge`
- [ ] **No shim.** Delete the old import path from `lazybridge` entirely
- [ ] Update report examples (`daily_news_report.py`, `parallel_report_pipeline.py`) to new import path
- [ ] Set up independent CI for `lazybridge-reports` (test + release workflows)
- [ ] Publish `lazybridge-reports` 0.1.0 to PyPI (parallel with v0.9.0)

### Encryption adapter

- [ ] New `lazybridge/store/encryption.py`: `EncryptedStoreAdapter(store, key=...)` using `cryptography.fernet`
- [ ] Optional extra: `pip install lazybridge[encryption]`
- [ ] Smoke test: round-trip a payload through the adapter
- [ ] SECURITY.md updated with PII threat model

### Strictness lift

- [ ] mypy strict phase 2: `engines/`, `core/providers/`
- [ ] Provider SDK objects get explicit type stubs for the surface we touch
- [ ] Core coverage 90%

### Tag v0.9.0 acceptance gate

- [ ] `pip install lazybridge` (clean venv) — no report deps
- [ ] `pip install lazybridge-reports` — both report examples run
- [ ] `EncryptedStoreAdapter` round-trip smoke green
- [ ] mypy strict on engines + providers — clean
- [ ] Tag `v0.9.0`

---

## Phase 6 — Stabilisation → tag v1.0.0 (~1 week)

- [ ] mypy strict full repo
- [ ] External security review pass (or internal threat-model review against SECURITY.md)
- [ ] `__stability__ = "stable"` in `lazybridge/__init__.py`
- [ ] README pin guidance updated: `lazybridge>=1,<2`
- [ ] Tag `v1.0.0`

---

## Cross-cutting principles (apply throughout)

- [ ] CI lint scans CHANGELOG `(bug fix)` markers and asserts a matching test exists
- [ ] Every error message follows the four-part format
- [ ] Every guide page has a runnable example tested with `MockAgent`

---

## Implementation status

| Phase | Status | Started | Tagged | Notes |
|---|---|---|---|---|
| Phase 1 — bugs + tests + CI hygiene | **in progress** | 2026-05-10 | | B1 cleared (false positive); B7, B8, B9 fixed + locked. 1689/1689 unit tests green. Remaining: B2, B3, B4, B5, B6, B10, B11, B12 + 8 regression tests + CI hygiene. |
| Phase 2 — deletions A–G | not started | | | |
| Phase 3 — validation parity + provider consistency + observability | not started | | | |
| Phase 4 — docs + examples + CI + skill_docs (v0.8.0) | not started | | | |
| Phase 5 — extract `report_builder` (v0.9.0) | not started | | | |
| Phase 6 — stabilisation (v1.0.0) | not started | | | |
