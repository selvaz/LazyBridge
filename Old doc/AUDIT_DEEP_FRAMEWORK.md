# LazyBridge — Deep Framework Audit (Full Sweep)

- **Base branch:** `claude/framework-deep-audit-FhGY8` @ `aba4cdf`
- **Reviewer branch (this report):** `claude/framework-deep-audit-2N6F6`, layered on top of FhGY8
- **Scope:** every Python module in `lazybridge/` (~26.5 k LOC, 85 files), the test suite, and the package surface — **not** just the diff
- **Audit date:** 2026-05-08
- **Version under review:** `lazybridge 0.7.0` (alpha)

The base branch already carries its own deep audit (`AUDIT.md`) covering 13 follow-up issues plus a replay-mode fix. This report (a) verifies that work, (b) surveys the rest of the framework, and (c) records the false-positive trail so future reviewers don't re-trace it.

## 1. Baseline (run from FhGY8 HEAD directly)

| Check | Result |
| --- | --- |
| `pytest -q` | **1644 passed, 53 skipped** in 18.7 s |
| `ruff check lazybridge/ tests/` | clean |
| `ruff format --check lazybridge/` | 85 files already formatted |
| `python -m lazybridge.skill_docs._build --check` | `No drift.` |

The 4-test gap vs the source-branch's claim of 1648 is optional-dep skips in this environment, not a regression. All gates green.

## 2. Verification of FhGY8's own claims

| Cluster | Claims | Verified | Notes |
| --- | --- | --- | --- |
| `9a34186` original 9 fixes (E9, S1, S5, X1, X2, S6, A5, C6, A2, B1) | 9 | 9 | Each at the cited line; tests in `test_resolution_fixes.py` (407 LOC) all green. |
| `1ab5ca2` 13 follow-up fixes (AUDIT.md §3.1–3.13) | 13 | 13 | Each at the cited line; tests in `test_audit_followup.py` (472 LOC) all green. |
| `aba4cdf` replay-mode graph reconstruction (R1/R2/R3) | 3 | 3 | `Plan.run()` wraps `_run_impl()` in try/finally so AGENT_FINISH fires on **all 8 exit paths** — resume-after-done, no-steps, missing-step, parallel-band error, sequential step error, max-iter, normal completion, unhandled exception. `TOOL_CALL` / `TOOL_RESULT` / `TOOL_ERROR` carry `agent_name` (`_plan.py:1102-1109, 1150-1156, 1160-1166`). `reconstruct_graph` derives nodes from `step` (Plan) / `tool` (LLM), edges from `agent_name`, and dedupes via `(agent, child, "tool")`. Edge / node payload shapes match `GraphSchema.to_dict()` exactly. |

**Verdict:** every fix the branch claims is honestly delivered, exercised by tests, and free of regressions in the suite.

**Caveats** (acknowledged, not bugs):
- The §3.2 parallel-band lost-write window is documented but not crash-tested. Simulating a process crash mid-step is impractical in unit tests; the docstring is the contract.
- 6 of 7 R1/R2/R3 tests are unidirectional; only `test_replay_visualizer_graph_matches_live_graph_for_mockagent_plan` is a true live → SQLite → replay round-trip.

## 3. New findings (not in FhGY8's AUDIT.md)

Found by independently sweeping the entire framework + test suite + package surface. Each was verified directly against source on FhGY8's HEAD.

### P1 — production / security

#### 3.1 MCP transports: concurrent `connect()` race leaks an `AsyncExitStack`

- **Location:** `lazybridge/ext/mcp/transports.py:62-86` (StdioTransport) and `:133-152` (HttpTransport).
- **Mechanism:** `connect()` opens with `if self._session is not None: return` (`:63`, `:134`) but the check is not atomic against the rest of the method. Two coroutines that race past the early-return both create an `AsyncExitStack`, both `enter_async_context(stdio_client(...))`, and the second assignment to `self._stack` / `self._session` clobbers the first. `close()` (`:110-116`) only unwinds the surviving stack — the orphaned subprocess (stdio) or HTTP session (http) leaks until process exit.
- **Risk:** real for any caller fanning out concurrent MCP requests through a shared transport (FastAPI middleware, async batch jobs). Subprocess leak is the largest blast radius.
- **Fix:** guard `connect()` with an `asyncio.Lock` per transport instance; on entry take the lock, re-check `self._session`, then proceed.
- **Confidence:** High.

#### 3.2 Quarto YAML: project-level `title` / `author` rendered unescaped

- **Location:** `lazybridge/external_tools/report_builder/quarto/project.py:99-102`.
- **Mechanism:** `qmd.py:265,267` correctly funnels metadata through `_yaml_escape` (`qmd.py:308`); `project.py` does not:
  ```python
  # qmd.py — escaped
  yaml_lines = ["---", f'title: "{_yaml_escape(report.title)}"']
  if report.metadata.get("author"):
      yaml_lines.append(f'author: "{_yaml_escape(report.metadata["author"])}"')
  # project.py:99-102 — NOT escaped
  if title:  parts.append(f'title: "{title}"')
  if author: parts.append(f'author: "{author}"')
  ```
  A `"`, `\`, or newline in `title` / `author` breaks the YAML structure; with a `\n  pre-render: bash -c '...'` payload, an attacker (or a buggy LLM-supplied metadata field) injects new top-level YAML keys, and Quarto runs the `pre-render` / `post-render` hooks as shell commands.
- **Fix:** route both writes through `_yaml_escape`. Easiest is to lift `_yaml_escape` to a shared helper (e.g., `quarto/_yaml.py`) since two files now need it.
- **Confidence:** High.

#### 3.3 `section_renderer.render_chart_section` reads arbitrary filesystem paths

- **Location:** `lazybridge/external_tools/report_builder/section_renderer.py:46-49`.
- **Mechanism:** `Path(section.path).exists()` then `_chart_to_figure_html(chart_path, …)` reads the bytes via `read_bytes()`. The high-level entry (`tools.py:230-237`) routes `chart_refs` through a closure-bound `_safe_input_path` that pins paths under `_input_root`; but the **typed `sections` payload** (an LLM-supplied tool argument — see `generate_report.sections: list[dict] | None`) is rendered through `sections_to_html` → `render_chart_section` *without* that validation. The LLM can put `/etc/passwd`, `~/.aws/credentials`, etc. in `sections[i].path` and have the bytes base64-encoded into the rendered HTML.
- **Fix:** add `input_root: Path | None = None` to `render_chart_section` and `sections_to_html`; when set, validate `section.path` resolves under it. Have `tools.py` thread `_input_root` into `sections_to_html(sections, input_root=_input_root)`.
- **Confidence:** High (verified end-to-end: `section.path` is LLM-controllable through the `sections` tool argument).

#### 3.4 `tool_schema._annotation_to_schema` silently coerces common stdlib types to `string`

- **Location:** `lazybridge/core/tool_schema.py:271-394`.
- **Mechanism:** the function dispatches on `__origin__` and a small `_PY_TO_JSON` map for primitives, then falls through to `return {"type": "string"}` (`:394`) for "unknown" types. Affected (verified by reading the dispatch chain): `bytes`, `bytearray`, `datetime.datetime`, `datetime.date`, `decimal.Decimal`, `complex`, `pathlib.Path`, `collections.abc.Sequence/Mapping/Set` (when typed against the abstract instead of the concrete generic), `typing.NewType`, `typing.ForwardRef`, generic `Callable`. Any of these on a tool argument means the LLM is told the parameter is a string, the model emits a string, and the dispatcher passes a string to a function expecting (e.g.) a `Decimal` — `TypeError` deep in user code.
- **Fix:** add explicit branches before the fallback:
  - `bytes` / `bytearray` → `{"type":"string","format":"byte"}`
  - `datetime.datetime` → `{"type":"string","format":"date-time"}`
  - `datetime.date` → `{"type":"string","format":"date"}`
  - `decimal.Decimal` → `{"type":"number"}`
  - `pathlib.Path` → `{"type":"string"}` (explicit, not via fallback)
  - `complex` → `{"type":"string"}` with a `description` noting `"a+bj"` form, or raise.
  For genuinely unknown types, raise `ToolSchemaBuildError` instead of silently degrading.
- **Confidence:** High.

#### 3.5 Report templates: `meta.*` and `css` rendered unescaped under `autoescape=False`

- **Location:** `external_tools/report_builder/renderer.py:325` (`Environment(autoescape=False)`) and templates `executive_summary.html.j2:54-58, 65-66`, `data_snapshot.html.j2:84`, `deep_dive.html.j2:78`.
- **Mechanism:** the `autoescape=False` is correct because `body_html` is bleach-pre-sanitised; `title` is escaped via `|e`; the recent fix already added `|e` to TOC `id` / `text`. Still unescaped: `meta.generated_at`, `meta.theme`, `meta.template`, `meta.charts_embedded`, and `{{ css }}`. `tools.py:245-249` builds `meta` from the tool-call arguments `theme` and `template` — both LLM-controllable — without an allowlist check before they reach the template. A caller (or LLM) supplying `theme="\" /><script>alert(1)</script>"` lands a stored XSS in the HTML; the same template surface is reused by `weasyprint.py:60-80` so the injected DOM is also embedded in the PDF (script doesn't fire there but the malicious HTML still ships).
- **Fix:** add `|e` on `meta.theme`, `meta.template`, `meta.generated_at` (and `meta.charts_embedded` for consistency, even though it's an int) in every template that consumes them. Treat `{{ css }}` as static — document at the renderer that `css` must come from `load_theme_css()` only and never from user input.
- **Confidence:** Med-High (plumbing verified; the `theme` argument is genuinely LLM-controllable).

### P2 — correctness / robustness / hygiene

#### 3.6 Strict-mode JSON schema may still fail OpenAI's stricter rules

- **Location:** `lazybridge/core/tool_schema.py:1051-1056`.
- **Mechanism:** the recent fix correctly emits `required: []` alongside `additionalProperties: false` in strict mode with no required args. But OpenAI's strict tool-call validator additionally requires *every* property to appear in `required` (it does not honour the JSON Schema "default" convention in strict). For a function with optional args + `strict=True`, the current shape — properties present but absent from `required` — is rejected by OpenAI strict; Gemini accepts it.
- **Fix:** add a docstring note describing the dialect divergence (changing the shape would break Gemini-targeted tests). Longer-term, split into `strict_dialect="openai" | "gemini"` if both must be supported precisely.
- **Confidence:** Med (depends on which provider's strict semantics the framework intends to target).

#### 3.7 `viz/server`: stdlib `http.server` POST is unbounded

- **Location:** `lazybridge/ext/viz/server.py:232-233` (`do_POST`).
- **Mechanism:** the handler reads `int(self.headers.get("Content-Length") or 0)` bytes from `rfile`. A misbehaving client can send `Content-Length: 1000000000` and pin the handler thread on a multi-GB allocation. Default bind is `127.0.0.1` and the path is token-gated, so the practical attack surface is "any local process on the same machine" — but the `host` constructor argument is configurable, so a user that exposes the visualiser internally (LAN dashboard, etc.) loses the bind-only mitigation.
- **Fix:** define `MAX_PAYLOAD = 1_000_000` and `_deny(413, "payload too large")` on overflow before allocation.
- **Confidence:** High for the bug; Low–Med for the exploitation risk in the default config.

#### 3.8 OTel exporter `_set_attr` doesn't bound attribute size

- **Location:** `lazybridge/ext/otel/exporter.py:311-314`.
- **Mechanism:** large prompts/results are stringified and passed straight to `span.set_attribute`. The OTel SDK has its own limits but exceeding them is silent truncation in most builds, and the wire payload can balloon and stall span flush in WAN-attached collectors. `_truncate()` exists elsewhere in the file but isn't applied here.
- **Fix:** route every string-valued attribute through `_truncate()`; expose a configurable `max_attr_chars` knob.
- **Confidence:** Med.

#### 3.9 `gateway.py`: `urllib.request.urlopen` follows cross-host redirects

- **Location:** `lazybridge/ext/gateway.py:154-156` (`_request`).
- **Mechanism:** `urlopen` auto-follows redirects via the default `HTTPRedirectHandler`. A configured tool URL that the operator believes lands on `https://api.example.com/v1/tool` could 302 to `http://internal-only/admin`. For a generic external-tool gateway pointed at user/operator-supplied endpoints, this is an SSRF surface — and the request still carries the gateway's `Authorization: Bearer ...` header to whatever host the redirect chain ends at.
- **Fix:** install a `HTTPRedirectHandler` subclass that rejects scheme changes (HTTPS→HTTP) and host changes; allow same-host path redirects only. Optionally provide an explicit allowlist arg.
- **Confidence:** Med.

#### 3.10 `Predicates._safe_get` masks attribute typos with `None`

- **Location:** `lazybridge/predicates.py:240-255`.
- **Mechanism:** `when.field("itmes").not_empty()` (typo for `items`) silently treats the missing attribute as "empty", and the predicate fires when the user almost certainly didn't intend it to. The docstring frames this as "fail clean" but in practice it masks routing bugs that look like data shape changes.
- **Fix:** add `strict=True` opt-in that raises `AttributeError` when the field is missing; keep the lax default for backwards compat. Document the trade-off in the predicate guide.
- **Confidence:** Low (UX, not data-loss).

#### 3.11 `Memory._plan_compression` documents an invariant the lock doesn't enforce

- **Location:** `lazybridge/memory.py:206-236`.
- **Mechanism:** the docstring says "Caller MUST hold `self._lock`" but the function does not assert that. The actual primitive that protects the snapshot-then-compress window is the `_compressing` flag, not the lock. A future contributor following the docstring could move the snapshot work outside the lock and break the invariant unnoticed.
- **Fix:** rewrite the docstring to name `_compressing` as the guard, or add an `assert self._lock._is_owned()` (RLock-only) at function entry.
- **Confidence:** Med (no current bug; an attractive nuisance).

#### 3.12 `EventLog.record` close-during-emit comment not mirrored on `record_many`

- **Location:** `lazybridge/session.py:340-357` vs `:367-376`.
- **Mechanism:** `record()` checks `self._closed` outside any lock and the comment at `:335-339` accepts the race ("bounded to a single INSERT + COMMIT"). `record_many()` does the same `_closed` check unguarded but lacks the explanatory comment. Two paths that look inconsistent erode confidence; a reader will assume one is wrong.
- **Fix:** copy the explanatory comment to `record_many` (or factor the check into a helper).
- **Confidence:** High (cosmetic).

### P3 — test-suite & package-surface hygiene

#### 3.13 Plan compiler / serialisation have no direct unit tests

- **Location:** `lazybridge/engines/plan/_compiler.py`, `lazybridge/engines/plan/_serialisation.py`. Verified: `grep -rln 'from lazybridge.engines.plan._compiler\|engines\.plan\._compiler\|from lazybridge.engines.plan._serialisation' tests/` returns zero results.
- **Mechanism:** validation (step-name uniqueness, route-target existence, Literal type checking) and serialisation (`to_dict` / `from_dict` round-trips, sentinel handling) are exercised only as side-effects of integration tests.
- **Fix:** add focused unit files (e.g., `tests/unit/test_plan_compiler.py`, `test_plan_serialisation.py`) covering invalid step references, route-target typos, round-trip fidelity for nested Pydantic models, and version-skew on `from_dict`.
- **Confidence:** High.

#### 3.14 Version is duplicated in two places

- **Location:** `pyproject.toml:7` (`version = "0.7.0"`) and `lazybridge/__init__.py:84` (`__version__ = "0.7.0"`). Verified.
- **Mechanism:** future bumps that touch only one site cause `lazybridge.__version__` to disagree with `importlib.metadata.version("lazybridge")`. Standard fix is to single-source.
- **Fix:** read from `importlib.metadata.version("lazybridge")` in `__init__.py`, **or** put the constant in `lazybridge/_version.py` and import it into both `pyproject.toml` (via build-tool dynamic version) and the package.
- **Confidence:** High.

#### 3.15 `tool_choice='parallel'` deprecation has no sunset version

- **Location:** `lazybridge/engines/llm.py:214-221`.
- **Mechanism:** the `DeprecationWarning` rewrites `'parallel'` to `'auto'` but the warning text doesn't say "will be removed in lazybridge 1.0". Users grep for sunset dates to schedule migration; without one, the warning is permanently ignorable.
- **Fix:** append "will be removed in lazybridge 1.0" (or whatever version line) to the warning message.
- **Confidence:** Med.

## 4. Findings investigated and dismissed (false-positive log)

These came up during exploration and were rejected after reading the source. Recorded so a future reviewer doesn't re-trace.

- **`Store.write` calls `_conn()` twice and could land on different connections.** Dismissed: `_conn()` (`store.py:48-66`) caches the connection on `self._local` (a `threading.local`); within a single thread the same object returns on every call. Cross-thread the connections are deliberately distinct (each thread gets its own SQLite connection with `check_same_thread=False`).
- **`Store.compare_and_swap` deadlocks because the re-raise happens inside `with self._lock:`.** Dismissed: `self._lock` is an `RLock` (`store.py:36`); the `with` block releases on exception unwind. Same-thread re-entry is also legal.
- **Plan `TOOL_CALL` is missing `agent_name`.** Dismissed: `_plan.py:1102-1109` adds `agent_name` to the payload when the wrapping agent passes it; `:1150-1156` and `:1160-1166` mirror it for `TOOL_RESULT` and `TOOL_ERROR`.
- **`Memory.add` token-scaling factor `* 1.3` applies even when explicit tokens are passed.** Dismissed: the scaling at `memory.py:137-145` is gated on `tokens == 0` (auto-estimation path); explicit `tokens=` bypasses it.
- **`reconstruct_graph` silently drops orphan tool-call edges.** Defensive behaviour — Plan and LLMEngine both always set `agent_name`, so the silent drop never fires in practice. Worth a debug-log if the field is ever missing, but not a bug.
- **`OTelExporter._on_agent_end` orphan-cleanup snapshot.** Verified: keys including `"agent"` are snapshotted under a single lock acquisition before the close pass. No race.
- **Plan `_routing` step-name lookup could trip on dict ordering.** Dismissed: insertion-order iteration is guaranteed in Python 3.7+, which is the project's floor.
- **Pydantic `Field(default_factory=...)` not invoked for missing args.** Dismissed: Pydantic v2 invokes default factories during `model_validate` when the field is absent in the input; the call path here (`tool_schema.py:462-464, 508`) builds the model and reads attributes after validation — factories fire correctly.
- **`Plan.run()` mints a fresh `run_id` on resume.** Documented contract: each invocation is its own run; cross-resume correlation goes through `plan_state` and the checkpoint, not `run_id`.
- **HumanEngine CSRF on the localhost form.** Server is bound to `127.0.0.1` (`hil/human.py:315`), the response port is short-lived, and the same-machine threat model already grants the attacker higher-privilege paths. Flag if the binding ever becomes user-configurable.
- **`asyncio.sleep(...)` patterns in tests look like sleep-and-pray timeouts.** Inspected the call sites — those sleeps live inside mock engines / mock tools that simulate hangs and are intended to be cancelled by the production timeout being tested. Not test-quality bugs.

## 5. Recommended remediation order

1. **3.1 (MCP `connect()` race)** — short, mechanical fix; prevents subprocess / HTTP-connection leak under any concurrent caller.
2. **3.2 (Quarto YAML escape)** — one-line application of an existing helper; closes a YAML→shell injection surface that becomes real once metadata flows from LLMs.
3. **3.3 (`section_renderer` arbitrary read)** — defense-in-depth on a public renderer; the safe path should not depend on which entry the caller chose.
4. **3.4 (`tool_schema` string fallback)** — small dispatch additions for stdlib types; prevents silent type degradation in tool-call arguments.
5. **3.5 (template `meta.*` escaping)** — one-line `|e` on three template tokens; closes the only XSS surface left in the report HTML.
6. The remainder (3.6–3.15) are P2 / P3 hygiene; group them in a single hardening PR.

## 6. Summary

The base branch (FhGY8) has done careful, honest work: **22 distinct fix-claims across three commits, 100 % verified delivered**, with focused regression tests for each. Baseline is fully green (1644 / 53, lint / format / docs all clean).

This audit adds **15 findings** the FhGY8 audit didn't reach, because that audit scoped tightly to the diff:

- **5 P1 issues** in production paths — MCP connect race, two report-builder injection / arbitrary-read surfaces, the schema-fallback degradation, and the template-XSS surface.
- **7 P2 issues** spanning correctness, robustness, and consistency — strict-mode dialect, viz/server payload cap, OTel attribute size, gateway redirects, predicate UX, Memory docstring drift, and EventLog comment parity.
- **3 P3 issues** at the test / package surface — missing direct tests for Plan compiler/serialisation, version duplication, and a deprecation sunset.

None of these are show-stoppers, none of them regressed in FhGY8, and none invalidate FhGY8's audit. They are pre-existing issues in scope-adjacent areas the FhGY8 audit didn't intend to cover.
