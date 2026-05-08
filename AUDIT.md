# LazyBridge — Deep Framework Audit

- **Source branch:** `claude/audit-competent-chebyshev-QauGF` @ `7d5cb08`
- **Branch under review (this report):** `claude/framework-deep-audit-FhGY8`
- **Version:** `lazybridge 0.7.0` (alpha)
- **Audit date:** 2026-05-08

## 1. Baseline health

| Check | Result |
| --- | --- |
| `pytest` (default selectors) | **1626 passed, 50 skipped** in 18.6 s |
| `ruff check lazybridge/` + `tests/` | clean |
| `ruff format --check lazybridge/` | **2 files drift** — `lazybridge/agent.py`, `lazybridge/tools.py` |
| `python -m lazybridge.skill_docs._build --check` | `No drift.` |

The framework imports cleanly, every test in the default selection passes, and every lint rule passes. The only mechanical baseline failure is `ruff format` on two files (mostly long-string wrap differences); not bugs, but the project's CI presumably runs `--check` and would fail.

## 2. Verification of source-branch claims

`9a34186 fix(audit): deliver all promised capabilities — 9 framework fixes` lists nine fixes plus a lint pass. Each was verified against the diff and the new tests in `tests/unit/test_resolution_fixes.py` (407 LOC, all passing). Summary:

| ID | Claim | Implementation | Test coverage | Verdict |
| --- | --- | --- | --- | --- |
| **E9** | Plan checkpoint-before-store-write ordering, parallel + sequential | `_plan.py:642-657` (parallel), `_plan.py:703-714` (sequential): `_save_checkpoint` then `effective_store.write` | `test_plan_sequential_checkpoint_before_store_write`, `test_plan_parallel_checkpoint_before_store_write` | **Delivered** (caveat below) |
| **S1** | `Memory.add` auto-estimates tokens from word count | `memory.py:137-143` — when `tokens=0`, set `tokens = len((user + " " + assistant).split())` | 3 tests covering trigger, explicit-tokens override, empty-content no-op | Delivered |
| **S5** | `OTelExporter` defaults to `BatchSpanProcessor` | `exporter.py:91, 98` — `batch: bool = True` selects `BatchSpanProcessor` | `test_otel_exporter_default_uses_batch_span_processor`, `…_simple_span_processor` | Delivered |
| **X2** | `OTelExporter` no longer calls `trace.set_tracer_provider()` | Confirmed absent in `exporter.py:86-108`; only `provider.get_tracer(...)` is held | `test_otel_exporter_does_not_set_global_provider` | Delivered |
| **X1** | `OTelExporter.flush()` exists | `exporter.py:342-348` — wraps `provider.force_flush(...)` | covered by smoke / S5 tests | Delivered |
| **S6** | `Session.emit` warns on **every** exporter failure | `session.py:735-741` — `try/except` per-exporter, `warnings.warn` each time, no `_warn_once` flag | `test_session_emit_warns_on_every_exporter_failure` | Delivered |
| **A5** | `Agent.__init__` detects `fallback=` cycles | `agent.py:358-367` — id-set walk over the chain, `ValueError` on revisit | `test_fallback_cycle_detected_when_chain_already_loops`, `…linear_chain_is_fine`, `…none_is_fine` | Delivered |
| **C6** | `_annotation_to_schema` handles `TypedDict` and `NamedTuple` | `tool_schema.py:367-386` — both branches present; expands to `{"type":"object", "properties": …, "required": …}` | 4 tests (raw schema + via `tool()` factory) | Delivered |
| **A2** | `Agent.stream()` writes to store on full-stream completion | `agent.py:689-715` — `_completed` flag in the `finally`, only writes when stream actually exhausted | `test_stream_writes_to_store_on_completion`, `…not_…on_early_break` | Delivered |
| **B1** | `06_reference.md` flows through `_write()` so `--check` catches drift | `_build.py:558-567` — three reference paths now go through `_write(... , changed)` | `_build --check` returns "No drift." | Delivered |

All nine claims are honestly delivered. Caveats and follow-ups follow in §3.

## 3. New findings (verified)

### P0 — production-impacting

#### 3.1 Provider registration is non-thread-safe

- **Location:** `lazybridge/engines/llm.py:351-377`
- **Mechanism:** `register_provider_alias` and `register_provider_rule` both perform a *read-then-write* on a class attribute:
  ```python
  cls._PROVIDER_ALIASES = {**cls._PROVIDER_ALIASES, alias.lower(): provider}
  cls._PROVIDER_RULES = [(kind, pattern.lower(), provider), *cls._PROVIDER_RULES]
  ```
  Two threads calling these concurrently can each unpack the *current* dict / list, each rebuild a new value, and the second `cls.X = ...` clobbers the first — silently dropping a registration.
- **Risk:** Most users register providers once at import, so this rarely fires in practice. But (a) the framework targets multi-threaded servers (FastAPI, Flask) and (b) the module-level guidance is "extend at runtime", inviting the bad pattern.
- **Fix:** Guard both methods with a module-level `threading.Lock`.

### P1 — correctness / robustness

#### 3.2 Plan parallel-band: data is durably *lost* (not duplicated) on a between-checkpoint-and-store crash

- **Location:** `lazybridge/engines/plan/_plan.py:642-657` (parallel) and `:703-714` (sequential)
- **Mechanism:** The audit fix (E9) chose the *checkpoint-first, store-write-second* ordering, intentionally. That eliminates double-writes on resume — but the inverse failure mode is now silent **lost writes**: if the process crashes between the CAS-checkpoint and `effective_store.write(...)`, the resumed run advances past the band (because the checkpoint already says "done") and never retries the writes. The values still live in the in-memory `kv` snapshot inside the checkpoint, so the Plan itself reads them correctly — but anyone reading `effective_store` directly (sidecar consumers, tests, debugging) sees stale or absent data for the affected keys.
- **Risk:** Users who treat the Store as a side-effect surface (e.g. "step writes go into a shared SQLite that another service reads") will see silent corruption on the rare crash window. Not a regression — same trade-off as before, just now codified — but the docstring at `_plan.py:619-623` only mentions the no-double-write side and frames the new ordering as universally "safe".
- **Fix:** Either (a) document explicitly that durable Store-write is best-effort under crash, with `kv` in the checkpoint as the authoritative replay copy, or (b) add a post-resume reconciliation pass that re-writes any `kv` keys whose `step.writes` value is missing from `effective_store`.

#### 3.3 Stale comment contradicts the B1 fix

- **Location:** `lazybridge/skill_docs/_build.py:533`
  ```python
  # 06_reference.md is written unconditionally below (excluded from drift check)
  ```
- **Mechanism:** The fix routes `06_reference.md` *into* `_write()` (and therefore into `changed`, and therefore into `--check`). The comment says the opposite. Will mislead the next reader who tries to debug a CI drift failure.
- **Fix:** Delete or rewrite the comment to "06_reference.md is written below; like every other file it's drift-checked via `_write()`."

#### 3.4 `MCP` transports gate state with `assert`

- **Location:** `lazybridge/ext/mcp/transports.py:84, 88, 102, 147, 151, 163`
- **Mechanism:** Six sites of the form `assert self._session is not None, "list_tools called before connect"` gate `list_tools` / `call_tool` against an unentered async-context-manager. Under `python -O` / `PYTHONOPTIMIZE=1` (common in slim production images) these are stripped and the next line dereferences `None`, surfacing as an unhelpful `AttributeError: 'NoneType' has no attribute 'list_tools'` instead of the explicit message.
- **Fix:** Replace with `if self._session is None: raise RuntimeError("…")` on all six sites. Asserts are appropriate for `mypy`-narrowing only when followed by code that can never execute on the False branch.

#### 3.5 `OTelExporter._on_agent_end` orphan-cleanup walks stale dict snapshot

- **Location:** `lazybridge/ext/otel/exporter.py:222-236`
- **Mechanism:** The handler reads `self._spans.get(run_id, {}).get("agent")` *outside* the lock (line 222-223), then re-takes the lock to enumerate orphan keys (`:233-234`), then closes them outside the lock (`:235-236`). If a concurrent thread emits a `model_response`/`tool_result` for the same `run_id` between the two lock acquisitions, the just-finished span gets re-added to the dict and is then double-closed by `_end_span`. `_end_span` is itself lock-guarded and tolerates already-popped entries, so this is contained, but the second `.end()` triggers a UserWarning deeper in OTel.
- **Fix:** Take a single snapshot of the run_id's keys under the lock, including `"agent"`, then close them all in one pass.

#### 3.6 `_idle_guarded_stream` cleanup widens the asyncio cancellation surface

- **Location:** `lazybridge/agent.py:703-715`
- **Mechanism:** The `finally` swallows every exception from `gen.aclose()` (`except Exception: pass`). When the stream is aborted because the consumer broke early, `aclose` may need to `await` cleanup logic in the underlying provider — if that raises during shutdown, the agent silently loses the diagnostic. The current shape is fine for `CancelledError` but masks legitimate errors from buggy providers.
- **Fix:** Distinguish the two cases — let `CancelledError` propagate (it's a `BaseException`, so this already works), but log non-`CancelledError` exceptions as a `UserWarning`.

#### 3.7 `report_builder`: TOC text rendered with `autoescape=False`

- **Location:** `lazybridge/external_tools/report_builder/renderer.py:325-336` + `templates/deep_dive.html.j2:82`
- **Mechanism:** `extract_toc()` strips tags from heading HTML with `re.sub(r"<[^>]+>", "")` but does **not** decode-then-re-encode entities. The result is rendered through `Environment(autoescape=False)` as `{{ item.text }}` and `href="#{{ item.id }}"`. The `body_html` *is* bleach-sanitised upstream, but defense-in-depth is missing: any future relaxation of the bleach allowlist (or an upstream parser quirk that lets through a stray `&quot;`) becomes a TOC-shaped XSS.
- **Fix:** Add `|e` to `{{ item.id }}` and `{{ item.text }}` in `deep_dive.html.j2`. The body itself genuinely needs the `autoescape=False` (it's pre-sanitised HTML), but the TOC strings do not.

### P2 — nits / hygiene

#### 3.8 `ruff format` drift on two committed files

- **Location:** `lazybridge/agent.py`, `lazybridge/tools.py` (multiple line-wrap deltas)
- **Mechanism:** Manual edits during the audit fix introduced wrap shapes that `ruff format` would rejoin/re-wrap. Tests don't catch this; CI presumably does.
- **Fix:** `ruff format lazybridge/agent.py lazybridge/tools.py` and commit. Ten-line diff.

#### 3.9 `ToolDefinition.parameters` strict-mode emits `additionalProperties: false` only when there are required params

- **Location:** `lazybridge/core/tool_schema.py:1048-1052`
- **Mechanism:** When `strict=True` and the function takes zero required parameters, the resulting JSON Schema includes `"additionalProperties": false` but no `"required"` list — Gemini's strict-mode validator accepts this; OpenAI strict mode rejects it on some endpoints. Edge case (a strict tool with no required args is rare).
- **Fix:** Always emit `"required": []` alongside `"additionalProperties": false` when `strict` is requested.

#### 3.10 `Session.usage_summary` charges every model_response to the **last started** agent

- **Location:** `lazybridge/session.py:778-829`
- **Mechanism:** The "agent name for this run_id" map (`run_agent`) is built from `AGENT_START` events. If a Plan re-uses an outer agent's `run_id` for a sub-agent's MODEL_RESPONSE (which happens through the wrapping path), the model spend is attributed to the outer agent. Cosmetic for total spend; misleading per-agent.
- **Fix:** When emitting `MODEL_RESPONSE`, propagate the *innermost* agent name as `payload["agent_name"]` and prefer that over the run_id-mapped one when present.

#### 3.11 `_WebUI` HumanEngine: 1 s thread leak after timeout

- **Location:** `lazybridge/ext/hil/human.py:328-341`
- **Mechanism:** `asyncio.wait_for(timeout=timeout)` cancels the asyncio future but the underlying `response_q.get(timeout=timeout + 1)` runs in a daemon thread and continues blocking for the extra second. `server.shutdown()` runs in the `finally` and stops the HTTP listener, but the executor thread doesn't observe the cancellation. Daemon, so it dies with the process — minor.
- **Fix:** Drop the `+ 1` margin (use `timeout` for both deadlines), or replace the executor-shim with an `asyncio.Event` set from `do_POST`.

#### 3.12 `replay.py` race between `step()` and the auto-runner

- **Location:** `lazybridge/ext/viz/replay.py:140-167`
- **Mechanism:** `_run` publishes the event at line 165 *before* taking the lock to bump `_idx` at line 166-167. A concurrent `step()` (manual UI advance) takes the lock, observes the un-bumped `_idx`, and re-publishes the same event. Visualisation only — no engine state corruption.
- **Fix:** Move the publish call inside the same locked region as the increment.

#### 3.13 Memory token estimate is words, not tokens

- **Location:** `lazybridge/memory.py:142-143`
- **Mechanism:** S1 stores `len(text.split())` as the token estimate. Real LLM tokenisation runs ~1.3 tokens per English word, so the estimate undershoots by ~25 %. Compression triggers later than `max_tokens` would suggest. The framework documents `max_tokens=4000` as the default budget; users tuning this against real provider behaviour will observe a discrepancy on their own callers.
- **Fix:** Bump the conservative estimate (e.g. `int(words * 1.3)`) or document the convention explicitly: "this is a word-count heuristic; pass `tokens=` for true token counts."

## 4. Findings investigated and dismissed

These were flagged during exploration but did not survive verification:

- **`Memory._enforce_turn_cap` off-by-one (`<= max_turns`):** Inclusive cap is the documented contract ("Hard cap on retained turns") — `<=` is correct.
- **`EvalSuite.arun` swallows exceptions silently:** The inner `_run_one` wraps the body in `try/except Exception` and reports the failure as `EvalResult.error`. The outer `gather` doesn't need `return_exceptions=True` because individual cases never raise back to it.
- **OTel `flush()` with `batch=False` no-op claim:** The docstring already calls out that `SimpleSpanProcessor` flushes synchronously; behaviour matches.

## 5. Summary

The fix-pass landed honestly: every claimed capability is implemented and exercised by a focused test, ruff and the suite are green, and the resume / parallel-band tests directly assert the ordering invariant. The framework is structurally healthy.

The remaining issues fall into three buckets:

1. **Production-grade thread safety** — `register_provider_*` (P0) is a real race if anyone hot-registers providers under concurrency. Easy fix.
2. **Crash-window corner cases** — the parallel-band durable-write window (P1, §3.2) is a deliberate trade-off but under-documented. OTel orphan cleanup (P1, §3.5) and replay race (P2, §3.12) are minor and well-contained.
3. **Defensive surface** — MCP `assert`-gated state (P1, §3.4), the TOC autoescape gap (P1, §3.7), and the strict-mode JSON Schema shape (P2, §3.9) all matter when the framework runs against unfamiliar inputs or stripped builds.

The lint-format drift, the stale `_build.py` comment, and the misleading word-count estimate are pure hygiene fixes (~30 LOC total).
