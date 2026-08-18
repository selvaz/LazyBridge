# Project patterns — the zero-boilerplate cookbook

`docs/for-llms/codegen-contract.md` says how to declare **one agent** and
**one tool**. This page is the layer above it: how to build a whole
*project* — a scheduled pipeline, a multi-phase workflow, a set of tools
over a real database — without hand-written glue.

**The rule this page enforces:** the only code you write by hand is domain
logic (pure functions) and a declaration of how the pieces connect. If you
are writing a `for` loop over steps, a `state` dict threaded through
closures, or a class that re-declares a library function's signature, you
have left the framework and are paying the boilerplate tax it exists to
remove.

**Provenance of every claim below.** Entries marked **[verified]** were
executed against the installed `lazybridge` (1.1.0) before being written
down. Entries marked **[source]** are read from the framework source with a
file citation but not executed. Nothing here is asserted from memory.

---

## Section 1 — Pipelines and composition

### 1.1 A pipeline with side effects is a `Plan`, not a loop

**Need:** run *fetch → analyse → persist → render → notify*, where most
steps are ordinary Python with side effects and only one calls an LLM.

**❌ Antipattern** (the real shape in `LazyStats/daily_anomaly.py::build_live_plan`
and `investmentcommittee/scripts/run_daily_anomaly_investigation.py`):

```python
state: dict[str, object] = {}

def explain():  state["expl"] = agent(...)
def persist():  state["rows"] = save(state["expl"])
def render():   state["html"] = to_html(state["rows"])
def send():     telegram(state["html"])

for name, step in build_live_plan(ctx, explain=explain, persist=persist,
                                  render=render, send=send):
    step()
```

A hand-rolled re-implementation of `Plan` with no checkpointing, no resume,
no cost roll-up, no session events, and an invisible data flow.

**✅ Canonical:**

```python
from lazybridge import Agent, Plan, Step

pipeline = Agent(
    engine=Plan(
        Step(explain_anomalies, name="explain"),
        Step(persist_batch,     name="persist"),
        Step(render_rows,       name="render"),
        Step(send_reports,      name="send"),
    ),
    name="daily_anomaly",
)

result = pipeline(gate_id)
if not result.ok:                      # <-- see §1.5, this is not optional
    raise SystemExit(f"pipeline failed: {result.error}")
```

**Mechanism [verified]:** `Step.target` accepts a *plain Python function*,
not only an `Agent` or a tool name (`engines/plan/_plan.py`, the
`elif callable(target)` branch). Sync functions are dispatched to the
default executor with the caller's contextvars copied in, so blocking I/O
in a step does not stall the event loop.

**Always pass `name=` [verified].** A `Step`'s default name is
`target.__name__`, so two unnamed `lambda` steps are both `"<lambda>"` and
the plan fails at construction with *"duplicate step name(s)"*. Explicit
names are also what `routes=`, `from_step()`, and checkpoints reference —
name every step, always.

**Do not use when:** the whole job is one function with no stages.

---

### 1.2 A complex phase is a nested sub-`Plan`

**Need:** one stage is itself five steps, and inlining them makes the
parent unreadable.

**✅ Canonical:**

```python
ingest = Agent(
    engine=Plan(Step(download), Step(validate), Step(normalise)),
    name="ingest",
)

pipeline = Agent(
    engine=Plan(
        Step(ingest,  name="ingest"),     # <- an Agent is a legal target
        Step(analyse, name="analyse"),
        Step(publish, name="publish"),
    ),
    name="daily",
)
```

**Mechanism [verified]:** when `Step.target` is an `Agent`, `Plan`
dispatches it via `target._run_as_tool(env)`. Cost and token metadata roll
up recursively through every nesting level (`_aggregate_nested_metadata`).

**The trade-off, measured [verified]:** a nested sub-plan is **atomic** to
the parent's checkpoint. In a direct test, a sub-plan failed on its second
inner step; on parent resume, the sub-plan re-ran **from the top** —
`inner_1` executed again even though it had already succeeded. Nest along
boundaries where "redo the whole phase" is acceptable and cheap.

To make the sub-plan resumable in its own right you need **all three**
settings, not two — `store=`, `checkpoint_key=`, **and `resume=True`**;
`resume` defaults to `False` and the checkpoint loader refuses to load
anything without it (`engines/plan/_checkpoint.py`, `_load_checkpoint`).

**Do not use when:** the sub-steps must be individually routable or
resumable from the parent. Flatten them.

---

### 1.3 Branching needs a predicate for *every* branch

**Need:** classify an input, handle it one way or another, rejoin a common
tail.

**❌ The trap [verified].** This looks right and is wrong:

```python
Step(classify, name="classify",
     routes={"urgent": lambda e: e.text() == "urgent"},   # only ONE branch
     after_branches="archive"),
Step(handle_urgent, name="urgent"),
Step(handle_normal, name="normal"),
Step(archive,       name="archive"),
```

Measured execution traces:

| input | trace |
|---|---|
| `"fire in the datacenter"` | `classify → urgent → archive` ✅ |
| `"routine check"` | `classify → urgent → normal → archive` ❌ |

When no predicate matches, routing **falls through linearly** to the next
declared step — which is the `urgent` handler. Both handlers run. And
because `after_branches` is registered only when a route actually fires, it
does not save you.

**✅ Canonical — predicates that exhaustively cover the input [verified]:**

```python
Step(classify, name="classify",
     routes={
         "urgent": lambda e: e.text() == "urgent",
         "normal": lambda e: e.text() != "urgent",   # catch-all, not a second exact match
     },
     after_branches="archive"),
```

Predicates are evaluated in declaration order and the **first match wins**
(`engines/plan/_plan.py`, `_route`) — matching two exact strings (`"urgent"`
/ `"normal"`) is *not* the fix; an unclassified value like `"unknown"` would
still match neither and fall through. What makes exactly one branch fire
every time is that the **last predicate is a true catch-all** (`!= "urgent"`),
so the route set covers the whole input space, not just the two values seen
in testing.

Traces after the fix: `classify → urgent → archive` and
`classify → normal → archive` for every input, including ones neither
branch was written to expect.

`routes_by="field"` lets the **LLM** pick the branch by returning a
`Literal` field on its structured output — there every declared `Literal`
member has an implicit exact-match branch, so exhaustiveness comes from the
type, but a value that matches no step name (including `None`) still falls
through linearly, same as above.

**Do not use when:** the branch is a genuine early exit with no rejoin —
put the terminal step last and let the plan end.

---

### 1.4 A scheduled job needs a **run-specific** checkpoint key

**Need:** a nightly pipeline that must not redo completed, expensive, or
side-effecting steps after a crash.

**❌ The trap [verified] — this runs exactly once, ever:**

```python
Plan(..., store=store, checkpoint_key="nightly", resume=True)
```

Measured: three consecutive "nightly" runs executed the work function
**once**. Runs 2 and 3 returned the *cached* payload from the first run's
checkpoint without executing a single step. A completed checkpoint under
`resume=True` is deliberately not re-claimed, and the plan short-circuits
to the stored `kv`. A permanent key plus `resume=True` silently converts a
daily job into a one-shot job.

**✅ Canonical — the key names the *run*, not the pipeline:**

```python
Plan(
    Step(fetch,   writes="fetched"),
    Step(analyse, writes="analysis"),
    Step(publish, writes="published"),
    store=Store(db="run.sqlite"),
    checkpoint_key=f"nightly:{run_date:%Y-%m-%d}",   # <-- run-specific
    resume=True,
)
```

Now a crashed run of 2026-08-17 resumes on retry, while 2026-08-18 is a
fresh key and runs from the start. Verified separately: after a step
raises, a `resume=True` rebuild re-runs **only** the failed step.

**Concurrency, precisely [source]:** `on_concurrent="fail"` is *not* a
live-run mutex when `resume=True`. A second process with `resume=True`
**adopts** an existing in-flight checkpoint by overwriting its `run_uid`
(`engines/plan/_checkpoint.py`); the original process only discovers the
takeover when its next CAS fails, so both can execute side effects
concurrently in the meantime. If two runs of the same key must never
overlap, take a lock outside the Plan.

`on_concurrent="fork"` gives each run its own suffixed key — but for
fan-out, prefer `run_many` (§1.6) over hand-rolling it.

---

### 1.5 Check `result.ok` — a failed pipeline returns, it does not raise

**The failure mode [verified]:** `_exec_step` catches ordinary exceptions and
converts them into an **error envelope**; the Plan checkpoints the failure
and *returns* that envelope rather than raising
(`engines/plan/_plan.py`). A scheduled script that calls
`pipeline(task)` and ignores the result **exits with status 0 after a
failed fetch, a failed persist, or an unsent report** — and your scheduler
records a green run.

**✅ Canonical:**

```python
result = pipeline(task)
if not result.ok:
    raise SystemExit(f"{pipeline.name} failed: {result.error}")
```

The same applies to `fallback=` (§4.1): it fires on a *returned* error
envelope, not on a raised exception.

---

### 1.6 Fan-out over N inputs is `run_many`, not an executor

**❌ Antipattern:** the `ThreadPoolExecutor` + `asyncio.run` driver loop.

**✅ Canonical [verified]:**

```python
plan = Plan(Step(fetch), Step(score), on_concurrent="fork",
            store=store, checkpoint_key="per_ticker")

envelopes = plan.run_many(tickers, concurrency=8)
```

Signature verified: `run_many(tasks, *, concurrency=None, tools=None,
memory=None, session=None, output_type=str) -> list[Envelope]`, with an
async twin `arun_many`. It always uses `asyncio.gather`, bounded by a
semaphore when `concurrency` is set; it does **not** vary by
`on_concurrent`. Results come back **in input order**, and a raised
exception is wrapped as an error envelope rather than lost — so check each
envelope's `.ok`, not just the call.

**Pass `tools=` when any step uses a string-name target.** Omitting it
works only when every step target is an `Agent` or a plain callable.

**Independent stages over the same input** are a parallel band instead —
contiguous `parallel=True` steps run concurrently and rejoin linearly:

```python
Step(fetch_prices, name="prices", parallel=True, writes="prices"),
Step(fetch_news,   name="news",   parallel=True, writes="news"),
Step(combine,      name="combine", task=from_parallel_all("prices")),
```

**Two band semantics that bite [verified]:** routing is **ignored** on
parallel branches (control always falls through linearly after the band),
and a band is **atomic** — if any branch fails or pauses, *no* branch's
writes are applied and the whole band re-runs on resume.

---

### 1.7 "Not ready yet" is `PlanPaused`, not a failure and not a sleep

**Need:** a step cannot proceed because a webhook has not arrived, an
upstream file is missing, or a human has not approved.

**✅ Canonical [verified]:**

```python
from lazybridge.engines.plan import PlanPaused

def await_settlement(task: str) -> str:
    if not settlement_available():
        raise PlanPaused("waiting for settlement file")
    return process(task)
```

The Plan writes a `paused` checkpoint pointing at the **same step** and
returns a retryable error envelope, so a later `resume=True` run re-invokes
exactly that step. This is the declarative alternative to polling inside a
step (which burns the run's wall clock) or treating "not ready" as a
failure (which pollutes your alerting).

**⚠️ The pause only persists if there is somewhere to persist it.** The
Plan must carry **`store=` and `checkpoint_key=`**, or the pause is
returned to the caller and then lost — the next run starts from the
beginning. The full working shape is:

```python
Plan(
    Step(prepare,        name="prepare"),
    Step(await_settlement, name="wait"),
    Step(book,           name="book"),
    store=Store(db="run.sqlite"),
    checkpoint_key=f"settlement:{run_date:%Y-%m-%d}",
    resume=True,
)
```

Verified: the first run executes `prepare` then `wait` and stops; the
resumed run executes `wait` then `book` — the paused step is re-invoked,
not skipped.

`PlanPaused` subclasses `BaseException`, so a stray `except Exception` in
your own step code will not swallow it.

---

## Section 2 — Tools and data

### 2.1 Move a handle, not a payload — and know exactly where the boundary is

**The mechanism, stated precisely [verified].** Two different things travel
between steps, and conflating them is how raw data ends up in a prompt:

| Field | What it carries | Who receives it |
|---|---|---|
| `task` | **always a string** — `prev.text()` of the upstream step | every target |
| `payload` | the **typed Python object**, preserved | `Agent` targets only |

Verified both directions: a step returning `{"series": list(range(500))}`
delivers the **full typed dict** to a downstream `Agent` target's engine
(500 elements intact), while a downstream **plain function** receives a
`str` and nothing else.

The consequence that matters: `task` is what an LLM step turns into prompt
text. So a large upstream payload **does** reach the model — through
`text()`, not through `payload`. Keeping data out of the context window is
therefore a discipline about what your steps *return*, not something the
envelope does for you.

**❌ Antipattern** (real, `investmentcommittee/.../daily_anomaly_live.py`):

```python
agent(PROMPT.format(artifact=json.dumps(artifact, indent=1)))
```

Every target, item, and z-score serialised into the prompt.

**✅ Canonical — the data stays put, the handle travels:**

```python
def load_into_depot(task: str) -> str:
    depot.write("run42", heavy_dataframe)
    return "run42"                       # only the handle moves

def consume_by_handle(key: str) -> str:
    df = depot.read(key.strip())
    return f"rows={len(df)}"             # a summary, not the data
```

**The reference implementation to copy** is `LazyCrawler`'s tool surface:
`web_search` returns truncated snippets **plus** a `session_id`/`url`, and
the model must call `get_page(url)` to pull full text on demand.

**⚠️ An in-process depot does not survive a restart [source].** Checkpoints
persist step history and `writes` values but explicitly **not** live
in-memory state. After a crash, resume skips the completed loader step and
hands the consumer `"run42"` — while the new process's depot is empty. If
the pipeline is resumable, the depot must be **durable** (a Store, a file,
a database), not a module-level dict.

**Do not use when:** the value genuinely is small and the model must reason
over its contents. A handle to a 12-character string is pure overhead.

---

### 2.2 A tool over a library function is one line

**❌ Antipattern:** a bridge class whose methods re-declare the library
function's signature, rebuild its docstring as `description=`, and
re-implement its data loading.

**✅ Canonical:**

```python
Tool.wrap(lazystats.regression.fit_ols, name="fit_ols")
```

Put `Annotated[type, "description"]` on the **library function's**
parameters, in the library — `Tool.wrap` reads them natively.

A wrapper is justified only for what the library cannot know: an
LLM-context output cap on an unbounded result, or an envelope/provenance
shape. Even then it must *call* the unmodified function.

**Do not use when:** the schema is already known and not derivable from a
Python signature (MCP, OpenAPI) — use `Tool.from_schema(...)`.

---

### 2.3 Tools that share expensive state share a depot key

**✅ Canonical:** one loader tool populates the depot and returns the key;
every other tool takes `data_key: str` and reads by it.

**Non-obvious constraint [verified]:** `Step(writes="k")` is **not** a
channel for this. No sentinel reads the `writes` bucket — passing
`from_memory("k")` where `k` is a writes-key fails at **compile time** with
*"references tool 'k' which is not in the tool map"*. `writes=` exists for
durability, resume, and external Store consumers.

Sentinel meanings, easy to confuse:

| Sentinel | Actually means |
|---|---|
| `from_prev` (default) | previous step's output becomes this step's task |
| `from_step("name")` | that named step's output, skipping intermediates |
| `from_parallel_all("name")` | labelled aggregate of a whole parallel band |
| `from_agent("name")` | that agent's last output, read from the shared Store |
| `from_memory("name")` | that **agent's conversation memory**, as context |

**⚠️ `from_agent` fails open, two different ways [verified].** An agent's
output is written to the Store only after a *successful* run
(`Agent._run_body`) — a failed run never overwrites what is already there.
That makes the failure mode depend on history, and a naive presence check
only catches one of the two shapes:

* **Key never written** (first-ever run, or the agent never ran) —
  resolution returns an **empty envelope with no error**. A presence
  assertion in the consuming step catches this.
* **Key written by an earlier successful run, current run's source agent
  fails** — resolution returns that **earlier run's value**, not empty,
  because nothing overwrote it. A presence assertion **passes** here and
  the downstream stage silently processes stale data with no indication
  anything is wrong.

For a genuinely cross-run dependency, presence alone is not enough — pair
it with a freshness or provenance check (e.g. a run-specific key, or a
timestamp/run-id written alongside the value) so the consumer can tell
*this run's* output from a stale one. When the dependency is within the
*same* plan, prefer `from_step` instead — it reads the step's envelope
directly rather than the Store's last-good value, so a failed step is
never silently masked by an older success.

---

## Section 3 — Artifacts across processes

### 3.1 One registry, addressed by convention

**The problem, measured:** the ecosystem currently has at least ten
independent persistence locations (`market_data.duckdb`,
`market_data_artifacts.db`, `news.db`, `crawler_artifacts.db`,
`result_depot.sqlite`, `regime_depot.db`, `anomaly_explanations.sqlite`,
`tree_studio.sqlite3`, `lazyportfolio_artifacts.db`, plus an undeclared
145 MB `run_cache.sqlite3`) and **zero** shared registry.
`config/databases.toml` is a hand-maintained audit document no pipeline
reads.

**✅ Canonical:** artifacts that outlive a single agent go to one `Store`,
under a reconstructable key, with `agent_id` for provenance:

```python
store.write(
    f"artifact:{domain}:{kind}:{run_date}:{content_hash[:12]}",
    payload,
    agent_id=producer_name,
)
```

**Verified end to end:** a prefix scan over `keys()` discovers artifacts
from any producer; `read_entry()` returns the `agent_id` provenance;
`compare_and_swap()` lets the first writer win and rejects a stale one; and
a second `Store` handle on the same file — a separate process — sees
everything. A registry is a *naming convention plus CAS*, not new
infrastructure.

Publishing a "current head" safely is the same CAS pattern LazyPortfolio's
tree revisions already use:

```python
if not store.compare_and_swap(f"head:{domain}", expected_rev, new_rev):
    raise RuntimeError("another writer published first — re-read and retry")
```

**⚠️ Three limits you must design around [source]:**

1. **The Store is a JSON round-trip.** It normalises Pydantic models and
   nests dicts/lists, but `write()` ultimately uses
   `json.dumps(..., default=str)`. A DataFrame, ndarray, domain object, or
   binary artifact is **silently persisted as its string representation**.
   Store a *reference* (a parquet path, a content hash) and keep the bytes
   in a real artifact store.
2. **There is no TTL, expiry, or compaction.** A content-hashed key per
   changed payload grows forever; the only tools are manual `delete()` and
   whole-store `clear()`. Decide a retention rule when you adopt the
   convention, not after the file reaches 145 MB.
3. **It is plaintext SQLite.** For credentials, private source material, or
   anything regulated, wrap it — `lazybridge/store/encryption.py` ships
   `EncryptedStoreAdapter` with authenticated at-rest encryption and key
   rotation.

**Do not use when:** the value is scoped to one agent's own run — that is
what the agent's `Memory`/`Store` already is.

**Open question, undecided:** whether the ecosystem adopts one physical
Store or one convention over several. `investmentcommittee/reports/db.py`
(four tables, `version_id`/`content_hash`) is the closest existing design,
but it is referenced only from tests and its declared database file did not
exist at last capture. Do not treat it as the standard until something
actually writes to it.

---

### 3.2 Your process alone cannot make an external effect exactly-once

Two things get conflated here, and the difference is the whole section:

- **exactly-once *delivery*** — the request crosses the network exactly
  once. Unachievable. You can never know whether a request that timed out
  was received.
- **exactly-once *effect*** — the remote system ends in the same state no
  matter how many times you send. Achievable, but **only with the
  receiver's cooperation**.

Everything below is about buying the second one, or choosing which failure
you prefer when you cannot.

**The failure mode [source]:** a step's target runs **before** the Plan
appends history and saves its checkpoint. If `send_reports` successfully
sends and the process dies before `_save_checkpoint`, a resumed run
**sends it again**. `writes=` cannot atomically commit an external effect
together with the checkpoint.

**❌ The pattern that looks like a fix and is not:**

```python
def send_reports(task: str) -> str:
    key = f"{run_date}:{report_id}"
    if store.read(f"sent:{key}"):
        return "already sent"
    telegram_send(render(task))        # <-- crash HERE
    store.write(f"sent:{key}", True)   # <-- and this never runs
    return "sent"
```

This has the *same* crash window it claims to close, just narrower: die
between the send and the write and the resumed run sends again. No
check-then-act sequence in your own process can be atomic with an effect
that happens in someone else's.

**✅ What actually works — pick your failure, or push dedup to the receiver:**

1. **Reserve before acting** (at-most-once). Claim the key with
   `compare_and_swap` *first*, then send. A crash after the claim but
   before the send means the report is **never** sent — so this trades
   duplicates for silent omissions, and needs a reconciliation sweep over
   claimed-but-unconfirmed keys.

   ```python
   if not store.compare_and_swap(f"sent:{key}", None, "in-flight"):
       return "already claimed"
   telegram_send(render(task))
   store.write(f"sent:{key}", "confirmed")
   ```

2. **Receiver-side idempotency key** (exactly-once *effect*). Pass a
   deterministic key the *remote system* dedupes on — the mechanism payment
   and messaging APIs provide for exactly this. Your process may retry
   freely; the receiver collapses the duplicates.

   **The guarantee is only as good as the receiver.** It holds only if the
   receiver records the key **atomically with the effect** (same
   transaction). A receiver that applies the effect and then stores the key
   has merely moved your crash window to its side of the wire. Read the
   API's idempotency documentation for that property specifically — and if
   the receiver offers nothing (a plain webhook, a Telegram send), this
   option is not available to you and you are choosing between (1) and
   at-least-once.

3. **Make the effect naturally idempotent.** An upsert to a known row, a
   write to a content-addressed path, or a message whose delivery is keyed
   by content hash. Prefer this whenever the effect's shape allows it.

**The decision to make consciously:** for a report, at-least-once (a
duplicate Telegram message) is usually the cheaper failure. For money or an
order, at-most-once plus reconciliation is — unless the receiver gives you
a real idempotency key, in which case take it. Nothing in the Plan chooses
for you — and `writes=`/checkpointing does not, either.

Put irreversible effects as **late** in the plan as possible, and behind a
human gate (§4.3) when the cost of being wrong is high.

---

## Section 4 — Reliability, operation, and tests

The `Agent` constructor already carries the knobs most projects
re-implement by hand:

| Instead of hand-writing… | Declare |
|---|---|
| `try: primary() except: backup()` around a **Plan step or provider call** | `fallback=backup_agent` — narrower than it looks, see §4.1 |
| a retry loop around a flaky provider | `max_retries=`, `retry_delay=` |
| parsing an LLM's prose back into fields | `output=MyPydanticModel` |
| an `assert` + retry when output is malformed | `output_validator=`, `max_output_retries=` |
| a second agent call to check the first | `verify=judge_agent`, `max_verify=` |
| an input/output safety filter | `guard=` |
| a wall-clock guard | `timeout=` |
| hand-crafting provider cache-control blocks for the system prompt | `cache=True` / `CacheConfig(ttl="1h")` |

### 4.1 `fallback=` — and the boundary that will surprise you

```python
primary = Agent(engine=LLMEngine("claude-opus-4-7"), name="primary",
                fallback=Agent(engine=LLMEngine("gpt-5.4-mini"), name="backup"))
```

**The boundary, measured — it is narrower than it looks:**

| What raises | Result |
|---|---|
| a **step target** inside a `Plan` | converted to an error envelope → **fallback fires** ✅ |
| a provider / LLM call | arrives as an error envelope → **fallback fires** ✅ |
| a **tool** called by an `LLMEngine` | becomes a *tool result* handed back to the model → **does not itself trigger fallback**; the model retries or works around it ⚠️ |
| the agent's **own engine**, at engine level | propagates to the caller → **fallback does not fire** ❌ |

Verified: a step function raising `RuntimeError` inside a `Plan` reached the
backup agent and the run returned `ok=True`; a custom engine raising
directly propagated past the fallback.

**Do not read this as "fallback covers my raising tools."** It covers a
raising *Plan step*. A tool that raises inside an LLM's tool loop is a
different path entirely — the exception is surfaced to the model as a tool
result, so the agent keeps going and `fallback=` never sees *that* event.

The nuance that keeps this from being absolute: a tool failure can still
reach the fallback **indirectly**, if it causes the engine to end in an
error envelope of its own — a model that keeps retrying a broken tool until
it hits `MaxTurnsExceeded`, for instance. So the guarantee runs one way
only: an error envelope reliably triggers the fallback; a tool exception
does not reliably produce one. If a tool failure must abort the run, return
a typed failure and check it in a step, rather than relying on the exception
to escalate.

Because engine-level raises are the one row in the boundary table where
`fallback=` **does not fire**, `fallback=` is not a blanket replacement for
`try`/`except` — an agent built on a custom engine that can raise directly
(rather than returning an error envelope) still needs its own
`try`/`except` around the call site if a caller must survive that failure.

Chains are cycle-checked at construction.

---

### 4.2 Test the wiring with `MockAgent`, at zero cost

**Need:** assert a multi-step plan wires its steps together correctly
without paying for LLM calls in CI.

**✅ Canonical [verified]:**

```python
from lazybridge.testing import MockAgent

research = MockAgent(responses=["research-key"], name="research")
write    = MockAgent(responses=["done"],         name="write")

pipeline = build_pipeline(research=research, write=write)
result   = pipeline("run-42")

research.assert_call_count(1)
write.assert_called_with(contains="research-key")
assert result.ok
```

`MockAgent` is a drop-in `Step` target and records every handoff
(`call_count`, `last_call`, `assert_called_with`, `reset`) — so the
*wiring* is assertable, not just the final string. **Build your pipeline
behind a factory that takes its agents as arguments**, so tests can inject
mocks; a pipeline that constructs its own agents inline is untestable
without a provider.

This is the cheap regression test that would have caught the Node Advisor
defect where a shipped LLM path was never reachable from the live worker.

For output *quality* rather than wiring, `lazybridge.ext.evals` ships
`EvalSuite`/`EvalCase` with `contains`, `exact_match`, `max_length`,
`min_length`, `not_contains`, and `llm_judge`.

---

### 4.3 An irreversible step sits behind a human gate

**❌ The trap [verified] — this executes the orders after a rejection:**

```python
Plan(
    Step(prepare_orders, name="prepare"),
    Step(approval,       name="approve",
         routes={"execute": lambda e: e.text().startswith("approve")}),
    Step(execute_orders, name="execute"),          # <-- runs anyway
)
```

This is §1.3's rule with money attached. A rejection matches no predicate,
so routing **falls through linearly** — to `execute`. Measured trace for a
`"reject: too risky"` verdict: `prepare → approve → execute`, and the
orders went out.

Note that adding an explicit `halt` step is **not** sufficient on its own:
if `halt` sits before `execute` in declaration order, linear progression
carries on into `execute` after it. Verified — the naive fix still executed.

**✅ Canonical [verified] — a predicate per branch, both rejoining:**

```python
from lazybridge.ext.hil import HumanEngine

approval = Agent(
    engine=HumanEngine(timeout=300, ui="terminal", default="reject"),
    name="approve",
)

Plan(
    Step(prepare_orders, name="prepare"),
    Step(approval, name="approve",
         routes={
             "execute": lambda e: e.text().strip().lower() == "approve",
             "halt":    lambda e: e.text().strip().lower() != "approve",
         },
         after_branches="finish"),
    Step(execute_orders, name="execute"),
    Step(halt,           name="halt"),
    Step(finish,         name="finish"),
)
```

Measured traces:

| verdict | trace | orders executed |
|---|---|---|
| `"reject: too risky"` | `prepare → approve → halt → finish` | **no** |
| `"approve"` | `prepare → approve → execute → finish` | once |

Set `default=` to the **safe** outcome, not the convenient one: it is what
happens when the timeout expires unattended. For an irreversible action,
write the rejection branch first and test it first.

**A second trap sits inside the predicate itself, not just the routing.**
`HumanEngine` collects unrestricted free text, so a substring check like
`e.text().startswith("approve")` — the tempting shorthand — is not safe on
its own: `"approve? no"` or `"approved: rejected"` also starts with
`"approve"` and would fire the execute branch on a rejection. Route on an
exact normalized token (as above) or, better, a structured decision. Note
`output=` is not a `HumanEngine` constructor argument (it only takes
`timeout=`, `ui=`, `default=`) — the schema is assigned to the **Agent**
wrapping it, and to the **Step**, whose `output` model `routes_by=` reads:

```python
from typing import Literal
from pydantic import BaseModel

class Decision(BaseModel):
    # routes_by jumps straight to the step named by the field's VALUE —
    # there is no predicate layer to translate "approve"/"reject" into a
    # target step name, so the Literal values must BE the step names.
    decision: Literal["execute", "halt"]

approval = Agent(
    engine=HumanEngine(timeout=300, ui="terminal", default='{"decision":"halt"}'),
    output=Decision,
    name="approve",
)

Plan(
    Step(prepare_orders, name="prepare"),
    Step(approval, name="approve", output=Decision, routes_by="decision", after_branches="finish"),
    Step(execute_orders, name="execute"),
    Step(halt,           name="halt"),
    Step(finish,         name="finish"),
)
```

**[verified]** — a `Literal` value that isn't a known step name (e.g. the
tempting `Literal["approve", "reject"]` naming, left as-is) fails at
**compile time**: `routes_by='decision' includes Literal value 'reject'
which is not a known step name`. Caught before the plan ever runs, not a
runtime surprise.

---

### 4.4 One pipeline, one `Session`

**Need:** know what a scheduled multi-agent run actually did — which steps
ran, which tools were called, what it cost — without print statements.

**✅ Canonical:** attach one `Session` at the root agent. `Plan` emits
`AGENT_START`/`AGENT_FINISH` and per-step `TOOL_CALL`/`TOOL_RESULT`/
`TOOL_ERROR` events sharing one `run_id`. Cost rolls up transitively via
`metadata.nested_cost_usd`.

**⚠️ Inheritance is not automatic for `Step` targets [verified].** Agents
passed in `tools=[...]` inherit the parent's session (as do `fallback=` and
`verify=` agents, with edge labels distinguishing provenance). An agent used
**directly as a `Step` target does not**.

Measured, with a sub-agent whose engine genuinely emits spans:

| wiring | `agent_start` spans in the log |
|---|---|
| `Agent(..., name="sub")` — no `session=` | `{root}` |
| `Agent(..., name="sub", session=sess)` | `{root, sub}` |

Pass it explicitly wherever you build a sub-agent for a `Step`:

```python
sub  = Agent(engine=Plan(...), name="sub", session=sess)   # <-- explicit
root = Agent(engine=Plan(Step(sub, name="sub")), name="root", session=sess)
```

**Two details that make logs confusing if you don't know them:** the *engine*
emits the agent span, not `Agent` — so a custom engine that emits nothing
leaves its agent invisible no matter how the session is wired. And `Plan`
tags its per-step `tool_call`/`tool_result` events with the **parent's**
`agent_name` plus a `step` field, so seeing the step name in the log is not
evidence that the sub-agent reported for itself.

This is the observability hole that makes a scheduled multi-agent run look
half-empty in its own event log.

Do **not** pass `runtime=`, `resilience=`, or `observability=` config
objects — those were deleted in 0.7.9. Pass flat kwargs.

---

## Quick reference — pick the pattern

| You are about to write… | Use instead | § |
|---|---|---|
| `for step in steps: step()` | `Plan(Step(...), ...)` | 1.1 |
| a `state = {}` threaded through closures | `Step` chaining / `writes=` | 1.1 |
| a deeply nested plan nobody can read | a sub-`Agent(engine=Plan(...))` | 1.2 |
| `if kind == "urgent": ...` around agent calls | `routes=` **with every branch** | 1.3 |
| a `--force` flag for a crashed job | `checkpoint_key=f"job:{date}"` + `resume=True` | 1.4 |
| nothing, after calling the pipeline | `if not result.ok: raise` | 1.5 |
| `ThreadPoolExecutor` over N tickers | `plan.run_many(tasks, concurrency=8)` | 1.6 |
| `while not ready: sleep(60)` | `raise PlanPaused(...)` | 1.7 |
| `json.dumps(big_thing)` into a prompt | a handle + a summary | 2.1 |
| a class mirroring a library signature | `Tool.wrap(fn, name=...)` | 2.2 |
| each tool loading its own copy of the data | one loader + `data_key: str` | 2.3 |
| `sqlite3.connect("my_new_thing.db")` | the shared artifact registry | 3.1 |
| a send/publish with no dedup | reserve-before-acting, or a receiver-side idempotency key | 3.2 |
| `try: primary() except: backup()` | `fallback=` | 4.1 |
| a real LLM call in a unit test | `MockAgent` + a pipeline factory | 4.2 |
| an irreversible action with no gate | `HumanEngine` with a safe `default=` | 4.3 |
| `print()` in a scheduled job | one `Session` at the root | 4.4 |

---

## The one gap the framework does not close

Every need above has a declarative primitive **except one**: a typed,
checkpoint-aware artifact channel between steps.

The resolver preserves an upstream envelope's typed `.payload`, but
dispatch collapses it back to text for named tools and plain callables, and
the `Store` cannot durably hold a DataFrame (JSON round-trip, `default=str`).
`Step.input`/`Step.output` validate declared compatibility at *compile*
time without creating a typed runtime handoff. That is precisely why §2.1
has to paper over the gap with a user-owned `depot` variable, and why §2.1's
warning about restarts exists at all.

Closing it would mean something like `produces="prices"` on a `Step`
(storing the real payload via an explicit codec and checkpointing an
`ArtifactRef`) plus a `from_artifact("prices")` sentinel that deserialises
for a Python target while giving an LLM only the reference and a bounded
summary — with unsupported types failing at Plan construction instead of
silently becoming strings. It would live in `lazybridge/artifacts.py`,
`sentinels.py`, and the Plan resolver.

Until that exists, §2.1 and §3.1 are the supported workaround, and their
warnings are load-bearing.
