# FragmentBus

`FragmentBus` is the runtime collector for the fragment-based reporting
pipeline.  Steps in a Plan share one bus; concurrent appends are
serialised through `Store.compare_and_swap`; state survives a crash and
resumes cleanly.

## Construction

```python
from lazybridge.external_tools.report_builder import FragmentBus, OutlineAssembler
from lazybridge.store import Store

# In-memory (default) — fragments live for the duration of the process.
bus = FragmentBus("daily-news")

# SQLite-backed — share with the Plan's checkpoint Store so resume works.
store = Store(db="./pipeline.sqlite")
bus = FragmentBus(
    report_id="daily-news-2026-05-04",
    store=store,
    assembler=OutlineAssembler({"1.us": "United States", "2.cn": "China"}),
)
```

| Argument     | Purpose                                                      |
|--------------|--------------------------------------------------------------|
| `report_id`  | Stable identifier — the Store key is `__report_fragments__:{report_id}`. |
| `store`      | Optional `Store`; defaults to a private in-memory Store.    |
| `assembler`  | Default `BlackboardAssembler()`; pass an `OutlineAssembler` for structured reports. |

The `report_id` is your handle for **idempotent reruns**: appending the
same `Fragment.id` twice is a silent no-op, and pointing a fresh bus at
the same `(store, report_id)` reads back the existing fragment list.

## API

### Append

```python
bus.append(Fragment(kind="text", body_md="Hello world", section="intro"))
```

* Returns `Fragment.id`.
* Thread-safe — call concurrently from `parallel=True` Steps.
* Idempotent — re-appending the same `id` is a no-op.

### Read

```python
all_frags        = bus.fragments()           # list[Fragment], ordered by created_at
intro_only       = bus.by_section("intro")   # list[Fragment], same order
count            = len(bus)                  # int
serialised       = bus.to_jsonl()            # debugging / log dump
```

### Assemble & export

```python
report = bus.assemble(title="My Report")     # AssembledReport
paths  = bus.export(                         # dict[format, Path]
    ["html", "pdf", "revealjs"],
    "./out",
    title="My Report",
    theme="cosmo",
    backend="auto",                          # "auto" | "quarto" | "weasyprint"
)
```

`export()` is a one-shot convenience — it runs the configured assembler
and dispatches to the chosen exporter.  When you need finer control
(e.g., a different assembler per output, or a custom CSS pass), call
`bus.assemble()` and feed the result into the exporter directly.

### Clear

```python
bus.clear()    # delete every fragment for this report_id
```

Useful in tests; rare in production.

## Thread-safety model

The bus stores the full fragment list under one Store key and updates it
through `compare-and-swap`:

1. Read current list.
2. Append the new fragment locally.
3. CAS-swap the new list against the read snapshot.
4. On miss (someone else wrote first), retry — bounded to 32 attempts.

This is a deliberate single-key choice — fragment volumes are O(10s–100s)
per report, so keeping the list contiguous is cheap.  If a real
contention storm hits the 32-attempt limit, that's a signal something
else is wrong (a misbehaving Store, or far more parallelism than the
bus was designed for) and you'll see a `RuntimeError` rather than silent
data loss.

```python
import threading
from lazybridge.external_tools.report_builder import FragmentBus, Fragment

bus = FragmentBus("concurrent-demo")

def emit(i: int):
    bus.append(Fragment(kind="text", body_md=f"frag-{i}"))

threads = [threading.Thread(target=emit, args=(i,)) for i in range(50)]
for t in threads: t.start()
for t in threads: t.join()

assert len(bus) == 50          # every write made it
```

## Persistence + resume

When you pass an SQLite-backed `Store`, the bus state lives in the
`store` SQLite table under key `__report_fragments__:{report_id}`.

Crash + restart:

```python
from lazybridge.store import Store
from lazybridge.external_tools.report_builder import FragmentBus, Fragment

# First run — partial.
store = Store(db="./run.sqlite")
bus = FragmentBus("rep-1", store=store)
bus.append(Fragment(kind="text", body_md="part 1"))
bus.append(Fragment(kind="text", body_md="part 2"))
# Process crashes before export.

# Second run — picks up where we left off.
bus2 = FragmentBus("rep-1", store=Store(db="./run.sqlite"))
print(len(bus2))           # 2 — fragments survived
bus2.append(Fragment(kind="text", body_md="part 3"))
bus2.export(["html"], "./out", title="Recovered")
```

The same Store the bus uses can be the Plan's checkpoint store.  When a
`Plan(resume=True)` re-runs a Step that already emitted fragments, those
appends are deduplicated by `Fragment.id`.

## Sharing one bus across many agents

The bus instance itself is the synchronisation point.  Pass it into
`fragment_tools(bus)` for every agent that contributes:

```python
bus = FragmentBus("multi")

us_agent  = Agent(model="…", tools=fragment_tools(bus=bus, default_section="2.us",  step_name="us"))
cn_agent  = Agent(model="…", tools=fragment_tools(bus=bus, default_section="2.cn",  step_name="cn"))
in_agent  = Agent(model="…", tools=fragment_tools(bus=bus, default_section="2.in",  step_name="in"))
synth     = Agent(model="…", tools=fragment_tools(bus=bus, default_section="1.exec", step_name="synth"))
```

`default_section` is a per-agent shortcut — fragments emitted via that
agent's tool calls without an explicit `section=` argument will land in
the default.  `step_name` becomes the `Provenance.step_name` stamped on
every fragment that agent appends.

## Caveats

* **Fragment list size.** The current single-key layout is comfortable
  through low-thousands of fragments per report.  If you're emitting tens
  of thousands, batch them through a synthesis Step rather than appending
  each one individually.
* **Cross-report bus.** A `FragmentBus` is scoped to one `report_id`;
  reuse the same bus across reports only if you want the union.
* **Hot reload.** Two processes pointing at the same SQLite file with
  different `report_id`s are isolated; with the same `report_id` they
  serialise through the SQLite reserved lock.
* **In-memory + crash.** With the default in-memory Store, a crash loses
  every fragment.  Pass an SQLite-backed Store any time you care about
  durability.

## See also

- [Fragment schema](report-fragments.md)
- [Assemblers](report-assemblers.md)
- [Store](store.md) — the persistence layer the bus is built on
