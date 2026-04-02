# LazyStore

Shared key-value blackboard. All values are stored as JSON and must be JSON-serializable.

## Constructor

```python
LazyStore(db: str | None = None)
```

| `db` value | Backend | Persistence |
|------------|---------|-------------|
| `None` | `_InMemoryBackend` | Process-local, lost on exit |
| `"path.db"` | `_SQLiteBackend` | Persistent, WAL mode |

---

## Write

```python
store.write(key: str, value: Any, *, agent_id: str | None = None) -> None
```

Overwrites any existing entry for `key`. `agent_id` is optional metadata used for filtering.

```python
store[key] = value    # equivalent; no agent_id
```

---

## Read

```python
store.read(key: str, default: Any = None) -> Any
store[key]            # raises KeyError if absent
```

```python
store.read_entry(key: str) -> StoreEntry | None
```

Returns the full `StoreEntry` including metadata, or `None` if the key does not exist.

```python
store.read_all() -> dict[str, Any]
store.read_by_agent(agent_id: str) -> dict[str, Any]
store.keys() -> list[str]
store.entries() -> list[StoreEntry]
```

---

## Mutation

```python
store.delete(key: str) -> None
store.clear() -> None
```

---

## Containment

```python
"key" in store    # True if key exists and value is not None
```

---

## Text rendering

```python
store.to_text(keys: list[str] | None = None) -> str
```

Returns a formatted string suitable for injection into a system prompt. Used internally by `LazyContext.from_store()`.

```
[shared store]
  key1: value1
  key2: value2
```

---

## Async API

Use the async methods when writing to or reading from the store inside `async def` agent code. The synchronous SQLite backend would otherwise block the event loop and starve parallel agents running in the same loop (e.g. via `asyncio.gather` or `LazySession.gather()`).

```python
store.awrite(key: str, value: Any, *, agent_id: str | None = None) -> None
store.aread(key: str, default: Any = None) -> Any
store.aread_all() -> dict[str, Any]
store.akeys() -> list[str]
```

Each method offloads the synchronous call to the default thread-pool executor so the loop stays free during the I/O wait.

```python
async def run():
    sess = LazySession(db="pipeline.db")
    await sess.store.awrite("result", {"status": "ok"}, agent_id="agent-1")
    value = await sess.store.aread("result")
    all_data = await sess.store.aread_all()
```

The in-memory backend benefits from thread-pool offloading too (lock contention under `asyncio.gather`). Use the async API consistently in all async contexts.

---

## StoreEntry

```python
@dataclass
class StoreEntry:
    key:        str
    value:      Any
    agent_id:   str | None
    written_at: datetime    # UTC
```

---

## SQLite backend details

| Setting | Value | Purpose |
|---------|-------|---------|
| Journal mode | WAL | Concurrent readers + one writer |
| `check_same_thread` | `False` | Safe with WAL + per-thread connections |
| `busy_timeout` | 10000 ms | Avoids `SQLITE_BUSY` errors under load |

---

## Usage pattern (Pattern C)

Agent A writes results; Agent B reads them without any direct reference to Agent A.

```python
from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore()

# Agent A writes results
agent_a = LazyAgent("anthropic", name="analyst")
agent_a.loop("analyse market data for Q4")
store.write("q4_analysis", agent_a._last_output, agent_id=agent_a.id)

# Agent B reads without knowing about agent A
ctx = LazyContext.from_store(store, keys=["q4_analysis"])
agent_b = LazyAgent("openai", name="writer", context=ctx)
agent_b.chat("write an executive summary")
```
