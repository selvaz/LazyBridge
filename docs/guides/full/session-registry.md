# Session aliases and `SessionRegistry`

Both coding-agent engines can keep a conversation alive across processes:
`CodexEngine(thread_id=..., persist_thread=True)` and
`ClaudeCodeEngine(session_id=..., persist_session=True)`. The native id is an
opaque UUID the caller has to store somewhere. `SessionRegistry` is that
somewhere, once, for every caller: a small JSON file that maps a **name you
choose** to the native id, so a launcher can say "use session `reviewer`" and an
operator can rename or rebind it by hand.

```python
from lazybridge import Agent, CodexEngine

engine = CodexEngine(model="gpt-6-luna", reasoning_effort="low", cwd=repo, session_alias="reviewer")
Agent(engine, name="r")("Remember the word PINEAPPLE.")       # binds reviewer -> <thread id>

# any later process, any engine instance in the same cwd:
engine = CodexEngine(model="gpt-6-luna", reasoning_effort="low", cwd=repo, session_alias="reviewer")
Agent(engine, name="r")("What was the word?")                  # resumes the same thread
```

`ClaudeCodeEngine` takes the same `session_alias=` and `session_registry=`
keywords. (`ClaudeCodeEngine.session_name` is unrelated: it is the human title
written onto the session on disk.)

## The registry

```python
from lazybridge import SessionRegistry

reg = SessionRegistry()                      # path: arg > $LAZYBRIDGE_SESSIONS_FILE > ~/.lazybridge/sessions.json
reg.resolve("codex", repo, "reviewer")       # native id, or None
reg.bind("codex", repo, "reviewer", "0198...", model="gpt-6-luna", effort="low")
reg.rename("codex", repo, "reviewer", "reviewer-v2")   # KeyError if missing, ValueError if target exists
reg.forget("claude", repo, "scratch")        # True if it existed
reg.entries(kind="codex", scope=repo)        # list[dict]: kind, scope, name, native_id, model, effort, created_at, updated_at
```

| Part | Rule |
|---|---|
| Key | `(kind, scope, name)`; `kind` is `"codex"` or `"claude"` |
| Scope | the engine's `cwd` (or the working directory), as a resolved absolute posix path, compared case-insensitively on Windows |
| Name | `^[A-Za-z][A-Za-z0-9_.-]{0,63}$`, otherwise `ValueError` |
| Value | `native_id`, `model`, `effort`, `created_at`, `updated_at` |

The scope is part of the key because both CLIs store their sessions per working
directory: the same name in two repositories is two different sessions.

Storage is dependency-free. Writes are atomic (temp file, `fsync`,
`os.replace`), serialised across threads by a lock and across processes by an
exclusive lock file next to the registry (a lock older than 30 s is taken over).
A missing file is an empty registry. A file that does not parse is renamed to
`sessions.json.corrupt` (timestamped if one already exists) and the registry
starts empty -- it is never silently overwritten.

`default_session_registry()` returns the process-wide instance engines use when
no `session_registry=` is passed; `set_default_session_registry(reg)` replaces
it (and returns the previous one) for tests.

## How an engine resolves the alias

| You pass | Result |
|---|---|
| alias only, name is known | the recorded id is adopted and persistence is switched on |
| alias only, name is unknown | a fresh durable session; the alias is bound after the first turn |
| alias **and** an explicit `thread_id` / `session_id` | the explicit id wins and the alias is (re)bound to it |
| no alias | unchanged behaviour |

The engine binds as soon as it learns the native id -- including on a turn that
fails or times out, as long as an id was obtained -- and records the `model` and
`reasoning_effort` it ran with. The Claude SDK can hand back a different id for
a resumed session, so the alias follows it. Binding never fails a turn: an
unwritable registry logs a warning and the run continues.

Known limits:

- Within one process, engines sharing an alias queue on a per-alias lock, so a
  second engine waits for the turn that creates the session and then resumes
  it. Across *processes* two first turns on a not-yet-bound alias each create a
  session and the last writer wins; bind once (run one turn) before fanning out.
- `ClaudeCodeEngine.stream()` does not take the per-session lock (as before);
  do not overlap a stream with another turn on the same alias.
- The registry maps names to ids; it does not check that the CLI still has the
  session. A rebind to an id the CLI has forgotten fails at resume time, with
  the CLI's own error.
