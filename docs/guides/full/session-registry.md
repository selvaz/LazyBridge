# Session aliases and SessionRegistry

SessionRegistry maps a name to an engine's native conversation ID. Both engines
take session_alias and session_registry. An alias implies durable persistence;
an explicit thread_id/session_id wins over lookup and rebinds the alias.

## Name and resume a conversation

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lazybridge import Agent, CodexEngine, SessionRegistry

repo = str(Path.cwd().resolve())
# The example cleans up its registry. Use a fixed path across real processes.
with TemporaryDirectory() as directory:
    registry = SessionRegistry(path=Path(directory) / "sessions.json")
    first = CodexEngine(cwd=repo, session_alias="reviewer", session_registry=registry)
    Agent(name="reviewer", engine=first)("Remember the word PINEAPPLE.")
    second = CodexEngine(cwd=repo, session_alias="reviewer", session_registry=registry)
    assert second.thread_id == first.thread_id
    print(Agent(name="reviewer", engine=second)("What was the word?").text())
```

ClaudeCodeEngine uses the same alias/registry parameters with native session_id
and persist_session. Claude session_name is only the runtime-mode slot label on
a LazyBridge Session, not the alias or a session title.

## Registry operations

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lazybridge import SessionRegistry

repo = str(Path.cwd().resolve())
with TemporaryDirectory() as directory:
    registry = SessionRegistry(path=Path(directory) / "sessions.json")
    assert registry.resolve("codex", repo, "reviewer") is None
    registry.bind("codex", repo, "reviewer", "example-native-id", model=None, effort="low")
    registry.rename("codex", repo, "reviewer", "reviewer-v2")
    assert registry.resolve("codex", repo, "reviewer-v2") == "example-native-id"
    assert len(registry.entries(kind="codex", scope=repo)) == 1
    assert registry.forget("codex", repo, "reviewer-v2")
```

Path precedence is explicit path, LAZYBRIDGE_SESSIONS_FILE, then
~/.lazybridge/sessions.json. default_session_registry() gives the process-wide
default; set_default_session_registry(registry) replaces it and returns the old
one. A private project can select its own file.

| Field | Rule |
| --- | --- |
| Key | (kind, scope, name); kind is codex or claude |
| Scope | Resolved absolute posix cwd (or working directory); case-normalized on Windows |
| Name | Starts with an ASCII letter, then letters/digits/underscore/dot/hyphen; at most 64 characters |
| Value | native_id, model, effort, created_at, updated_at |

The same alias in two repositories, or for two engines, has separate bindings.
Model/effort are recorded metadata, not immutable pins or validation constraints.
Rename raises KeyError for a missing source and ValueError for an existing target.

## Persistence and failures

Known alias adopts its ID; unknown alias creates a fresh durable conversation.
The engine binds as soon as it learns an ID, including interrupted turns.
Claude may return a new ID on resume and the alias follows it. Registry write
failure warns/logs and does not fail the engine turn.

Writes use atomic replace/fsync, a thread lock and a cross-process lock file
(with stale-lock takeover). A missing registry is empty. Invalid JSON is
quarantined as .corrupt, with a timestamp if necessary, rather than overwritten.

Registry locking protects the JSON file, not the native transcript. Durable
runs and streams serialize in-process, including first-turn alias creation and
stream teardown. Separate event loops/processes are not a shared scheduler.
Across processes, simultaneous first use can create two conversations and the
last binding wins; bind once before sharing and coordinate every resumed turn.
A missing/deleted native conversation still fails at runtime resume: the
registry does not verify runtime retention.

## Keep three kinds of session distinct

- LazyBridge Session holds observability events and in-process approval/runtime slots.
- Claude session / Codex thread holds native model history and tool reads.
- SessionRegistry persists human names pointing to those native IDs.

Python Memory/Store/Session references can be shared when swapping engines.
Native history remains engine-specific. LazyTools code-support tools expose
their own session_name interface on top of native engine aliases; see
[code sessions](https://tools.lazybridge.com/code-support/) and the
[code bridge](https://tools.lazybridge.com/code-bridge/).
