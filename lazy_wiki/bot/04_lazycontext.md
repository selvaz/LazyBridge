# LazyContext — Complete Reference

## 1. Overview

`LazyContext` is a lazily-evaluated, composable string injected into the system prompt. Sources are evaluated at `build()` time — which is called inside `LazyAgent.chat()` / `loop()` just before the LLM request is assembled. Construction is always free of side effects.

---

## 2. Key principle

`LazyContext` is NOT a communication channel between agents. It injects strings into system prompts. Agent-to-agent data transfer happens via return values; `LazyContext` only makes those results *visible* in another agent's context at execution time.

```python
# Wrong mental model: "LazyContext sends data from agent A to agent B"
# Correct mental model: "LazyContext reads agent A's _last_output and injects it
#                        into agent B's system prompt when agent B is called"
```

---

## 3. Factory methods

### `from_text(text)` — static string

```python
from lazybridge import LazyContext, LazyAgent

ctx = LazyContext.from_text("You are a senior data analyst. Always cite sources.")
agent = LazyAgent("anthropic", context=ctx)
```

The string is captured at construction time and returned verbatim on every `build()` call.

---

### `from_function(fn)` — called at execution time

```python
from lazybridge import LazyContext

def get_current_user() -> str:
    return f"Current user: {database.get_user()}"

ctx = LazyContext.from_function(get_current_user)
# get_current_user() is called each time ctx.build() is invoked,
# i.e. on each agent call — always reflects the current state
```

`fn` must be `() -> str`. If `fn()` raises, the exception is silently swallowed and the source contributes an empty string to the result.

---

### `from_store(store, *, keys, prefix)` — reads `LazyStore`

```python
from lazybridge import LazyContext

ctx = LazyContext.from_store(sess.store)                     # all keys
ctx = LazyContext.from_store(sess.store, keys=["findings"])  # specific keys only
```

Output format produced by `build()`:

```
[shared store]
  findings: <value>
```

This uses `store.to_text(keys=keys)` internally. If the store is empty (or all requested keys are absent), an empty string is returned.

---

### `from_agent(agent, *, prefix)` — reads `agent._last_output`

```python
from lazybridge import LazyAgent, LazyContext

researcher = LazyAgent("anthropic", name="researcher")
researcher.loop("find the latest AI news")    # sets researcher._last_output

ctx = LazyContext.from_agent(researcher)
# When ctx.build() is called, produces:
# "[researcher output]\n<researcher._last_output>"
```

Two cases produce an empty result (both are safe and return `""`):

| `_last_output` | Meaning | Debug log emitted |
|---|---|---|
| `None` | Agent has not been run yet | "has not been run yet" |
| `""` | Agent ran but returned empty output | "was run but returned an empty output" |

Debug messages are emitted via the standard `logging` module at `DEBUG` level so you can diagnose context gaps without noise in production. It is safe to create `LazyContext.from_agent(agent)` before the agent runs.

`prefix` overrides the label line. Default: `"[{agent.name} output]"` (or `"[agent output]"` if the agent has no name).

```python
ctx = LazyContext.from_agent(researcher, prefix="[Research findings]")
# Produces:
# "[Research findings]\n<researcher._last_output>"
```

---

## 4. Composition

Sources are evaluated in order and joined with `"\n\n"`.

### `+` operator

```python
ctx = LazyContext.from_text("Be concise.") + LazyContext.from_agent(researcher)
```

### `merge()` — classmethod, any number of sources

```python
ctx = LazyContext.merge(
    LazyContext.from_text("Role: analyst"),
    LazyContext.from_store(sess.store, keys=["raw_data"]),
    LazyContext.from_agent(researcher),
)
```

Both `+` and `merge()` return a new `LazyContext`; the originals are unchanged.

---

## 5. `build()` / `__call__()`

```python
ctx.build()    # → str
ctx()          # identical — __call__ delegates to build()
```

Each source is called in registration order. Sources that raise exceptions or return empty/whitespace-only strings are silently skipped. Non-empty results are stripped and joined with `"\n\n"`.

```python
from lazybridge import LazyContext

ctx = LazyContext.from_text("Hello") + LazyContext.from_text("World")
print(ctx())        # "Hello\n\nWorld"
print(ctx.build())  # same

# Test a context without running agents:
print(ctx())        # inspect exactly what the agent will receive
```

`bool(ctx)` is `True` if at least one source is registered (even if they all evaluate to empty strings at runtime).

---

## 6. Call-level context override

```python
agent = LazyAgent("anthropic", context=LazyContext.from_text("default context"))

# For one call only, replace the agent-level context entirely:
agent.chat("question", context=LazyContext.from_text("call-specific context"))
# "default context" is NOT included in this call's system prompt
```

The call-level `context` parameter **replaces** (does not append to) the agent-level `context` for that call. This applies to both `chat()` and is forwarded through `loop()` via `**chat_kwargs`.

---

## 7. Full pipeline example

```python
import asyncio
from lazybridge import LazyAgent, LazySession, LazyContext

sess = LazySession()
researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)

# Context is created BEFORE researcher runs — safe because from_agent() is lazy
writer_ctx = LazyContext.merge(
    LazyContext.from_text("You are a technical writer."),
    LazyContext.from_agent(researcher),
)

writer_with_ctx = LazyAgent("openai", name="writer2", context=writer_ctx, session=sess)

# Sequential: researcher must run first so _last_output is set
researcher.loop("summarize recent advances in transformer architectures")
result = writer_with_ctx.chat("Write a blog post based on the research above.")
print(result.content)
```
