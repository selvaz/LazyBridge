# LazyContext — Context Injection

`LazyContext` lets you inject dynamic content into an agent's system prompt — without modifying the prompt string itself.

---

## The problem it solves

The old approach: manually build system prompts with f-strings.

```python
# The old way — fragile, hard to compose
system = f"""
You are a writer.
Here is the research: {researcher_output}
Style guide: {style_guide}
Current date: {datetime.now()}
"""
```

Problems:
- You need to run `researcher` before building the string
- Hard to test — the string is baked at construction time
- Hard to compose — can't mix sources cleanly

**LazyContext**: sources are evaluated *at execution time*, not at construction time.

```python
from lazybridge import LazyContext

ctx = (
    LazyContext.from_text("You are a professional writer.")
    + LazyContext.from_agent(researcher)        # reads researcher._last_output when invoked
    + LazyContext.from_function(get_style_guide) # calls get_style_guide() when invoked
)

writer = LazyAgent("openai", context=ctx)
writer.chat("write an article")   # ctx is evaluated here, after researcher has run
```

---

## Four source types

### 1. Static text

```python
ctx = LazyContext.from_text("Always answer in formal English.")
```

Use for: instructions that never change.

### 2. Function (called at execution time)

```python
from datetime import date

def current_date() -> str:
    return f"Today is {date.today().isoformat()}"

ctx = LazyContext.from_function(current_date)
```

Use for: dynamic data (current date, user profile, database lookup).

### 3. From the shared store

```python
from lazybridge import LazyStore

store = LazyStore()

# Only specific keys:
ctx = LazyContext.from_store(store, keys=["research", "style_guide"])

# All keys:
ctx = LazyContext.from_store(store)
```

Output format injected into the system prompt:
```
[shared store]
  research: <value>
  style_guide: <value>
```

Use for: state written by other agents (pattern C pipelines).

### 4. From another agent's last output

```python
researcher = LazyAgent("anthropic", name="researcher")
ctx = LazyContext.from_agent(researcher)
```

When `ctx` is evaluated:
- If `researcher` has run and returned text → injects `[researcher output]\n{output}`
- If `researcher` hasn't run yet (`_last_output is None`) → injects nothing, logs at DEBUG
- If `researcher` ran but returned empty output → injects nothing, logs at DEBUG

No exception is raised in any case. Enable `DEBUG` logging to diagnose silent context gaps.

Use for: passing one agent's result to another in a sequential pipeline.

---

## Composing contexts

Use `+` to combine any number of sources:

```python
ctx = (
    LazyContext.from_text("You are a senior analyst.")
    + LazyContext.from_agent(data_collector)
    + LazyContext.from_store(sess.store, keys=["market_data"])
    + LazyContext.from_function(get_current_date)
)
```

Or `LazyContext.merge()` for the same result:

```python
ctx = LazyContext.merge(
    LazyContext.from_text("You are a senior analyst."),
    LazyContext.from_agent(data_collector),
    LazyContext.from_store(sess.store, keys=["market_data"]),
)
```

Sources are evaluated in order and joined with blank lines (`\n\n`).

---

## Testing your context

Because `ctx()` is just a string, you can test it independently:

```python
# Before running any agents:
print(ctx())   # only static parts appear

# After running the researcher:
researcher.loop("find data")
print(ctx())   # now includes researcher's output
```

This is much easier to debug than embedded f-strings.

---

## Agent-level vs call-level context

```python
# Context applied to every call on this agent:
writer = LazyAgent("openai", context=LazyContext.from_text("Write in Italian."))
writer.chat("What is Python?")    # "Python è un linguaggio..."

# Override for a single call:
writer.chat("What is Python?", context=LazyContext.from_text("Write in German."))
# call-level context replaces agent-level context for this call only
```

---

## Full example

```python
from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore()

# Step 1: collect data
collector = LazyAgent("anthropic", name="collector")
collector.loop("Collect the top 5 AI papers this month")
store.write("papers", collector._last_output)

# Step 2: analyse (no reference to collector)
analyst = LazyAgent("anthropic", name="analyst")
ctx_a = LazyContext.from_store(store, keys=["papers"])
analyst.chat("Identify the 3 most impactful findings", context=ctx_a)
store.write("findings", analyst._last_output)

# Step 3: write (no reference to collector or analyst)
ctx_w = (
    LazyContext.from_text("You write for a non-technical audience.")
    + LazyContext.from_store(store, keys=["findings"])
)
writer = LazyAgent("openai", context=ctx_w)
writer.chat("Write a blog post from these findings")
```
