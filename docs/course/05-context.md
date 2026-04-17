# Module 5: Context Injection

`LazyContext` lets you inject dynamic information into an agent's system prompt — lazily evaluated, composable, and testable.

## Why context?

Without context, you'd hardcode information into system prompts:

```python
# Fragile — hardcoded at construction time
ai = LazyAgent("anthropic", system=f"The user's name is {get_user_name()}. Today is {get_date()}.")
```

With `LazyContext`, information is fetched *at call time*:

```python
from lazybridge import LazyAgent, LazyContext

ctx = LazyContext.from_function(lambda: f"Today is {get_date()}")
ai = LazyAgent("anthropic", context=ctx)
# The date is resolved fresh on every chat() call
```

## Context sources

### From text

```python
ctx = LazyContext.from_text("You are an expert on Python programming.")
```

### From a function (evaluated at call time)

```python
def get_user_profile():
    return "User: Marco, Role: Developer, Timezone: CET"

ctx = LazyContext.from_function(get_user_profile)
```

### From another agent's output

```python
researcher = LazyAgent("anthropic", name="researcher")
researcher.chat("Find the latest AI news")

# Writer gets researcher's output as context
ctx = LazyContext.from_agent(researcher)
writer = LazyAgent("openai", context=ctx)
writer.chat("Write a blog post based on the research")
```

### From a store

```python
from lazybridge import LazyStore

store = LazyStore()
store.write("findings", "Key finding: LLMs can reason about code")

ctx = LazyContext.from_store(store, keys=["findings"])
ai = LazyAgent("anthropic", context=ctx)
ai.chat("Summarize the findings")
```

## Composing contexts

Combine multiple contexts with `+` or `merge()`:

```python
profile = LazyContext.from_function(get_user_profile)
rules = LazyContext.from_text("Always respond in formal English.")
research = LazyContext.from_agent(researcher)

# Combine with +
combined = profile + rules + research

# Or merge()
combined = LazyContext.merge(profile, rules, research)

ai = LazyAgent("anthropic", context=combined)
```

All sources are concatenated (with separators) when the agent runs.

## Testing context

Since `LazyContext` is callable, you can inspect what it produces without running an agent:

```python
ctx = LazyContext.from_text("Hello") + LazyContext.from_function(lambda: "World")
print(ctx())  # "Hello\n\nWorld"
# or
print(ctx.build())  # same thing
```

## Per-call context override

Override the agent's default context for a single call:

```python
ai = LazyAgent("anthropic", context=LazyContext.from_text("Default context"))

# This call uses different context
special_ctx = LazyContext.from_text("Special instructions for this call only")
ai.chat("Do something", context=special_ctx)
```

---

## Exercise

1. Create a context that includes the current time and a user profile
2. Create two agents: a researcher and a writer. Have the writer use `LazyContext.from_agent(researcher)` to build on the research
3. Test your context by calling `ctx()` directly to see what the agent will receive

**Next:** [Module 6: Multi-Agent Pipelines](06-pipelines.md) — chain and parallelize agents.
