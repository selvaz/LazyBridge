# LazyRouter — Conditional Branching

`LazyRouter` routes the result of one agent to one of several agents based on a condition.

---

## When to use a router

Most of the time, a plain Python `if/else` is fine:

```python
result = checker.chat("evaluate this draft")
if "approved" in result.content.lower():
    writer.chat("publish this: " + result.content)
else:
    reviewer.chat("revise this: " + result.content)
```

Use `LazyRouter` when:
- You want the branching to appear in the graph schema (visible to a GUI)
- You have 3+ routes and want them named cleanly
- You want the routing condition reusable across pipelines

---

## Basic usage

```python
from lazybridge import LazyAgent, LazyRouter

writer   = LazyAgent("anthropic", name="writer")
reviewer = LazyAgent("openai",    name="reviewer")

router = LazyRouter(
    condition=lambda response: "writer" if "APPROVED" in response.upper() else "reviewer",
    routes={"writer": writer, "reviewer": reviewer},
    name="approval_gate",
)

checker = LazyAgent("anthropic", name="checker")
result  = checker.chat("Evaluate this draft: [...]")

next_agent = router.route(result.content)
final = next_agent.chat("Proceed with: " + result.content)
print(final.content)
```

---

## Multiple routes

```python
researcher = LazyAgent("anthropic", name="researcher")
analyst    = LazyAgent("openai",    name="analyst")
writer     = LazyAgent("anthropic", name="writer")
reviewer   = LazyAgent("openai",    name="reviewer")

def pick_route(response: str) -> str:
    response_lower = response.strip().lower()
    if "research" in response_lower:
        return "research"
    elif "analyse" in response_lower or "analyze" in response_lower:
        return "analyse"
    elif "write" in response_lower:
        return "write"
    else:
        return "review"   # default

router = LazyRouter(
    condition=pick_route,
    routes={
        "research": researcher,
        "analyse":  analyst,
        "write":    writer,
        "review":   reviewer,
    },
    name="task_router",
    default="review",   # fallback if condition returns unknown key
)
```

---

## Fallback default

If the condition might return an unexpected value, set `default=`:

```python
router = LazyRouter(
    condition=lambda r: r.strip().lower(),
    routes={"approve": publisher, "reject": drafter},
    name="publish_gate",
    default="reject",   # any unknown response → drafter
)
```

Without `default`, an unknown key raises `KeyError`.

---

## Async condition

Your condition can be an async function. Create the classifier agent once at setup time — not inside the condition, which runs on every route call:

```python
import asyncio
from lazybridge import LazyAgent, LazyRouter

# Create once — reused for every routing decision
classifier = LazyAgent("anthropic")

async def classify_with_llm(text: str) -> str:
    label = await classifier.atext(
        f"Classify this task as one of: research / analyse / write. Task: {text}. Return only the label."
    )
    return label.strip().lower()

router = LazyRouter(
    condition=classify_with_llm,
    routes={"research": researcher, "analyse": analyst, "write": writer},
    default="write",
)

next_agent = asyncio.run(router.aroute("What are the latest GPU benchmark numbers?"))
```

---

## Full pipeline example

```python
from lazybridge import LazyAgent, LazyRouter, LazySession

sess = LazySession()

drafter   = LazyAgent("anthropic", name="drafter",   session=sess)
reviewer  = LazyAgent("openai",    name="reviewer",  session=sess)
publisher = LazyAgent("anthropic", name="publisher", session=sess)

router = LazyRouter(
    condition=lambda r: "publisher" if "APPROVED" in r.upper() else "reviewer",
    routes={"publisher": publisher, "reviewer": reviewer},
    name="quality_gate",
    default="reviewer",
)

# Pipeline loop
content = "Write a blog post about AI safety."
for _ in range(3):  # up to 3 revision cycles
    draft = drafter.chat(content)
    check = reviewer.chat(f"Review this and say APPROVED or REJECTED with reason: {draft.content}")
    next_agent = router.route(check.content)
    if next_agent is publisher:
        result = publisher.chat(f"Publish: {draft.content}")
        print("Published:", result.content[:200])
        break
    else:
        content = f"Revise based on this feedback: {check.content}\n\nOriginal draft: {draft.content}"
```
