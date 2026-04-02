### LazyRouter

Conditional branching node. Routes execution to one of several `LazyAgent` instances based on a condition function.

```python
@dataclass
class LazyRouter:
    condition: Callable[[Any], str]    # receives the input value, returns a route key
    routes: dict[str, LazyAgent]       # key → LazyAgent
    name: str = "router"
    default: str | None = None         # fallback key if condition returns unknown key
```

### Methods

```python
router.route(value: Any) -> LazyAgent
```
Evaluates `condition(value)` synchronously. Returns the matching agent.
- Raises `KeyError` if key not in routes AND `default` is None.
- Raises `TypeError` if condition returns a non-string.

```python
await router.aroute(value: Any) -> LazyAgent
```
Async version. Awaits `condition(value)` if condition is a coroutine function.

```python
router.agent_names -> list[str]      # property: names of all routable agents
router.to_graph_node() -> dict       # serializable representation for GraphSchema
```

### When to use vs plain if/else

Use `LazyRouter` when:
- You want the conditional branch to appear in the `GraphSchema` (visible to GUI)
- You have 3+ routes
- The routing condition is reusable across pipelines

Use plain Python `if/else` when:
- Simple binary logic not needed in the graph
- Throwaway pipeline

### Examples

**Basic binary routing:**
```python
from lazybridge import LazyAgent, LazyRouter

writer   = LazyAgent("anthropic", name="writer")
reviewer = LazyAgent("openai",    name="reviewer")

router = LazyRouter(
    condition=lambda response: "writer" if "approved" in response.lower() else "reviewer",
    routes={"writer": writer, "reviewer": reviewer},
    name="quality_gate",
)

checker = LazyAgent("anthropic", name="checker")
result = checker.chat("evaluate this draft: ...")
next_agent = router.route(result.content)
final = next_agent.chat("proceed with: " + result.content)
```

**Multi-route with default:**
```python
router = LazyRouter(
    condition=lambda r: r.strip().lower().split()[0],  # first word of response
    routes={
        "research":  researcher,
        "analyse":   analyst,
        "write":     writer,
        "review":    reviewer,
    },
    name="task_router",
    default="writer",   # fallback if condition returns unknown key
)
```

**Async condition:**
```python
async def route_by_classification(text: str) -> str:
    classifier = LazyAgent("anthropic")
    label = await classifier.atext(f"Classify into: research/write/review. Text: {text}")
    return label.strip().lower()

router = LazyRouter(
    condition=route_by_classification,
    routes={"research": researcher, "write": writer, "review": reviewer},
    name="ai_router",
)

import asyncio
next_agent = asyncio.run(router.aroute("what is the capital of France?"))
```

**Full pipeline with router:**
```python
from lazybridge import LazyAgent, LazyRouter, LazySession

sess = LazySession()
planner  = LazyAgent("anthropic", name="planner",  session=sess)
executor = LazyAgent("openai",    name="executor", session=sess)
checker  = LazyAgent("anthropic", name="checker",  session=sess)

router = LazyRouter(
    condition=lambda r: "executor" if "actionable" in r.lower() else "checker",
    routes={"executor": executor, "checker": checker},
    name="plan_router",
    default="checker",
)

plan = planner.chat("create a plan to write a Python web scraper")
agent = router.route(plan.content)
agent.chat("proceed: " + plan.content)
```
