## signature
Plan.to_dict() -> dict
Plan.from_dict(data: dict, *, registry: dict[str, Any] | None = None) -> Plan

# Persisted shape (v=1):
# {
#   "version": 1,
#   "max_iterations": int,
#   "steps": [
#     {
#       "name": str,
#       "target": {"kind": "tool"|"agent"|"callable", "name": str},
#       "task": {"kind": "from_prev"|"from_start"|"from_step"|"from_parallel"|"literal", ...},
#       "context": {... or null},
#       "parallel": bool,
#       "writes": str | null,
#     },
#     ...
#   ]
# }

## rules
- Only topology is serialised: step names, sentinels, ``writes``,
  ``parallel`` flag, ``max_iterations``. Callables / Agents are
  serialised by **name only**.
- Rebind by passing ``registry={name: callable_or_agent}`` to
  ``from_dict``. Unknown names raise ``KeyError`` with the offending
  entry — the load fails loud rather than producing a silently-broken
  Plan.
- ``target.kind == "tool"`` (a string target) survives round-trip
  without a registry entry — the tool is resolved at run time from the
  Agent's tool map.

## narrative
Plan serialisation is what lets LazyBridge's crown jewel (compile-time
DAG validation) cross process boundaries. You can:

* Save a plan to disk, ship it with the code, reload on startup.
* Hand a plan shape to a worker service over HTTP, have the worker
  reconstruct the plan with its own agent bindings and run it.
* Version-control plan definitions separately from the agents they
  drive.

What you CAN'T do: ship live functions / agents across processes.
The registry pattern is explicit about this — both sides need to know
the same names, and the loader rebinds them on arrival. This is a
feature, not a limitation: it keeps execution targets in the host
process where they belong.

For a human-readable topology that includes runtime metadata (agent
providers, models), use `GraphSchema.to_yaml` instead — `Plan.to_dict`
is deliberately implementation-focused.

## example
```python
from lazybridge import Plan, Step, Agent, from_step
import json

def fetch(task: str) -> str: ...
def rank(task: str) -> str: ...

plan = Plan(
    Step(fetch, name="fetch", writes="hits"),
    Step(rank,  name="rank",  task=from_step("fetch"), writes="ranked"),
)

# Persist.
with open("plan.json", "w") as f:
    json.dump(plan.to_dict(), f, indent=2)

# Later / elsewhere:
with open("plan.json") as f:
    saved = json.load(f)

plan_reloaded = Plan.from_dict(saved, registry={
    "fetch": fetch,   # rebind to live functions
    "rank":  rank,
})

Agent.from_engine(plan_reloaded)("AI trends")
```

## pitfalls
- The registry is a positional contract: every non-tool target must be
  in the registry or ``from_dict`` raises ``KeyError``. Keep target
  names stable across versions.
- Tool-name targets (``target=str``) survive without a registry — they
  are resolved by the outer Agent's ``tools=[...]`` at run time.
- The JSON shape is versioned (``version: 1``). Breaking changes will
  bump the number and ``from_dict`` will refuse older shapes; migrate
  explicitly rather than silently.

## see-also
[plan](plan.md), [graph_schema](graph-schema.md),
[engine_protocol](engine-protocol.md)
