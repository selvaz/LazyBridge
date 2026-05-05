# Plan serialization

Callables and Agents are serialised by **name only** — they don't
cross process boundaries. Both sides must know the same names; the
loader rebinds them via `registry=`. For a human-readable topology with
runtime metadata (providers, models) use `GraphSchema.to_yaml` instead.

## Example

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

## Pitfalls

- The registry is a positional contract: every non-tool target must be
  in the registry or ``from_dict`` raises ``KeyError``. Keep target
  names stable across versions.
- Tool-name targets (``target=str``) survive without a registry — they
  are resolved by the outer Agent's ``tools=[...]`` at run time.
- The JSON shape is versioned (``version: 1``). Breaking changes will
  bump the number and ``from_dict`` will refuse older shapes; migrate
  explicitly rather than silently.

!!! note "API reference"

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
    #       "task":     {"kind": "from_prev"|"from_start"|"from_step"|"from_parallel"|"literal", ...},
    #       "context":  {... single ref}  OR  [{...}, {...}]  OR  null,    # single OR list
    #       "parallel": bool,
    #       "writes": str | null,
    #     },
    #     ...
    #   ]
    # }

!!! warning "Rules & invariants"

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

## See also

- [Plan](plan.md) — what gets serialised.
