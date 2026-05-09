# Plan serialization

Round-trip a `Plan`'s topology through JSON. `Plan.to_dict()` produces
a JSON-friendly description of every step, sentinel, route, and
parallel flag; `Plan.from_dict(data, registry=...)` rebuilds the
plan, rebinding callables and Agents through a name-keyed registry.

This is **descriptor-only** serialisation: targets that aren't string
tool names (callables, Agents, predicates) cross the JSON boundary as
names, and the loader rebinds them. Use it to ship a pipeline
definition between processes, version-control a topology, or
diff-render a plan without instantiating it.

## Signature

```python
plan.to_dict() -> dict
Plan.from_dict(data: dict, *, registry: dict[str, Any] | None = None) -> Plan


# Persisted shape (version 1)
{
    "version": 1,
    "max_iterations": int,
    "steps": [
        {
            "name": str,
            "target": {"kind": "tool" | "agent" | "callable" | "unknown", "name": str},
            "task":     {"kind": "from_prev" | "from_start" | "from_step" | "from_parallel"
                                | "from_parallel_all" | "literal", ...} | None,
            "context":  <task ref> | [<task ref>, ...] | None,    # single OR list, preserves shape
            "parallel": bool,
            "writes":   str,              # omitted when None
            "routes":   [str, ...],       # target step names; predicates rebound via registry
            "routes_by": str,             # omitted when None
            "after_branches": str,        # omitted when None
        },
        ...
    ],
}
```


# Public utility for catching dangling references after load.
from lazybridge.engines.plan._serialisation import validate_plan_refs

validate_plan_refs(steps: list[dict]) -> list[str]


## Synopsis

`to_dict()` walks the live plan and produces a JSON-compatible dict
that captures **topology only**:

- **Step targets** serialise by `kind` + `name`. A string target
  (`Step("research")`) round-trips as
  `{"kind": "tool", "name": "research"}` and gets resolved by the
  outer agent's `tools=[...]` map at run time — no registry entry
  required. A callable serialises as
  `{"kind": "callable", "name": fn.__name__}`. An `Agent` target
  serialises as `{"kind": "agent", "name": agent.name}`.
- **Sentinels** serialise by `kind` + `name`. `from_prev` and
  `from_start` carry no name; `from_step("…")` /
  `from_parallel("…")` / `from_parallel_all("…")` carry the
  referenced step name. A string `task=` serialises as
  `{"kind": "literal", "value": "…"}`.
- **`context=` shape is preserved.** A single sentinel/string
  serialises to one ref dict; a list of sentinels/strings
  serialises to a list of ref dicts. The runtime treats these
  identically; the round-trip preserves the on-disk shape so a
  diff is meaningful.
- **Routes** serialise as a sorted list of target step names. The
  predicates themselves cannot be JSON-encoded — the loader rebinds
  them via the registry under the key
  `f"routes:{step_name}:{target_name}"`.
- **`routes_by` / `after_branches` / `writes` / `parallel`** are
  preserved verbatim (omitted from the dict when their default
  applies, so the JSON stays small).
- **`max_iterations`** and the schema `version` are at the top
  level.

Note that the serialiser does **not** capture `store=` /
`checkpoint_key=` / `resume=` / `on_concurrent=` from the live
`Plan` constructor — those are runtime concerns set when the plan
is instantiated, not topology.

## When to use it

- **Cross-process pipeline transport.** Serialise on a build
  server, deploy the JSON, deserialise on the runtime. Both sides
  must know the same names; the loader binds them to live
  objects.
- **Version control for topology.** Commit `plan.json` alongside
  the code that builds it. Diffs show step additions / removals /
  re-orderings clearly.
- **Render to other formats.** Pass `plan.to_dict()["steps"]`
  through your own renderer to produce Mermaid diagrams, GraphViz
  output, or in-house pipeline visualisations without
  instantiating a Plan.
- **External plan editors.** A web UI that lets users build
  pipelines visually serialises to the same shape — the runtime
  loads the JSON and rebinds tools / callables / predicates.

## When NOT to use it

- **Persisting plan execution state across runs.** That's
  [Checkpoint & resume](../full/checkpoint.md) — `Plan(store=...,
  checkpoint_key=..., resume=True)` writes runtime state to a
  `Store` after every step, separate from the topology.
- **Sharing a runnable Agent.** A serialised plan is just the DAG;
  it doesn't know about provider keys, sessions, or wrappers. The
  runtime side has to construct the live `Agent(engine=Plan, ...)`.
- **Cross-version migration.** The `version: 1` field signals the
  shape; future versions will bump it and `from_dict` will refuse
  older shapes. Migrate explicitly when bumping rather than
  assuming round-trip compatibility.

## Example

```python
import json

from pydantic import BaseModel

from lazybridge import Agent, LLMEngine, Plan, Step, from_step, when


class Hits(BaseModel):
    items: list[str]


def fetch(task: str) -> str:
    """Look up hits for ``task``."""
    return "..."


def rank(task: str) -> str:
    """Rank the supplied hits."""
    return "..."


def has_no_results(env) -> bool:
    return not env.payload.items


# 1) Build the plan in Python.
plan = Plan(
    Step(fetch, name="fetch", writes="hits", output=Hits,
         routes={"apology": when.field("items").empty()}),
    Step(rank,  name="rank",  task=from_step("fetch"), writes="ranked"),
    Step("write", name="write", task=from_step("rank")),
    Step("apology", name="apology"),                  # terminal early-out
)


# 2) Serialise the topology to JSON (lossless for shape).
saved = plan.to_dict()
with open("plan.json", "w") as f:
    json.dump(saved, f, indent=2)


# 3) Load on the other side. Rebind callables, predicates, and
#    Agent targets through the registry.  Tool-name targets ("write",
#    "apology") survive without a registry entry — they're resolved
#    by the outer agent's tools=[...] at run time.
with open("plan.json") as f:
    loaded = json.load(f)


writer = Agent(engine=LLMEngine("claude-opus-4-7"), name="write")
apologiser = Agent(engine=LLMEngine("claude-opus-4-7"), name="apology")


plan_reloaded = Plan.from_dict(
    loaded,
    registry={
        "fetch":  fetch,
        "rank":   rank,
        # Predicate rebinds — key shape: f"routes:{step}:{target}"
        "routes:fetch:apology": when.field("items").empty(),
    },
)


# Tool-name targets attach via the wrapping Agent.
agent = Agent(
    engine=plan_reloaded,
    tools=[writer, apologiser],
)
agent("AI trends April 2026")


# 4) Validate dangling sentinel references after loading
#    (useful when the JSON came from an external source).
from lazybridge.engines.plan._serialisation import validate_plan_refs

errors = validate_plan_refs(loaded["steps"])
assert errors == [], errors
```

## Pitfalls

- **The registry is a positional contract.** Every non-tool
  target (callable, Agent) must be present in the registry, and
  predicates must be present under
  `f"routes:{step_name}:{target_name}"`. Missing entries raise
  `KeyError` with the offending name — by design, the load fails
  loud rather than producing a silently broken plan.
- **Tool-name targets survive without a registry entry.** They're
  resolved at run time from the outer Agent's `tools=[...]`. If
  the loader tries to populate a registry entry for a `"tool"`
  target, the entry is ignored.
- **Step-name security.** `_validate_step_name(name)` rejects any
  name that doesn't match `^[\w][\w\-]*$` (alphanumerics, `_`,
  `-`). This guards against tampered checkpoint payloads with
  path-separator characters or shell metacharacters; it also
  means literal step names with dots, slashes, or spaces fail to
  load.
- **Predicates serialise as target names only.** The actual
  callable lives in Python and must be rebound. If you forget the
  registry entry the load raises `KeyError` with the missing
  `routes:<step>:<target>` key.
- **`parallel=True` is preserved**; `Step.input` / `Step.output`
  type annotations are **not**. The on-disk shape captures
  topology, not type metadata. The runtime re-derives types from
  the `Step.input` / `Step.output` defaults — pass them through
  the registry if you need typed structured output on a
  rebuilt plan.
- **Schema versioning.** `from_dict` accepts only `version: 1`
  today. Breaking changes will bump the version and old shapes
  will be rejected; migrate by re-serialising from the live Plan
  at upgrade time rather than assuming the format is stable
  across versions.
- **Live state isn't captured.** `store=` / `checkpoint_key=` /
  `resume=` / `on_concurrent=` are constructor-time arguments;
  they don't appear in `to_dict()`. Pass them again when you call
  `Plan.from_dict(...)` (or, more typically, when you wrap the
  plan in `Agent(engine=..., store=..., ...)`).

## See also

- [Plan](../full/plan.md) — the engine whose topology is
  serialised.
- [Checkpoint & resume](../full/checkpoint.md) — separate concept:
  runtime *state* persistence (writes-bucket, completed steps,
  status) across crashes, not topology.
- [GraphSchema](../full/graph-schema.md) — the topology view auto-
  populated by `Session`; complements `Plan.to_dict` (they capture
  different facets — Plan = runnable topology, GraphSchema = live
  agent registration with provider / model metadata).
