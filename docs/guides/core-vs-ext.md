# Core vs Ext — the import boundary policy

LazyBridge is split into three concentric layers:

| Layer | Lives in | Holds |
|---|---|---|
| **Core** | `lazybridge/` (excluding the two subtrees below) | `Agent`, `LLMEngine`, `Plan`, `Step`, `Tool`, `Envelope`, `Memory`, `Store`, `Session`, sentinels, predicates, guards, providers |
| **Framework extensions** | `lazybridge/ext/*` | MCP, OpenTelemetry, HumanEngine / SupervisorEngine, Evals, the HTTP gateway, planners, visualizer |
| **Domain tool kits** | `lazybridge/external_tools/*` | `read_docs`, `doc_skills`, and any future heavy-dep examples |

This split is enforced — not just convention — by three architectural
tests that fail the build the moment a forbidden import lands:

- `tests/unit/test_core_ext_boundary.py` — rule 1
- `tests/unit/test_ext_core_boundary.py` — rule 2
- `tests/unit/test_ext_factories.py` — rule 3

## The three rules

### Rule 1 — Core never imports from `ext.*` or `external_tools.*`

Core stays self-contained.  No lazy imports, no `TYPE_CHECKING`-only
imports, no function-local imports — the boundary is the import
statement itself, regardless of when it runs.

Why: if core depends on an extension, then installing the framework
without that extension installed breaks core.  More subtly, it makes
core's stability hostage to extension velocity, which is the opposite
of what the ext tier is for.

The runtime side of this rule is the reason `Agent` has **no factory
methods for ext engines** (e.g. no `Agent.from_supervisor(...)` /
`Agent.from_human(...)`).  Construct ext engines directly:

```python
from lazybridge import Agent
from lazybridge.ext.hil import SupervisorEngine

Agent(engine=SupervisorEngine(tools=[...], agents=[...]), name="sup")
```

Or use the module-level ergonomic factories shipped in each ext
package:

```python
from lazybridge.ext.hil import supervisor_agent
sup = supervisor_agent(tools=[...], agents=[...])
```

### Rule 2 — Extensions never reach into `core.*` internals

Ext code imports from the **top-level** `lazybridge` namespace (the
public surface) — never from `lazybridge.core.*` directly.

Why: the `core/` subtree carries types and helpers that exist to
implement the public surface.  An ext package that reaches past
`lazybridge` into `lazybridge.core.providers.base` (for example) is
coupling itself to the implementation layout, which we reshape
between minor releases.

The reverse direction stays open: core may freely import from
`lazybridge.core.*` (it's the same layer), and ext packages may
import from `lazybridge.*` (the public surface).

### Rule 3 — Ext-engine factories live in ext packages, not on `Agent`

Concretely: when an ext package ships a "build an agent that runs
this engine" helper, the helper is a module-level function in the
ext package — not a classmethod on `Agent`.  The contract is the
kwarg split: ext-engine kwargs go to the engine constructor;
remaining `**agent_kw` flow to `Agent(...)`.

```python
# lazybridge.ext.hil
def supervisor_agent(*, tools=None, agents=None, store=None, **agent_kw):
    return Agent(engine=SupervisorEngine(tools=tools, agents=agents, store=store), **agent_kw)
```

This keeps `Agent` itself free of ext-aware code (rule 1) while still
giving each ext package a one-liner.

## When to ship in ext vs core

Ship in **core** when:

- It's part of the universal `Agent = Engine + Tools + State`
  composition.
- It has no optional dependency outside `pydantic`.
- It's expected to be used by ≥80% of LazyBridge applications.

Ship in **ext** when:

- It introduces an optional dependency (`mcp`, `opentelemetry`,
  `cryptography`, a web framework, …).
- It crosses a process boundary (subprocess, HTTP server, …).
- It's a cross-cutting concern that augments the agent runtime
  rather than being part of the universal composition.

Ship in **external_tools** when:

- It's a domain example, not a framework primitive (PDF readers,
  BM25 skill builders, report renderers, …).
- It might be lifted out into a sibling package later (the
  reporting subsystem was — see `selvaz/LazyReport`).

## Stability

All three layers are `alpha` pre-1.0 (`lazybridge.__stability__ ==
"alpha"`).  Promotion to `beta` and `stable` happens per-symbol after
1.0; the import boundary doesn't change.

## See also

- [Concepts → Mental model](../concepts/mental-model.md) — the
  universal `Engine + Tools + State` shape that core encodes.
- [Decisions → Do I need Advanced?](../decisions/need-advanced.md) —
  when to ship in ext vs subclass in your own code.
- `lazybridge/ext/__init__.py` — the in-source policy summary that
  this page expands.
