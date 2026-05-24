# Core vs Ext — the import boundary policy

LazyBridge is split into three concentric layers:

| Layer | Lives in | Holds |
|---|---|---|
| **Core** | `lazybridge/` (excluding the subtree below) | `Agent`, `LLMEngine`, `Plan`, `Step`, `Tool`, `Envelope`, `Memory`, `Store`, `Session`, sentinels, predicates, guards, providers |
| **Framework extensions** | `lazybridge/ext/*` | OpenTelemetry, HumanEngine / SupervisorEngine, Evals, planners, visualizer |
| **Concrete tools** | sibling `lazytoolkit` package (`pip install lazytoolkit`) | connectors (Gmail, Telegram, MCP, the HTTP gateway), document readers (`read_docs`), skills (`doc_skills`) |

> **Moved in 0.8.** The MCP and gateway connectors (`lazybridge.ext.{mcp,gateway}`)
> and the domain tool kits (`lazybridge.external_tools.*`) moved to the standalone
> [`lazytoolkit`](https://github.com/selvaz/LazyTools) package. Old import paths
> still work via lazy deprecation shims until 0.9 — import from `lazytools.*`
> instead (e.g. `from lazytools.connectors.mcp import MCP`).

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

Ship in the sibling **`lazytoolkit`** package (`import lazytools`) when:

- It's a concrete tool or connector, not a framework primitive
  (PDF readers, BM25 skill builders, MCP / gateway connectors, …).
- It brings outbound I/O or a heavy optional dependency that the core
  framework should never pull in.

> The old `lazybridge.external_tools.*` namespace held these before 0.8;
> it now contains only lazy deprecation shims and is removed in 0.9. The
> reporting subsystem moved out the same way in 0.7.9 — see
> `selvaz/LazyReport`.

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
