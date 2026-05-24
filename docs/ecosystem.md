# The LazyBridge ecosystem — three packages

LazyBridge is one project split into three installable packages. Each has a
single, clear job, and they stack:

```
lazybridge   stable core runtime — Agent, Engine, Tool, Plan, Envelope, State
   ▲              the mental model never changes: Agent = Engine + Tools + State
   │
lazytools    capabilities — reusable tool providers, connector clients, safety
   ▲              Gmail · Telegram · MCP · tool gateways · document readers · skills
   │
lazypulse    always-on orchestration — tick loop, trust policy, inbound adapters
                  turns a one-shot agent into one that wakes itself up
```

**Dependency direction is one-way** (and enforced by tests, not convention):

| Allowed | Forbidden |
|---|---|
| `lazytools → lazybridge` | `lazybridge → lazytools` |
| `lazypulse → lazybridge` | `lazytools → lazypulse` |
| `lazypulse → lazytools` *(optional, behind extras)* | any circular import |

So the core never depends on the layers above it: `import lazybridge` pulls
nothing from `lazytools` or `lazypulse`.

## Which package do I need?

| You want to… | Install | Import from |
|---|---|---|
| Compose LLMs, functions, plans, humans as one uniform "tool" model | `pip install lazybridge` | `lazybridge` |
| Add concrete capabilities — Gmail/Telegram/MCP/gateway, doc readers, skills | `pip install lazytoolkit[...]` | `lazytools.*` |
| Run an always-on agent that watches an inbox and acts under a trust policy | `pip install lazypulse[...]` | `lazypulse` |

```bash
pip install lazybridge                  # the core runtime, dependency-light
pip install 'lazytoolkit[mcp]'          # + the MCP connector
pip install 'lazytoolkit[gmail]'        # + Gmail client & guarded tools
pip install 'lazypulse[gmail]'          # always-on Gmail agent (pulls lazytoolkit[gmail])
```

## Stability & cadence

| Package | Role | Stability | Cadence |
|---|---|---|---|
| **lazybridge** | core runtime + mental model | **stabilising** — the public API (`lazybridge.__all__`) is guarded by a snapshot test; breaking changes are rare and deliberate | slow, deliberate |
| **lazytools** (`lazytoolkit`) | capabilities | **active development** — new connectors and tools land here continually | fast, additive |
| **lazypulse** | always-on orchestration | **active development** | fast |

The split exists so the core can sit still while capabilities grow: you pin a
stable `lazybridge` and let the capability layer iterate underneath your app.

## Repos

- **lazybridge** — <https://github.com/selvaz/LazyBridge>
- **lazytoolkit** — <https://github.com/selvaz/LazyTools>
- **lazypulse** — <https://github.com/selvaz/LazyPulse>
