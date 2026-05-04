# Core vs Extension policy

LazyBridge ships as a single PyPI package. **Pre-1.0 (0.7.x), every
surface is `alpha`** — the entire package, top-level core and every
extension. Interfaces may change between any two releases. Pin exact
versions in production.

The package is split into four namespaces, organised by *what the code
is*, not by stability:

| Namespace                          | Role                                                     |
|------------------------------------|----------------------------------------------------------|
| `lazybridge.*` (top-level)         | Core runtime — `Agent`, `Plan`, `Tool`, engines, providers, memory, store, session, exporters, guardrails |
| `lazybridge.ext.*`                 | Framework extensions — `mcp`, `otel`, `hil`, `evals`, `gateway`, `planners`, `viz` |
| `lazybridge.external_tools.*`      | Domain tool packages — `read_docs`, `doc_skills`, `data_downloader`, `stat_runtime`, `report_builder` |

The single stability marker lives on the top-level package:

```python
import lazybridge
assert lazybridge.__stability__ == "alpha"
```

Individual modules do not carry their own stability tag in 0.7.

## What goes where

### Core (`lazybridge.*`)
The minimum a generalist user needs to build agents.

- `Agent`, `Envelope`, `Tool`, `Memory`, `Store`, `Session`, `EventLog`
- `LLMEngine` (human-in-the-loop engines — `HumanEngine` and
  `SupervisorEngine` — live in `lazybridge.ext.hil`)
- `Plan`, `Step`, `from_prev` / `from_start` / `from_step` / `from_parallel` / `from_parallel_all`
- `BaseProvider` and the supported providers (Anthropic, OpenAI,
  Google, DeepSeek, LM Studio, LiteLLM)
- `Guard`, `GuardChain`, `ContentGuard`, `LLMGuard`
- Basic exporters (`ConsoleExporter`, `JsonFileExporter`, `CallbackExporter`,
  `FilteredExporter`, `StructuredLogExporter`)

### Framework extensions (`lazybridge.ext.*`)
Augment the agent runtime itself.

- `mcp` — Model Context Protocol client (stdio + HTTP)
- `otel` — OpenTelemetry span exporter
- `hil` — `HumanEngine`, `SupervisorEngine`
- `evals` — `EvalSuite`, `EvalCase`, `llm_judge`, assertion helpers
- `gateway` — adapter for server-side tool gateways
- `planners` — DAG builder + blackboard planner factories
- `viz` — live + replay pipeline visualizer

### Domain tool packages (`lazybridge.external_tools.*`)
Tool kits an agent can call. Each module exposes a factory that
returns `list[Tool]`.

- `read_docs` — multi-format document reader
- `doc_skills` — BM25 local doc skill runtime
- `data_downloader` — Yahoo / FRED / ECB ingestion
- `stat_runtime` — econometrics & time-series sandbox
- `report_builder` — HTML/PDF report assembler

## Architectural rules

1. **Core never imports from `ext/` or `external_tools/`.** The reverse
   direction is fine.
2. **`ext/` and `external_tools/` only import from public
   `lazybridge.*`** — never from `lazybridge.core.*` or other private
   submodules. This is what keeps the boundary inspectable.
3. **Top-level `lazybridge` `__init__.py` is the public API surface.**
   Anything not re-exported there is private.
4. **Extensions never appear in the top-level `lazybridge` namespace.**
   Always accessed as `from lazybridge.ext.X import …` (or
   `lazybridge.external_tools.X`).

The boundary check (`tools/check_ext_imports.py`, run in CI) enforces
rules 1–2 with a static AST scan.

## Optional dependencies

Each module that needs heavy third-party packages declares them as
extras in `pyproject.toml`. Users install only what they need:

```bash
pip install lazybridge              # core only
pip install lazybridge[mcp]         # core + MCP
pip install lazybridge[mcp,otel]    # core + MCP + OTel
pip install lazybridge[all]         # everything
```

Importing an extension whose extra is not installed raises a clean
`ImportError` with the install hint.

## Path to 1.0

The 0.7.x line exists to shake out the layout, the boundaries, and the
public API. 1.0 lands when:

- The public API has held across two consecutive minor releases without
  breaking changes
- The boundary CI check is green and has been for several months
- Users are running it in production and reporting back

At 1.0 we'll re-introduce per-module stability tiers if it turns out to
be useful — until then, "everything is alpha" is the simpler honest
contract.
