# Core vs Extension policy

LazyBridge is split into two regimes within a single PyPI package. Both are
under active development; the regimes differ in their stability commitment
and iteration speed.

| Regime | Location | Stability | Versioning |
|---|---|---|---|
| **Core** | `lazybridge/` (top-level modules) | **Beta** — public API stable in spirit; breaking changes possible in minor releases but called out in the CHANGELOG and avoided where reasonable | Targeting semver once we declare 2.0 / "stable" |
| **Extension** | `lazybridge.ext.*` (subpackages) | **Alpha** — interface may change between any two releases | No compat guarantees; pin exact versions in production |

Calling core "beta" rather than "stable" is a deliberate honesty: we
haven't run it at scale long enough to claim semver-level commitments.
The policy will tighten once we cross that bar (see *Promotion to stable*
below).

## What goes where

### Core
The minimum a generalist user needs to build agents. Stays small, optimised,
and stable.

- `Agent`, `Envelope`, `Tool`, `Memory`, `Store`, `Session`, `EventLog`
- `LLMEngine`, `HumanEngine`, `SupervisorEngine`
- `Plan`, `Step`, `from_prev` / `from_start` / `from_step` / `from_parallel` / `from_parallel_all`
- `BaseProvider` and the four officially supported providers
  (Anthropic, OpenAI, Google, DeepSeek)
- `Guard`, `GuardChain`, `ContentGuard`, `LLMGuard`
- Basic exporters (`ConsoleExporter`, `JsonFileExporter`, `OTelExporter`,
  `CallbackExporter`, `FilteredExporter`, `StructuredLogExporter`)
- LiteLLM bridge (`lazybridge.core.providers.litellm.LiteLLMProvider`)

### Extension
Optional, evolving, or domain-specific code. Ships fast in beta.

- `lazybridge.ext.planners` — `make_planner` (DAG builder) and
  `make_blackboard_planner` (todo list)
- `lazybridge.ext.stat_runtime` — econometrics & time-series tools
- `lazybridge.ext.data_downloader` — market data ingestion adapters
- `lazybridge.ext.quant_agent` — pre-configured quant analysis agent
- `lazybridge.ext.doc_skills` — BM25 local documentation skill runtime
- `lazybridge.ext.read_docs` — multi-format document reader
- (planned) `lazybridge.ext.mcp` — Model Context Protocol client
- (planned) `lazybridge.ext.rag` — retrieval primitives + adapters

## Stability tags

Every extension module declares its maturity:

```python
# lazybridge/ext/<name>/__init__.py
__stability__ = "alpha" | "beta" | "stable"
__lazybridge_min__ = "1.0.0"
```

- **`alpha`** — interface may change between any two releases. Default
  for new extensions. Use is fine; pin exact versions in production.
- **`beta`** — interface is generally stable but breaking changes are
  allowed in minor releases. Documented in the module's CHANGELOG entry.
  Same level as core today.
- **`stable`** — strict semver: breaking changes only across major
  releases. Reaching this is the path to promotion (see "Promotion to
  core" below).

Extensions ship at `alpha` by default. They are promoted to `beta` once
the API has been settled across at least one minor release without
breaking changes; to `stable` once two minor releases without changes.

A module's `__stability__` is part of its public contract. Users can
introspect it programmatically (`getattr(lazybridge.ext.X, "__stability__")`)
to gate adoption decisions.

## Optional dependencies

Each extension that needs heavy third-party packages declares them as
extras in `pyproject.toml`:

```toml
[project.optional-dependencies]
mcp        = ["mcp>=0.5.0"]
rag        = ["chromadb>=0.4"]
stat       = ["statsmodels>=0.14", "pandas>=2.0"]
all        = ["lazybridge[mcp,rag,stat]"]
```

Users install only what they need:

```bash
pip install lazybridge              # core only
pip install lazybridge[mcp]         # core + MCP
pip install lazybridge[mcp,rag]     # core + MCP + RAG
pip install lazybridge[all]         # everything
```

Importing an extension whose extra is not installed raises a clean
`ImportError` with the install hint:

```python
try:
    from lazybridge.ext.mcp import MCP
except ImportError as e:
    # "lazybridge.ext.mcp requires 'mcp>=0.5.0'.  Install with: pip install lazybridge[mcp]"
    ...
```

## Architectural rules

These exist to keep the regimes from rotting into each other:

1. **Core never imports from `ext/`.** Extensions depend on core; the
   reverse would create cycles and tie core's stability to ext's
   velocity. Enforced by import-linter / a CI check.
2. **Extensions may depend on other extensions** but should declare it
   explicitly (`pip install lazybridge[a,b]`).
3. **Core's public API** (everything re-exported from `lazybridge`'s
   top-level `__init__.py`) is the SemVer surface. Removing or
   incompatibly changing one of those names requires a major bump.
4. **Extensions never appear in the top-level `lazybridge` namespace.**
   Always accessed as `from lazybridge.ext.X import …`.
5. **Tests for core** live in `tests/unit/` and gate every commit.
   **Tests for ext** live alongside (`tests/unit/ext/`) but may be
   marked `slow` / `requires_extra` and skipped in fast CI.

## Promotion to stable (and to core)

**Alpha → beta.** API has held across one minor release without breaking
changes. Bumps `__stability__` and removes "expect breakage" warnings.

**Beta → stable.** API has held across two minor releases without
breaking changes. Bumps `__stability__`. Earns strict semver.

**Stable ext → core.** A long-stable extension can be promoted to core
when:

- It has been at `__stability__ = "stable"` for ≥ 2 minor releases,
- It has no optional-extra dependencies (or its deps move to core),
- Its API has not changed in those two releases,
- A maintainer signs off the move in the changelog.

Promotion preserves the module name (a deprecated re-export at the old
`lazybridge.ext.X` location stays for one minor release).

**Core itself reaching stable.** When core has run uneventfully across
two consecutive minor releases — no breakages, no major API rework —
it earns the stable tag and strict semver.

## Demotion (when an "ext" sibling outpaces a core feature)

Rare but possible: a feature in core may be re-implemented in ext as the
canonical version. The core feature is then deprecated for one minor
release and removed.

## Why split at all

Single-package PyPI release keeps install simple. Two regimes keep two
needs in tension: users want core to never break under their feet; the
maintainers want to ship new patterns (MCP, RAG, planners, …) without
waiting for major-version cadence. Separating the two by namespace
lets both happen without compromising either.
