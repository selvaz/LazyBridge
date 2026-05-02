# Core vs Extension policy

LazyBridge ships as a single PyPI package with two regimes — `lazybridge/`
(core) and `lazybridge.ext.*` (extensions and worked examples). Both
co-exist in the same install; the regimes differ in their stability
commitment and iteration speed.

| Regime | Location | Stability | Versioning |
|---|---|---|---|
| **Core** | `lazybridge/` (top-level modules) | **Beta** — public API stable in spirit; breaking changes possible in minor releases but called out in the CHANGELOG and avoided where reasonable | Targeting semver once we declare 2.0 / "stable" |
| **Extension** | `lazybridge.ext.*` (subpackages) | Tagged per module — see *Stability tags* below | Per-tag commitment |

Calling core "beta" rather than "stable" is a deliberate honesty: we
haven't run it at scale long enough to claim semver-level commitments.
The policy will tighten once we cross that bar (see *Promotion to stable*
below).

## What goes where

### Core
The minimum a generalist user needs to build agents. Stays small, optimised,
and stable.

- `Agent`, `Envelope`, `Tool`, `Memory`, `Store`, `Session`, `EventLog`
- `LLMEngine` (the human-in-the-loop engines — `HumanEngine` and
  `SupervisorEngine` — are in `lazybridge.ext.hil`)
- `Plan`, `Step`, `from_prev` / `from_start` / `from_step` / `from_parallel` / `from_parallel_all`
- `BaseProvider` and the four officially supported providers
  (Anthropic, OpenAI, Google, DeepSeek)
- `Guard`, `GuardChain`, `ContentGuard`, `LLMGuard`
- Basic exporters (`ConsoleExporter`, `JsonFileExporter`, `CallbackExporter`,
  `FilteredExporter`, `StructuredLogExporter`)
- LiteLLM bridge (`lazybridge.core.providers.litellm.LiteLLMProvider`)
  — core provider bridge; optional install via `pip install lazybridge[litellm]`

### Extensions and worked examples

Everything under `lazybridge.ext.*`. Each module declares its tier via
`__stability__`. Four tiers exist:

| Tier | What it means | Modules today |
|---|---|---|
| **`stable`** | Field-tested, documented, semver-grade commitment. Same level as core. | `mcp`, `otel`, `hil`, `evals`, `external_tools` |
| **`beta`** | Generally stable; breakage possible in minor releases and called out in the CHANGELOG. | (none currently) |
| **`alpha`** | Experimental. Interface may change between any two releases. Use is fine; pin exact versions in production. | `planners` |
| **`domain`** | Worked **example** shipped with the framework — *not part of the LazyBridge framework contract*. Lives in the package as a reference for the patterns the framework enables, and may be removed or extracted to its own package without notice. | `stat_runtime`, `data_downloader`, `quant_agent`, `doc_skills`, `read_docs`, `veo` |

The `domain` tier is the new piece. It exists to be honest about what
the framework *is* versus what *uses* the framework. The framework is
`Agent` + `Plan` + `Tool` + the engines + the providers. The 6
`domain`-tagged modules are realistic examples of what you can build
on top — they happen to be useful, but they aren't framework primitives,
and we don't promise their APIs.

## Stability tags

Every extension module declares its maturity:

```python
# lazybridge/ext/<name>/__init__.py
__stability__ = "stable" | "beta" | "alpha" | "domain"
__lazybridge_min__ = "1.0.0"
```

A module's `__stability__` is part of its public contract. Users can
introspect it programmatically:

```python
import lazybridge.ext.mcp as mcp
assert getattr(mcp, "__stability__") == "stable"
```

### Promotion / demotion paths

- **alpha → beta** — the API has held across one minor release without
  breaking changes.
- **beta → stable** — the API has held across two further minor releases.
  Earns strict semver.
- **stable → core** — see *Promotion to core* below.
- **domain → ext** — a domain module that turns out to be generally
  useful can be promoted by removing the `domain` disclaimer, choosing
  an `alpha` or `beta` tag, and adopting the standard ext API contract.

The `domain` tier never auto-promotes. It's a deliberate choice the
maintainer makes when a module crosses from "example" to "framework
extension".

## Optional dependencies

Each extension that needs heavy third-party packages declares them as
extras in `pyproject.toml`:

```toml
[project.optional-dependencies]
mcp        = ["mcp>=1.0,<2.0"]
otel       = ["opentelemetry-api>=1.20", "opentelemetry-sdk>=1.20"]
stats      = ["statsmodels>=0.14", "pandas>=2.0"]   # cf. pyproject.toml
all        = ["lazybridge[mcp,otel,stats]"]
```

Users install only what they need:

```bash
pip install lazybridge              # core only
pip install lazybridge[mcp]         # core + MCP
pip install lazybridge[mcp,otel]    # core + MCP + OTel
pip install lazybridge[all]         # everything
```

Importing an extension whose extra is not installed raises a clean
`ImportError` with the install hint:

```python
try:
    from lazybridge.ext.mcp import MCP
except ImportError as e:
    # "lazybridge.ext.mcp requires 'mcp>=1.0,<2.0'.  Install with: pip install lazybridge[mcp]"
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
5. **`domain` modules are not part of the framework contract.** A
   release may move, restructure, or extract them to a separate
   package. If you depend on one, pin the lazybridge release and read
   the changelog before upgrading.
6. **Tests for core** live in `tests/unit/` and gate every commit.
   **Tests for ext** live alongside (`tests/unit/ext/`) but may be
   marked `slow` / `requires_extra` and skipped in fast CI.

## Promotion to core

A long-`stable` extension can be promoted to core when:

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

Single-package PyPI release keeps install simple. Tiered stability tags
keep two needs in tension: users want core and stable extensions to
never break under their feet; the maintainers want to ship new patterns
and worked examples without waiting for major-version cadence. The
split-by-namespace + per-module tag lets both happen without
compromising either.
