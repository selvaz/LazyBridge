"""lazybridge.gui — Core, reusable GUI surfaces for LazyBridge objects.

First-class (not an ``ext``) because the primitives here are intended to be
reused by any future LazyBridge GUI — including the planned whole-pipeline
inspector / editor.

Each sub-module provides a browser-based inspector / editor for one kind of
LazyBridge object, built on a shared set of stdlib-only primitives:
``ThreadingHTTPServer`` + token-gated endpoints + an inlined HTML page.

Sub-modules (current / planned):

- ``human`` — browser UI that supplies an ``input_fn`` for ``HumanAgent`` /
  ``SupervisorAgent``. Stable, shipped.

- ``agent`` — read/edit a ``LazyAgent`` (name, system prompt, model, enabled
  tools drawn from the enclosing session's scope). *Planned.*

- ``tool`` — inspect a ``LazyTool`` (schema, description, guidance). *Planned.*

- ``pipeline`` — view a ``LazyTool.chain`` / ``LazyTool.parallel`` topology
  and reorder / toggle participants. *Planned.*

- ``session`` — session-wide inspector: agents, tools, store entries, event
  log, graph topology. *Planned.*

Design principle: keep the HTTP server / token / page-assembly primitives
factored so the pipeline-wide GUI can embed the same panels as sub-views.
"""

__all__: list[str] = []
