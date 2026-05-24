"""lazybridge.ext — framework extensions.

Pre-1.0 (0.7.x): everything in lazybridge is ``alpha``. Interfaces may
change between any two releases. Pin exact versions in production.

This sub-package holds **framework-level** extensions — code that
augments the agent runtime itself.

Connectors and domain tool kits moved to the sibling ``lazytoolkit``
package in 0.8: the MCP connector (``lazytools.connectors.mcp``), the
external tool gateway (``lazytools.connectors.gateway``), document
reading (``lazytools.documents``), and doc skills
(``lazytools.skills``).  The ``lazybridge.ext.{mcp,gateway}`` and
``lazybridge.external_tools.*`` deprecation shims were removed in 0.9 —
import from ``lazytools`` instead.  HTML/PDF report assembly moved to the
``lazybridge-reports`` package in 0.7.9.

Available framework extensions::

    lazybridge.ext.otel      OpenTelemetry span exporter
    lazybridge.ext.hil       Human-in-the-loop engines (HumanEngine, SupervisorEngine)
    lazybridge.ext.evals     EvalSuite, EvalCase, llm_judge, assertion helpers
    lazybridge.ext.planners  Planner factories (DAG builder + blackboard)
    lazybridge.ext.viz       Live + replay pipeline visualizer

See ``docs/guides/core-vs-ext.md`` for the import boundary policy and
https://lazybridge.com/ for the three-package layout.
"""
