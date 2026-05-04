"""lazybridge.ext — framework extensions.

Pre-1.0 (0.7.x): everything in lazybridge is ``alpha``. Interfaces may
change between any two releases. Pin exact versions in production.

This sub-package holds **framework-level** extensions — code that
augments the agent runtime itself. Domain tool kits and pre-wired agent
pipelines live elsewhere:

- ``lazybridge.external_tools``     — domain tool packages (read_docs,
  doc_skills, data_downloader, stat_runtime, veo, report_builder)
- ``lazybridge.external_pipelines`` — pre-wired agent compositions
  (quant_agent)

Available framework extensions::

    lazybridge.ext.mcp       Model Context Protocol client (stdio + HTTP)
    lazybridge.ext.otel      OpenTelemetry span exporter
    lazybridge.ext.hil       Human-in-the-loop engines (HumanEngine, SupervisorEngine)
    lazybridge.ext.evals     EvalSuite, EvalCase, llm_judge, assertion helpers
    lazybridge.ext.gateway   Adapter for server-side tool gateways
    lazybridge.ext.planners  Planner factories (DAG builder + blackboard)
    lazybridge.ext.viz       Live + replay pipeline visualizer

See ``docs/guides/core-vs-ext.md`` for the import boundary policy.
"""
