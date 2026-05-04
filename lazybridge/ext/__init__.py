"""lazybridge.ext — extension and domain modules.

Extensions depend on the core framework (``lazybridge.*``) but the core
never imports from ``ext/``. Each extension is self-contained and may
require its own optional dependencies.

Stability: every extension declares ``__stability__``. Four levels exist:

- ``"stable"`` — settled API, breaking changes only across major
  releases. Same commitment level as core.
- ``"beta"`` — interface generally stable; breakage allowed in minor
  releases and called out in the CHANGELOG.
- ``"alpha"`` — interface may change between any two releases.
  Use is fine; pin exact versions in production.
- ``"domain"`` — *domain example shipped with the framework*. Not part
  of the LazyBridge framework contract. Lives in the package as a
  worked reference for the patterns the framework enables, and may be
  removed or extracted to its own package without notice.

See ``docs/guides/core-vs-ext.md`` for the regime policy.

Available modules::

    # stable — framework extensions
    lazybridge.ext.mcp               Model Context Protocol client (stdio + HTTP)
    lazybridge.ext.otel              OpenTelemetry span exporter
    lazybridge.ext.hil               Human-in-the-loop engines (HumanEngine, SupervisorEngine)
    lazybridge.ext.evals             EvalSuite, EvalCase, llm_judge, assertion helpers
    lazybridge.ext.external_tools    Adapter for server-side tool gateways

    # alpha — experimental
    lazybridge.ext.planners          Planner factories (DAG builder + blackboard)

    # domain — worked examples; not framework contract
    lazybridge.ext.stat_runtime      Econometrics & time-series analysis
    lazybridge.ext.data_downloader   Market data ingestion (Yahoo, FRED, ECB)
    lazybridge.ext.quant_agent       Pre-configured quantitative analysis agent
    lazybridge.ext.doc_skills        BM25 local documentation skill runtime
    lazybridge.ext.read_docs         Multi-format document reader
    lazybridge.ext.veo               Google Veo video-generation utilities
    lazybridge.ext.report_builder    Professional HTML report assembler (Markdown + chart PNGs → HTML)
"""
