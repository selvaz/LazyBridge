"""stat_runtime — econometrics and time-series execution layer for LazyBridge.

Provides shared data access, query compilation, model fitting, diagnostics,
visualization, forecasting, run persistence, and LazyTool integrations for
statistical workflows.

Quick start::

    from lazybridge.stat_runtime import StatRuntime

    rt = StatRuntime()
    rt.catalog.register_parquet("equities", "/data/returns.parquet",
                                 time_column="date", entity_keys=["symbol"])

    run = rt.execute(ModelSpec(family="garch", target_col="ret",
                               dataset_name="equities", params={"p": 1, "q": 1}))
    print(run.metrics_json)
    print(run.artifact_paths)

Install::

    pip install lazybridge[stats]
"""

__version__ = "0.1.0"

# Schemas are always importable (no heavy deps — only pydantic)
from lazybridge.stat_runtime.schemas import (
    # Enums
    AnalysisMode,
    Frequency,
    ModelFamily,
    RunStatus,
    # High-level discovery & analysis
    AnalysisDiscoveryResult,
    AnalysisResult,
    ArtifactSummary,
    ColumnRoleInference,
    ColumnSignals,
    DataDiscoveryResult,
    DatasetDiscovery,
    RunSummary,
    # Original schemas
    ArtifactRecord,
    ColumnProfile,
    DatasetMeta,
    DiagnosticResult,
    FitResult,
    ForecastResult,
    ModelSpec,
    ProfileResult,
    QueryResult,
    RunRecord,
)

__all__ = [
    # Enums
    "ModelFamily",
    "RunStatus",
    "Frequency",
    # Dataset
    "DatasetMeta",
    "ColumnProfile",
    "ProfileResult",
    # Query
    "QueryResult",
    # Model
    "ModelSpec",
    "FitResult",
    "ForecastResult",
    # Diagnostics
    "DiagnosticResult",
    # Artifacts
    "ArtifactRecord",
    # Run
    "RunRecord",
    # High-level discovery & analysis
    "AnalysisMode",
    "ColumnRoleInference",
    "ColumnSignals",
    "DatasetDiscovery",
    "DataDiscoveryResult",
    "ArtifactSummary",
    "RunSummary",
    "AnalysisDiscoveryResult",
    "AnalysisResult",
]
