"""stat_runtime — econometrics and time-series execution layer for LazyBridge.

Provides shared data access, query compilation, model fitting, diagnostics,
visualization, forecasting, run persistence, and LazyTool integrations for
statistical workflows.

Quick start::

    from lazybridge.ext.stat_runtime import StatRuntime

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
from lazybridge.ext.stat_runtime.schemas import (
    # High-level discovery & analysis
    AnalysisDiscoveryResult,
    # Enums
    AnalysisMode,
    AnalysisResult,
    # Tool input schemas (for agent_tool pipelines)
    AnalyzeInput,
    # Original schemas
    ArtifactRecord,
    ArtifactSummary,
    ColumnProfile,
    ColumnRoleInference,
    ColumnSignals,
    CompareModelsInput,
    DataDiscoveryResult,
    DatasetDiscovery,
    DatasetMeta,
    DiagnosticResult,
    DiscoverAnalysesInput,
    DownloadTickersInput,
    FitModelInput,
    FitResult,
    ForecastInput,
    ForecastResult,
    Frequency,
    GetPlotInput,
    GetRunInput,
    ListArtifactsInput,
    ListRunsInput,
    ListUniverseInput,
    ModelFamily,
    ModelSpec,
    ProfileDatasetInput,
    ProfileResult,
    QueryDataInput,
    QueryResult,
    RegisterDatasetInput,
    RunDiagnosticsInput,
    RunRecord,
    RunStatus,
    RunSummary,
    SearchTickersInput,
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
    # Tool input schemas
    "AnalyzeInput",
    "CompareModelsInput",
    "DiscoverAnalysesInput",
    "DownloadTickersInput",
    "FitModelInput",
    "ForecastInput",
    "GetPlotInput",
    "GetRunInput",
    "ListArtifactsInput",
    "ListRunsInput",
    "ListUniverseInput",
    "ProfileDatasetInput",
    "QueryDataInput",
    "RegisterDatasetInput",
    "RunDiagnosticsInput",
    "SearchTickersInput",
]
