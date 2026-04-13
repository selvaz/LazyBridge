"""Pydantic models defining every contract in the statistical runtime.

All tools, engines, and persistence layers speak through these types.
No heavy dependencies — only pydantic (a core LazyBridge dep).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelFamily(StrEnum):
    OLS = "ols"
    ARIMA = "arima"
    GARCH = "garch"
    MARKOV = "markov"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Frequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    INTRADAY = "intraday"
    IRREGULAR = "irregular"


class AnalysisMode(StrEnum):
    """Goal-oriented analysis modes for analyze().

    The LLM picks a goal, not a model family.  The runtime maps
    each mode to the appropriate model/workflow automatically.
    """
    DESCRIBE = "describe"
    FORECAST = "forecast"
    VOLATILITY = "volatility"
    REGIME = "regime"
    RECOMMEND = "recommend"


# ---------------------------------------------------------------------------
# Dataset contracts
# ---------------------------------------------------------------------------

class DatasetMeta(BaseModel):
    """Metadata for a registered dataset."""
    dataset_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str = Field(description="Logical name, e.g. equities.daily_returns")
    version: str = "1"
    uri: str = Field(description="Parquet file path or directory")
    file_format: str = "parquet"
    columns_schema: dict[str, str] = Field(
        default_factory=dict,
        description="Column name -> dtype string",
    )
    frequency: Frequency = Frequency.DAILY
    time_column: str | None = None
    entity_keys: list[str] = Field(default_factory=list)
    semantic_roles: dict[str, str] = Field(
        default_factory=dict,
        description="Column -> role (target, return, price, exogenous, ...)",
    )
    profile_json: dict[str, Any] = Field(
        default_factory=dict,
        description="Cached profile summary: null rates, row count, basic stats",
    )
    row_count: int | None = None
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Semantic layer — user-supplied business context
    business_description: str | None = Field(
        default=None,
        description="Human-readable description of what this dataset represents",
    )
    canonical_target: str | None = Field(
        default=None,
        description="Explicitly declared preferred target column for analysis",
    )
    identifiers_to_ignore: list[str] = Field(
        default_factory=list,
        description="Columns to exclude from modeling (IDs, hashes, keys)",
    )


class ColumnProfile(BaseModel):
    """Per-column profiling statistics."""
    dtype: str
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: int | None = None
    min_val: Any = None
    max_val: Any = None
    mean: float | None = None
    std: float | None = None


class ProfileResult(BaseModel):
    """Dataset profiling output."""
    dataset_name: str
    row_count: int
    columns: dict[str, ColumnProfile] = Field(
        default_factory=dict,
        description="Column name -> ColumnProfile",
    )


# ---------------------------------------------------------------------------
# Query contracts
# ---------------------------------------------------------------------------

class QueryResult(BaseModel):
    """Result of a validated and executed query."""
    query_hash: str
    original_sql: str
    normalized_sql: str
    columns: list[str]
    row_count: int
    truncated: bool = False
    data_json: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Row-oriented data (capped at max_rows)",
    )


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    """Specification for a model fit request."""
    family: ModelFamily
    target_col: str
    exog_cols: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Family-specific parameters (p, q, order, etc.)",
    )
    dataset_name: str | None = None
    query_sql: str | None = None
    time_col: str | None = None
    forecast_steps: int | None = Field(
        default=None,
        description="If set, auto-generate a forecast after fitting",
    )


# ---------------------------------------------------------------------------
# Fit result
# ---------------------------------------------------------------------------

class FitResult(BaseModel):
    """Structured output of a model fit."""
    family: ModelFamily
    summary_text: str = ""
    params: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    residuals_json: list[float] = Field(default_factory=list)
    fitted_values_json: list[float] = Field(default_factory=list)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific extras (regime probs, volatility, etc.)",
    )


# ---------------------------------------------------------------------------
# Forecast result
# ---------------------------------------------------------------------------

class ForecastResult(BaseModel):
    """Structured forecast output."""
    family: ModelFamily
    steps: int
    point_forecast: list[float] = Field(default_factory=list)
    lower_ci: list[float] = Field(default_factory=list)
    upper_ci: list[float] = Field(default_factory=list)
    ci_level: float = 0.95
    dates: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific forecast extras",
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class DiagnosticResult(BaseModel):
    """Structured diagnostic output."""
    test_name: str
    statistic: float | None = None
    p_value: float | None = None
    passed: bool | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    interpretation: str = ""


# ---------------------------------------------------------------------------
# Artifact record
# ---------------------------------------------------------------------------

class ArtifactRecord(BaseModel):
    """Metadata for a stored artifact (plot, data export, summary)."""
    run_id: str
    name: str
    artifact_type: str = Field(description="plot, data, summary, forecast")
    file_format: str = Field(description="png, svg, json, csv, parquet")
    path: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------

class RunRecord(BaseModel):
    """Persisted run metadata — the core execution ledger entry."""
    run_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    dataset_name: str = ""
    query_hash: str = ""
    spec_json: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    fit_summary: str = ""
    params_json: dict[str, float] = Field(default_factory=dict)
    metrics_json: dict[str, float] = Field(default_factory=dict)
    diagnostics_json: list[dict[str, Any]] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    engine: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_secs: float | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# High-level discovery & analysis contracts
# ---------------------------------------------------------------------------

class ColumnRoleInference(BaseModel):
    """Inferred semantic role for a dataset column."""
    column: str
    dtype: str
    inferred_role: str = Field(
        description="time, target, entity_key, exogenous, identifier, or unknown",
    )
    confidence: str = Field(description="high, medium, or low")
    reason: str = Field(description="Human-readable explanation of inference")


class ColumnSignals(BaseModel):
    """Lightweight quality signals for a column (from cached profile)."""
    null_pct: float | None = None
    unique_count: int | None = None
    min_val: Any = None
    max_val: Any = None
    mean: float | None = None


class DatasetDiscovery(BaseModel):
    """Enriched dataset metadata for LLM discovery."""
    name: str
    uri: str
    file_format: str
    frequency: str
    row_count: int | None = None
    time_column: str | None = None
    entity_keys: list[str] = Field(default_factory=list)
    columns: dict[str, str] = Field(
        default_factory=dict,
        description="Column name -> dtype string",
    )
    column_roles: list[ColumnRoleInference] = Field(default_factory=list)
    column_signals: dict[str, ColumnSignals] = Field(
        default_factory=dict,
        description="Column name -> quality signals (from cached profile, if available)",
    )
    suggestions: list[str] = Field(default_factory=list)
    has_profile: bool = False
    # Semantic layer
    business_description: str | None = None
    canonical_target: str | None = None
    summary: str = Field(
        default="",
        description="Auto-generated one-line natural language summary",
    )


class DataDiscoveryResult(BaseModel):
    """Result of discover_data() — all registered datasets with enriched metadata."""
    datasets: list[DatasetDiscovery] = Field(default_factory=list)
    total_datasets: int = 0
    suggestions: list[str] = Field(default_factory=list)


class ArtifactSummary(BaseModel):
    """Lightweight artifact reference for discovery results."""
    name: str
    artifact_type: str = Field(description="plot, data, summary, or forecast")
    file_format: str = ""
    path: str = ""
    description: str = ""


class RunSummary(BaseModel):
    """Enriched run summary for analysis discovery."""
    run_id: str
    dataset_name: str = ""
    engine: str = ""
    status: str = ""
    created_at: str = ""
    duration_secs: float | None = None
    # Key metrics inline
    aic: float | None = None
    bic: float | None = None
    log_likelihood: float | None = None
    # Diagnostics summary
    diagnostics_passed: int = 0
    diagnostics_failed: int = 0
    diagnostics_total: int = 0
    # Spec highlights
    target_col: str = ""
    model_params: dict[str, Any] = Field(default_factory=dict)
    # Full artifact catalog for this run
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    error_message: str | None = None


class AnalysisDiscoveryResult(BaseModel):
    """Result of discover_analyses() — all runs with inline metrics and artifacts."""
    runs: list[RunSummary] = Field(default_factory=list)
    total_runs: int = 0
    datasets_analyzed: list[str] = Field(default_factory=list)
    best_by_aic: str | None = Field(
        default=None, description="run_id with lowest AIC",
    )
    best_by_bic: str | None = Field(
        default=None, description="run_id with lowest BIC",
    )
    suggestions: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Structured output of analyze() — enriched single-call analysis result."""
    run_id: str
    status: str = "success"
    engine: str = ""
    dataset_name: str = ""
    target_col: str = ""
    # Mode selection rationale
    mode: str = Field(default="", description="Analysis mode that was used")
    mode_rationale: str = Field(
        default="",
        description="Why this analysis mode/family was chosen",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Key assumptions the model makes about the data",
    )
    # Model results
    params: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    fit_summary: str = ""
    # Diagnostics with pass/fail assessment
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics_passed: int = 0
    diagnostics_failed: int = 0
    model_adequate: bool = False
    # Forecast (if requested)
    forecast: dict[str, Any] | None = None
    # Inline artifact catalog
    plots: list[ArtifactSummary] = Field(default_factory=list)
    data_artifacts: list[ArtifactSummary] = Field(default_factory=list)
    # Interpretation for LLM narrative
    interpretation: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    # Timing
    duration_secs: float | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Tool input schemas — used as output_schema for sub-agent → function chains
# ---------------------------------------------------------------------------

class AnalyzeInput(BaseModel):
    """Input schema for the analyze() tool."""
    dataset_name: str = Field(description="Registered dataset name")
    target_col: str | None = Field(default=None, description="Target column (auto-resolved from metadata if omitted)")
    mode: str = Field(default="recommend", description="Analysis goal: describe, forecast, volatility, regime, recommend")
    time_col: str | None = Field(default=None, description="Time column for ordering (auto-detected if not set)")
    forecast_steps: int | None = Field(default=None, description="Forecast steps (None = auto based on mode)")
    group_col: str | None = Field(default=None, description="Column to filter/segment by (e.g. 'symbol')")
    group_value: str | None = Field(default=None, description="Value to filter group_col to (e.g. 'SPY')")
    params: dict[str, Any] | None = Field(default=None, description="Expert override: model parameters")


class FitModelInput(BaseModel):
    """Input schema for the fit_model() tool."""
    family: str = Field(description="Model family: ols, arima, garch, or markov")
    target_col: str = Field(description="Target column name in the dataset")
    dataset_name: str | None = Field(default=None, description="Registered dataset name")
    query_sql: str | None = Field(default=None, description="SQL query to extract data (alternative to dataset_name)")
    exog_cols: list[str] | None = Field(default=None, description="Exogenous/independent variable columns")
    params: dict[str, Any] | None = Field(default=None, description="Model-specific parameters (e.g. {'p': 1, 'q': 1} for GARCH)")
    forecast_steps: int | None = Field(default=None, description="Number of forecast steps (None = no forecast)")
    time_col: str | None = Field(default=None, description="Time column for ordering")


class ForecastInput(BaseModel):
    """Input schema for the forecast_model() tool."""
    run_id: str = Field(description="Run ID from a previous fit_model call")
    steps: int = Field(description="Number of forecast steps")
    ci_level: float = Field(default=0.95, description="Confidence interval level (0-1)")


class QueryDataInput(BaseModel):
    """Input schema for the query_data() tool."""
    sql: str = Field(description="SQL SELECT query. Use dataset('name') to reference registered datasets.")
    max_rows: int = Field(default=5000, description="Maximum rows to return")


class ProfileDatasetInput(BaseModel):
    """Input schema for the profile_dataset() tool."""
    name: str = Field(description="Name of a registered dataset")


class RunDiagnosticsInput(BaseModel):
    """Input schema for the run_diagnostics() tool."""
    series_name: str = Field(description="Dataset name for stationarity tests")
    column: str = Field(description="Column to test")


class CompareModelsInput(BaseModel):
    """Input schema for the compare_models() tool."""
    run_ids: list[str] = Field(description="List of run IDs to compare")


class RegisterDatasetInput(BaseModel):
    """Input schema for the register_dataset() tool."""
    name: str = Field(description="Logical name for the dataset (e.g. 'equities.daily')")
    uri: str = Field(description="File path to a Parquet or CSV file")
    time_column: str | None = Field(default=None, description="Primary time/date column name")
    frequency: str = Field(default="daily", description="Data frequency: daily, weekly, monthly, quarterly, annual, intraday, irregular")
    entity_keys: list[str] | None = Field(default=None, description="Key columns (e.g. ['symbol', 'country'])")
    business_description: str | None = Field(default=None, description="What this dataset represents")
    canonical_target: str | None = Field(default=None, description="Preferred target column for analysis")
    identifiers_to_ignore: list[str] | None = Field(default=None, description="Columns to exclude from modeling")


class GetRunInput(BaseModel):
    """Input schema for the get_run() tool."""
    run_id: str = Field(description="Run ID to retrieve")


class ListRunsInput(BaseModel):
    """Input schema for the list_runs() tool."""
    dataset_name: str | None = Field(default=None, description="Filter by dataset name")
    limit: int = Field(default=20, description="Maximum runs to return")


class ListArtifactsInput(BaseModel):
    """Input schema for the list_artifacts() tool."""
    run_id: str = Field(description="Run ID to list artifacts for")
    artifact_type: str | None = Field(default=None, description="Filter: plot, data, summary, forecast")


class GetPlotInput(BaseModel):
    """Input schema for the get_plot() tool."""
    run_id: str = Field(description="Run ID")
    name: str = Field(description="Plot name (e.g. 'residuals', 'volatility', 'forecast', 'regimes')")


class DiscoverAnalysesInput(BaseModel):
    """Input schema for the discover_analyses() tool."""
    dataset_name: str | None = Field(default=None, description="Filter by dataset name")
    limit: int = Field(default=20, description="Maximum runs to return")


# -- Data downloader input schemas --

class ListUniverseInput(BaseModel):
    """Input schema for the list_universe() tool."""
    asset_class: str | None = Field(default=None, description="Filter by asset class: EQUITY, FIXED_INCOME, COMMODITIES, REAL_ESTATE, ALTERNATIVES, MACRO, FX")
    sub_asset_class: str | None = Field(default=None, description="Filter by sub-asset class (e.g. 'Developed', 'Government', 'Energy')")


class SearchTickersInput(BaseModel):
    """Input schema for the search_tickers() tool."""
    query: str = Field(description="Search query: ticker symbol, name, asset class, sector, or country")


class DownloadTickersInput(BaseModel):
    """Input schema for the download_tickers() tool."""
    tickers: list[str] = Field(description="Ticker symbols to download (e.g. ['SPY', 'AAPL', 'DGS10'])")
    start: str | None = Field(default=None, description="Start date YYYY-MM-DD (default: 2000-01-01)")
    end: str | None = Field(default=None, description="End date YYYY-MM-DD (default: today)")
    auto_register: bool = Field(default=True, description="Auto-register downloaded data in stat_runtime")
