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
