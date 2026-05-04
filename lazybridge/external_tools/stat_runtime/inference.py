"""Column role inference, dataset suggestions, and analysis interpretation.

Pure functions — no side effects, no I/O, no persistence writes.
Used by the high-level discovery and analysis tools.
"""

from __future__ import annotations

import math
import re
from typing import Any

from lazybridge.external_tools.stat_runtime.schemas import (
    ColumnRoleInference,
    DatasetMeta,
    RunRecord,
)

# ---------------------------------------------------------------------------
# Column name pattern banks
# ---------------------------------------------------------------------------

_TIME_NAMES: set[str] = {
    "date",
    "time",
    "timestamp",
    "datetime",
    "dt",
    "period",
    "month",
    "quarter",
    "year",
    "week",
    "day",
    "hour",
    "trade_date",
    "obs_date",
    "report_date",
}
_TIME_DTYPE_PREFIXES: tuple[str, ...] = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "Date",
    "Datetime",
    "Timestamp",
    "Time",
)

_TARGET_NAMES: set[str] = {
    "ret",
    "return",
    "returns",
    "log_ret",
    "log_return",
    "log_returns",
    "close",
    "price",
    "adj_close",
    "adj_price",
    "value",
    "yield",
    "growth",
    "rate",
    "spread",
    "sales",
    "revenue",
    "gdp",
    "cpi",
    "inflation",
}

_ENTITY_NAMES: set[str] = {
    "symbol",
    "ticker",
    "stock",
    "asset",
    "fund",
    "country",
    "region",
    "sector",
    "industry",
    "company",
    "firm",
    "issuer",
}

_IDENTIFIER_NAMES: set[str] = {
    "id",
    "code",
    "isin",
    "cusip",
    "sedol",
    "figi",
    "permno",
    "gvkey",
}

_NUMERIC_DTYPE_PREFIXES: tuple[str, ...] = (
    "float",
    "Float",
    "int",
    "Int",
    "UInt",
    "f32",
    "f64",
    "i32",
    "i64",
    "u32",
    "u64",
    "Decimal",
)

_STRING_DTYPE_PREFIXES: tuple[str, ...] = (
    "str",
    "Str",
    "Utf8",
    "utf8",
    "String",
    "string",
    "Categorical",
    "categorical",
    "object",
    "Object",
)


# ---------------------------------------------------------------------------
# Column role inference
# ---------------------------------------------------------------------------


def infer_column_roles(meta: DatasetMeta) -> list[ColumnRoleInference]:
    """Infer semantic roles for all columns in a dataset.

    Uses heuristics based on column names, dtypes, and declared metadata.
    Returns one ColumnRoleInference per column.
    """
    roles: list[ColumnRoleInference] = []
    name_lower_map = {col: col.lower().strip() for col in meta.columns_schema}

    for col, dtype in meta.columns_schema.items():
        name_lc = name_lower_map[col]
        role, confidence, reason = _infer_single(
            col,
            name_lc,
            dtype,
            meta,
        )
        roles.append(
            ColumnRoleInference(
                column=col,
                dtype=dtype,
                inferred_role=role,
                confidence=confidence,
                reason=reason,
            )
        )

    return roles


def _infer_single(
    col: str,
    name_lc: str,
    dtype: str,
    meta: DatasetMeta,
) -> tuple[str, str, str]:
    """Infer role for a single column. Returns (role, confidence, reason)."""

    # 1. Declared time column — highest precedence
    if meta.time_column and col == meta.time_column:
        return "time", "high", "Declared as time_column in dataset metadata"

    # 2. Declared entity keys
    if col in meta.entity_keys:
        return "entity_key", "high", "Declared in entity_keys"

    # 3. Declared semantic roles
    if col in meta.semantic_roles:
        declared = meta.semantic_roles[col]
        return declared, "high", f"Declared semantic role: {declared}"

    # 4. Time inference from dtype + name
    if dtype.startswith(_TIME_DTYPE_PREFIXES):
        return "time", "high", f"Dtype '{dtype}' is a temporal type"
    if name_lc in _TIME_NAMES:
        return "time", "medium", f"Column name '{col}' matches common time patterns"

    # 5. Entity key inference from name + string dtype
    if name_lc in _ENTITY_NAMES and dtype.startswith(_STRING_DTYPE_PREFIXES):
        return "entity_key", "medium", (f"String column named '{col}' matches entity key patterns")

    # 6. Identifier inference
    if name_lc in _IDENTIFIER_NAMES and dtype.startswith(_STRING_DTYPE_PREFIXES):
        return "identifier", "medium", (f"String column named '{col}' matches identifier patterns")

    # 7. Target inference from name + numeric dtype
    if name_lc in _TARGET_NAMES and dtype.startswith(_NUMERIC_DTYPE_PREFIXES):
        return "target", "medium", (f"Numeric column named '{col}' matches common target variable patterns")

    # 8. Generic numeric → exogenous
    if dtype.startswith(_NUMERIC_DTYPE_PREFIXES):
        return "exogenous", "low", "Numeric column not matching known target patterns"

    # 9. Generic string → unknown
    if dtype.startswith(_STRING_DTYPE_PREFIXES):
        return "unknown", "low", "String column with no recognized role"

    return "unknown", "low", f"Unrecognized dtype '{dtype}'"


# ---------------------------------------------------------------------------
# Dataset suggestions
# ---------------------------------------------------------------------------


def suggest_for_dataset(
    meta: DatasetMeta,
    roles: list[ColumnRoleInference],
) -> list[str]:
    """Generate actionable suggestions for a dataset based on its metadata and roles."""

    suggestions: list[str] = []

    # Time column not set but detected
    time_roles = [r for r in roles if r.inferred_role == "time" and r.confidence != "high"]
    if not meta.time_column and time_roles:
        candidates = ", ".join(r.column for r in time_roles)
        suggestions.append(
            f"Time column not set but detected candidates: {candidates}. Consider re-registering with time_column set."
        )

    # Multiple potential targets
    target_roles = [r for r in roles if r.inferred_role == "target"]
    if len(target_roles) > 1:
        names = ", ".join(r.column for r in target_roles)
        suggestions.append(f"Multiple potential target columns: {names}. Specify target_col explicitly when fitting.")
    elif len(target_roles) == 1:
        suggestions.append(f"Likely target column: '{target_roles[0].column}' ({target_roles[0].reason}).")

    # No profile cached
    if not meta.profile_json:
        suggestions.append(
            f"No profile cached for '{meta.name}'. Call profile_dataset('{meta.name}') for column-level statistics."
        )

    # Panel data hint
    if meta.entity_keys:
        keys = ", ".join(meta.entity_keys)
        suggestions.append(
            f"Panel data detected (entity_keys: [{keys}]). "
            f"Filter by entity using query_sql before fitting single-entity models."
        )

    # Row count guidance
    if meta.row_count is not None:
        if meta.row_count < 30:
            suggestions.append(
                f"Very small dataset ({meta.row_count} rows). "
                f"Most time-series models need 50+ observations for reliable estimates."
            )
        elif meta.row_count < 100:
            suggestions.append(
                f"Small dataset ({meta.row_count} rows). "
                f"ARIMA and OLS should work. GARCH and Markov may need more data."
            )

    return suggestions


# ---------------------------------------------------------------------------
# Analysis interpretation
# ---------------------------------------------------------------------------


def build_interpretation(
    run: RunRecord,
    family: str,
) -> tuple[list[str], list[str], list[str]]:
    """Build (interpretation, warnings, next_steps) from a completed run.

    Returns three lists of human-readable strings:
    - interpretation: what the results mean
    - warnings: concerning findings
    - next_steps: suggested follow-up actions
    """
    interpretation: list[str] = []
    warnings: list[str] = []
    next_steps: list[str] = []

    metrics = run.metrics_json or {}
    params = run.params_json or {}
    diag_list = run.diagnostics_json or []
    spec = run.spec_json or {}

    if run.status != "success":
        warnings.append(f"Run failed: {run.error_message or 'unknown error'}")
        next_steps.append("Check error_message and retry with adjusted parameters.")
        return interpretation, warnings, next_steps

    # Dispatch to family-specific logic
    family_lc = family.lower() if family else run.engine.lower()
    if family_lc == "garch":
        _interpret_garch(params, metrics, diag_list, interpretation, warnings, next_steps)
    elif family_lc == "arima":
        _interpret_arima(params, metrics, diag_list, spec, interpretation, warnings, next_steps)
    elif family_lc == "markov":
        _interpret_markov(params, metrics, diag_list, spec, interpretation, warnings, next_steps)
    elif family_lc == "ols":
        _interpret_ols(params, metrics, diag_list, interpretation, warnings, next_steps)

    # Common: diagnostics summary
    passed = sum(1 for d in diag_list if d.get("passed") is True)
    failed = sum(1 for d in diag_list if d.get("passed") is False)
    total = passed + failed
    if total > 0:
        interpretation.append(f"Diagnostics: {passed}/{total} tests passed.")
        if failed > 0:
            failed_names = [d["test_name"] for d in diag_list if d.get("passed") is False]
            warnings.append(f"Failed diagnostics: {', '.join(failed_names)}.")

    # Common: suggest comparison if no next_steps yet
    if not next_steps:
        next_steps.append("Consider fitting alternative specifications and using compare_models().")

    return interpretation, warnings, next_steps


def _interpret_garch(
    params: dict[str, Any],
    metrics: dict[str, Any],
    diags: list[dict[str, Any]],
    interpretation: list[str],
    warnings: list[str],
    next_steps: list[str],
) -> None:
    alpha = _get_float(params, "alpha[1]")
    beta = _get_float(params, "beta[1]")
    gamma = _get_float(params, "gamma[1]")

    if alpha is not None and beta is not None:
        persistence = alpha + beta
        interpretation.append(f"Volatility persistence (alpha+beta) = {persistence:.4f}.")
        if persistence >= 1.0:
            warnings.append(f"IGARCH-like: persistence >= 1.0 ({persistence:.4f}). Volatility shocks are permanent.")
        elif persistence > 0.95:
            half_life = math.log(0.5) / math.log(persistence)
            interpretation.append(f"High persistence. Half-life of volatility shocks: {half_life:.0f} periods.")

    if gamma is not None and gamma != 0:
        direction = "negative" if gamma > 0 else "positive"
        interpretation.append(
            f"Asymmetric volatility (gamma={gamma:.4f}): "
            f"{direction} shocks have larger volatility impact (leverage effect)."
        )

    # Check for remaining ARCH effects
    sq_lb = [d for d in diags if "squared" in d.get("test_name", "").lower()]
    if sq_lb and any(d.get("passed") is False for d in sq_lb):
        warnings.append("Ljung-Box on squared residuals failed: remaining ARCH effects. Consider increasing p or q.")
        next_steps.append("Try GARCH(2,1) or GARCH(1,2) to capture remaining effects.")

    next_steps.append("Try EGARCH or TARCH for asymmetric volatility comparison.")
    if "nu" not in params:
        next_steps.append("Try dist='t' (Student-t) for fat-tailed returns.")


def _interpret_arima(
    params: dict[str, Any],
    metrics: dict[str, Any],
    diags: list[dict[str, Any]],
    spec: dict[str, Any],
    interpretation: list[str],
    warnings: list[str],
    next_steps: list[str],
) -> None:
    order = spec.get("params", {}).get("order", [])
    if order:
        interpretation.append(f"ARIMA order: ({', '.join(str(o) for o in order)}).")

    seasonal = spec.get("params", {}).get("seasonal_order", [])
    if seasonal and any(s != 0 for s in seasonal):
        interpretation.append(f"Seasonal order: ({', '.join(str(s) for s in seasonal)}).")

    # Residual autocorrelation check
    lb = [d for d in diags if d.get("test_name", "").startswith("Ljung-Box")]
    if lb and any(d.get("passed") is False for d in lb):
        warnings.append("Ljung-Box failed: residual autocorrelation detected. Model may be under-specified.")
        next_steps.append("Increase AR order (p) or MA order (q) and re-fit.")

    if not seasonal or all(s == 0 for s in seasonal):
        next_steps.append("Consider seasonal ARIMA if data has periodic patterns.")


def _interpret_markov(
    params: dict[str, Any],
    metrics: dict[str, Any],
    diags: list[dict[str, Any]],
    spec: dict[str, Any],
    interpretation: list[str],
    warnings: list[str],
    next_steps: list[str],
) -> None:
    k_regimes = spec.get("params", {}).get("k_regimes", 2)
    interpretation.append(f"{k_regimes}-regime Markov switching model.")

    # Regime means
    regime_means = {}
    for key, val in params.items():
        match = re.match(r"const\[(\d+)\]", key)
        if match:
            regime_means[int(match.group(1))] = val

    if regime_means:
        parts = [f"Regime {k}: mean={v:.4f}" for k, v in sorted(regime_means.items())]
        interpretation.append(f"Regime means: {'; '.join(parts)}.")

    # Regime classification certainty
    rcc = [d for d in diags if "certainty" in d.get("test_name", "").lower()]
    if rcc:
        stat = rcc[0].get("statistic")
        if stat is not None and stat < 0.7:
            warnings.append(
                f"Regime classification certainty is low ({stat:.2f} < 0.70). Regimes may not be well-separated."
            )
            next_steps.append("Try fewer regimes or switching_variance=True.")
        elif stat is not None:
            interpretation.append(f"Regime classification certainty: {stat:.2f} (good separation).")

    # Transition persistence
    for key, val in params.items():
        match = re.match(r"p\[(\d+)->(\d+)\]", key)
        if match and match.group(1) == match.group(2):
            regime = int(match.group(1))
            if val > 0.9:
                duration = 1.0 / (1.0 - val) if val < 1.0 else float("inf")
                interpretation.append(
                    f"Regime {regime} is persistent (p={val:.3f}, expected duration: {duration:.0f} periods)."
                )

    if k_regimes == 2:
        next_steps.append("Consider a 3-regime model for comparison.")
    next_steps.append("Use compare_models() to evaluate alternative specifications.")


def _interpret_ols(
    params: dict[str, Any],
    metrics: dict[str, Any],
    diags: list[dict[str, Any]],
    interpretation: list[str],
    warnings: list[str],
    next_steps: list[str],
) -> None:
    r2 = _get_float(metrics, "r_squared")
    adj_r2 = _get_float(metrics, "adj_r_squared")
    f_pval = _get_float(metrics, "f_pvalue")

    if r2 is not None:
        interpretation.append(f"R-squared: {r2:.4f}.")
    if adj_r2 is not None:
        interpretation.append(f"Adjusted R-squared: {adj_r2:.4f}.")

    if f_pval is not None:
        if f_pval < 0.01:
            interpretation.append("F-test: model is significant at 1% level.")
        elif f_pval < 0.05:
            interpretation.append("F-test: model is significant at 5% level.")
        else:
            warnings.append(f"F-test p-value = {f_pval:.4f}: model may not be significant.")

    # Durbin-Watson
    dw = [d for d in diags if "durbin" in d.get("test_name", "").lower()]
    if dw:
        stat = dw[0].get("statistic")
        if stat is not None and (stat < 1.5 or stat > 2.5):
            warnings.append(
                f"Durbin-Watson = {stat:.2f}: potential autocorrelation in residuals. "
                f"Consider ARIMA for time-series data."
            )
            next_steps.append("Fit ARIMA if this is a time-series.")

    next_steps.append("Add exogenous variables for richer regression models.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_float(d: dict[str, Any], key: str) -> float | None:
    """Safely extract a float from a dict, returning None on failure."""
    val = d.get(key)
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Natural language dataset summary
# ---------------------------------------------------------------------------


def generate_dataset_summary(
    meta: DatasetMeta,
    roles: list[ColumnRoleInference],
) -> str:
    """Generate a concise one-line natural language summary of a dataset."""
    parts: list[str] = []

    # Business description takes precedence
    if meta.business_description:
        return meta.business_description

    # Row count + frequency
    if meta.row_count is not None:
        freq_str = str(meta.frequency) if meta.frequency else ""
        parts.append(f"{meta.row_count:,} {freq_str} observations".strip())

    # Time range hint (from profile if available)
    time_roles = [r for r in roles if r.inferred_role == "time"]
    if time_roles:
        parts.append(f"time column: {time_roles[0].column}")

    # Target candidates
    targets = [r for r in roles if r.inferred_role == "target"]
    if meta.canonical_target:
        parts.append(f"target: {meta.canonical_target}")
    elif targets:
        names = ", ".join(r.column for r in targets[:3])
        parts.append(f"likely target(s): {names}")

    # Entity keys
    if meta.entity_keys:
        keys = ", ".join(meta.entity_keys[:3])
        parts.append(f"grouped by {keys}")

    # Numeric column count
    numeric = [r for r in roles if r.inferred_role in ("target", "exogenous")]
    if numeric:
        parts.append(f"{len(numeric)} numeric columns")

    if not parts:
        return f"Dataset '{meta.name}' with {len(meta.columns_schema)} columns."

    return f"{meta.name}: {'; '.join(parts)}."


# ---------------------------------------------------------------------------
# Analysis mode resolution
# ---------------------------------------------------------------------------

_MODE_TO_FAMILY: dict[str, str] = {
    "forecast": "arima",
    "volatility": "garch",
    "regime": "markov",
}

_MODE_ASSUMPTIONS: dict[str, list[str]] = {
    "describe": [
        "No model fitting — descriptive statistics and stationarity tests only.",
    ],
    "forecast": [
        "ARIMA model assumes the series is stationary (or can be differenced to stationarity).",
        "Forecast accuracy degrades with horizon length.",
        "Default order ARIMA(1,0,1) if not specified; adjust based on ACF/PACF.",
    ],
    "volatility": [
        "GARCH model assumes volatility clustering in the return series.",
        "Target should be returns, not prices.",
        "Default GARCH(1,1) with normal distribution; Student-t recommended for fat tails.",
    ],
    "regime": [
        "Markov switching assumes the data-generating process switches between distinct states.",
        "Default 2 regimes with switching variance.",
        "Requires sufficient data for regime identification (typically 200+ observations).",
    ],
}


def resolve_analysis_mode(
    mode: str,
    meta: DatasetMeta | None,
    roles: list[ColumnRoleInference] | None,
    target_col: str | None,
) -> tuple[str, str, list[str]]:
    """Resolve an AnalysisMode into (family, rationale, assumptions).

    For 'recommend' mode, inspects dataset metadata and roles to choose
    the best analysis. For explicit modes, maps directly.

    Returns:
        (family, rationale, assumptions) where family is a ModelFamily string.
    """
    mode_lc = mode.lower()

    # Direct mode mappings
    if mode_lc in _MODE_TO_FAMILY:
        family = _MODE_TO_FAMILY[mode_lc]
        rationale = f"Mode '{mode_lc}' maps to {family.upper()} family."
        assumptions = _MODE_ASSUMPTIONS.get(mode_lc, [])
        return family, rationale, assumptions

    if mode_lc == "describe":
        # Describe mode still needs a family for the runner — use OLS as baseline
        return (
            "ols",
            (
                "Describe mode: fitting a simple trend model for diagnostic summary. "
                "Focus is on data profiling, stationarity tests, and distribution analysis."
            ),
            _MODE_ASSUMPTIONS["describe"],
        )

    # Recommend mode — inspect data to choose
    if mode_lc == "recommend":
        return _recommend_family(meta, roles, target_col)

    # Fallback: treat as family name directly (backward compat)
    if mode_lc in ("ols", "arima", "garch", "markov"):
        return mode_lc, f"Explicit family '{mode_lc}' specified.", []

    raise ValueError(
        f"Unknown analysis mode: '{mode}'. "
        f"Valid modes: describe, forecast, volatility, regime, recommend, "
        f"or explicit families: ols, arima, garch, markov."
    )


def _recommend_family(
    meta: DatasetMeta | None,
    roles: list[ColumnRoleInference] | None,
    target_col: str | None,
) -> tuple[str, str, list[str]]:
    """Auto-select the best analysis family based on dataset characteristics."""
    reasons: list[str] = []

    # Check for time column
    has_time = False
    if meta and meta.time_column:
        has_time = True
    elif roles:
        has_time = any(r.inferred_role == "time" for r in roles)

    # Check for return-like target
    target_is_return = False
    if target_col:
        tgt_lc = target_col.lower()
        target_is_return = tgt_lc in _TARGET_NAMES or "ret" in tgt_lc

    # Decision logic
    if target_is_return and has_time:
        reasons.append(f"Target '{target_col}' looks like a return series.")
        reasons.append("Time column present — time-series analysis appropriate.")
        reasons.append("Recommending GARCH for volatility analysis of returns.")
        return "garch", " ".join(reasons), _MODE_ASSUMPTIONS["volatility"]

    if has_time:
        reasons.append("Time column present — time-series analysis appropriate.")
        if meta and meta.row_count and meta.row_count >= 200:
            reasons.append(f"Sufficient data ({meta.row_count} rows) for ARIMA forecasting.")
            return "arima", " ".join(reasons), _MODE_ASSUMPTIONS["forecast"]
        else:
            reasons.append("Recommending ARIMA for time-series forecasting.")
            return "arima", " ".join(reasons), _MODE_ASSUMPTIONS["forecast"]

    # No time column — default to OLS
    reasons.append("No time column detected — using linear regression.")
    return (
        "ols",
        " ".join(reasons),
        [
            "OLS assumes a linear relationship between target and predictors.",
            "Residuals should be independent and normally distributed.",
        ],
    )


def get_model_assumptions(family: str) -> list[str]:
    """Return standard assumptions for a model family."""
    _FAMILY_ASSUMPTIONS: dict[str, list[str]] = {
        "ols": [
            "Linear relationship between target and predictors.",
            "Residuals are independent and normally distributed.",
            "No perfect multicollinearity among predictors.",
        ],
        "arima": _MODE_ASSUMPTIONS["forecast"],
        "garch": _MODE_ASSUMPTIONS["volatility"],
        "markov": _MODE_ASSUMPTIONS["regime"],
    }
    return _FAMILY_ASSUMPTIONS.get(family.lower(), [])
