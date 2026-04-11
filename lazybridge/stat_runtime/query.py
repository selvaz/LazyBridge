"""SQL validation, macro expansion, and query execution via DuckDB.

The query contract uses restricted SQL with a small macro layer.
The ``dataset('name')`` macro resolves to the registered dataset's URI.

Usage::

    engine = QueryEngine(catalog)
    result = engine.execute("SELECT date, ret FROM dataset('equities') ORDER BY date")

Validation rules:
  - Only SELECT statements allowed
  - dataset('name') macro expanded to Parquet file read
  - Dangerous functions and file access restricted
  - Normalized SQL hashed for lineage tracking
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from lazybridge.stat_runtime.catalog import DatasetCatalog
from lazybridge.stat_runtime.schemas import QueryResult

_logger = logging.getLogger(__name__)

# Pattern to match dataset('name') or dataset("name")
_DATASET_MACRO_RE = re.compile(
    r"""dataset\(\s*['"]([^'"]+)['"]\s*\)""",
    re.IGNORECASE,
)

# Dangerous SQL patterns to reject
_FORBIDDEN_PATTERNS = [
    re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY)\b", re.IGNORECASE),
    re.compile(r"\b(EXPORT|IMPORT|LOAD)\b", re.IGNORECASE),
    re.compile(r"\bread_csv_auto\b", re.IGNORECASE),
    re.compile(r"\bread_json_auto\b", re.IGNORECASE),
]


class QueryEngine:
    """SQL query validation, macro expansion, and execution."""

    def __init__(self, catalog: DatasetCatalog) -> None:
        self._catalog = catalog

    def execute(
        self,
        sql: str,
        *,
        max_rows: int = 10_000,
    ) -> QueryResult:
        """Validate, expand macros, and execute a SQL query.

        Returns a QueryResult with the data capped at max_rows.
        """
        duckdb = _import_duckdb()

        original_sql = sql.strip()
        self._validate(original_sql)
        expanded_sql = self._expand_macros(original_sql)
        normalized_sql = self._normalize(expanded_sql)
        query_hash = self._hash(normalized_sql)

        _logger.debug("Executing query (hash=%s): %s", query_hash[:8], normalized_sql[:200])

        conn = duckdb.connect()
        try:
            result = conn.execute(expanded_sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            total_rows = len(rows)
            truncated = total_rows > max_rows
            if truncated:
                rows = rows[:max_rows]

            data_json = [dict(zip(columns, row)) for row in rows]
            # Convert non-serializable types
            for row_dict in data_json:
                for k, v in row_dict.items():
                    if hasattr(v, "isoformat"):
                        row_dict[k] = v.isoformat()
                    elif hasattr(v, "item"):
                        row_dict[k] = v.item()
        finally:
            conn.close()

        return QueryResult(
            query_hash=query_hash,
            original_sql=original_sql,
            normalized_sql=normalized_sql,
            columns=columns,
            row_count=total_rows,
            truncated=truncated,
            data_json=data_json,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, sql: str) -> None:
        """Reject non-SELECT statements and dangerous patterns."""
        stripped = sql.strip().rstrip(";").strip()

        # Must start with SELECT or WITH (CTE)
        first_word = stripped.split()[0].upper() if stripped else ""
        if first_word not in ("SELECT", "WITH"):
            raise ValueError(
                f"Only SELECT statements are allowed. Got: {first_word}... "
                "INSERT, UPDATE, DELETE, DROP, CREATE, and other mutations are blocked."
            )

        for pattern in _FORBIDDEN_PATTERNS:
            match = pattern.search(stripped)
            if match:
                raise ValueError(
                    f"Forbidden SQL keyword detected: {match.group(0)}. "
                    "Only SELECT queries are allowed."
                )

    # ------------------------------------------------------------------
    # Macro expansion
    # ------------------------------------------------------------------

    def _expand_macros(self, sql: str) -> str:
        """Replace dataset('name') with read_parquet('uri')."""
        def _replacer(match: re.Match) -> str:
            dataset_name = match.group(1)
            meta = self._catalog.get(dataset_name)
            if meta is None:
                raise ValueError(
                    f"Dataset '{dataset_name}' is not registered. "
                    f"Available: {[d.name for d in self._catalog.list_datasets()]}"
                )
            if meta.file_format == "csv":
                return f"read_csv_auto('{meta.uri}')"
            return f"read_parquet('{meta.uri}')"

        expanded = _DATASET_MACRO_RE.sub(_replacer, sql)

        # Warn if time-series query without ORDER BY
        if "ORDER BY" not in expanded.upper():
            datasets = _DATASET_MACRO_RE.findall(sql)
            for ds_name in datasets:
                meta = self._catalog.get(ds_name)
                if meta and meta.time_column:
                    _logger.warning(
                        "Query on time-series dataset '%s' without ORDER BY. "
                        "Consider adding ORDER BY %s for deterministic results.",
                        ds_name, meta.time_column,
                    )
        return expanded

    # ------------------------------------------------------------------
    # Normalization & hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(sql: str) -> str:
        """Normalize whitespace and remove trailing semicolons."""
        normalized = " ".join(sql.split())
        return normalized.rstrip(";").strip()

    @staticmethod
    def _hash(normalized_sql: str) -> str:
        """SHA-256 hash of normalized SQL for dedup and lineage."""
        return hashlib.sha256(normalized_sql.encode("utf-8")).hexdigest()[:16]


def _import_duckdb():
    from lazybridge.stat_runtime._deps import require_duckdb
    return require_duckdb()
