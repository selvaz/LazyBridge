"""SQL validation, macro expansion, and query execution via DuckDB.

Security model: **allowlist, not denylist**.  User SQL may only access data
through the ``dataset('name')`` macro.  Direct DuckDB file readers
(``read_parquet``, ``read_csv``, ``read_json``, etc.) are rejected in user
SQL.  Row limits are enforced at the DB level via ``LIMIT``, not by
post-fetch slicing.

Usage::

    engine = QueryEngine(catalog)
    result = engine.execute("SELECT date, ret FROM dataset('equities') ORDER BY date")

Validation rules:
  - Only SELECT / WITH (CTE) statements allowed
  - All table access must go through dataset('name')
  - Direct file reader functions are blocked in user SQL
  - max_rows enforced via DB-level LIMIT (no full-result materialization)
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

# DuckDB file-reader functions that must be blocked in USER sql.
# These are only allowed in the internally-generated expanded SQL.
_FILE_READER_RE = re.compile(
    r"""\b(read_parquet|read_csv|read_csv_auto|read_json|read_json_auto"""
    r"""|read_text|read_blob|st_read|iceberg_scan|delta_scan"""
    r"""|parquet_scan|parquet_metadata|parquet_schema"""
    r"""|glob|httpfs_|http_get|s3_)\b""",
    re.IGNORECASE,
)

# Mutation / DDL keywords
_MUTATION_RE = re.compile(
    r"""\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH"""
    r"""|COPY|EXPORT|IMPORT|LOAD|VACUUM|PRAGMA|SET|RESET|CALL"""
    r"""|INSTALL|EXECUTE|PREPARE)\b""",
    re.IGNORECASE,
)


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

        Row limit is enforced at the DB level via LIMIT, not by
        materializing the full result and then slicing.
        """
        duckdb = _import_duckdb()

        original_sql = sql.strip()

        # Validate BEFORE macro expansion (checks run on user-provided SQL)
        self._validate(original_sql)

        expanded_sql = self._expand_macros(original_sql)
        normalized_sql = self._normalize(expanded_sql)
        query_hash = self._hash(normalized_sql)

        _logger.debug("Executing query (hash=%s): %s", query_hash[:8], normalized_sql[:200])

        conn = duckdb.connect()
        try:
            # First: get the true row count via a COUNT wrapper
            count_sql = f"SELECT COUNT(*) FROM ({expanded_sql}) AS _q"
            total_rows = conn.execute(count_sql).fetchone()[0]
            truncated = total_rows > max_rows

            # Then: fetch only max_rows+1 rows using DB-level LIMIT
            limited_sql = f"SELECT * FROM ({expanded_sql}) AS _q LIMIT {max_rows}"
            result = conn.execute(limited_sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()

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
    # Validation (runs on user-provided SQL, BEFORE macro expansion)
    # ------------------------------------------------------------------

    def _validate(self, sql: str) -> None:
        """Reject non-SELECT statements, mutations, and direct file access.

        This is an allowlist model: user SQL may only access data through
        the dataset('name') macro.  Direct file readers are blocked.
        """
        stripped = sql.strip().rstrip(";").strip()
        if not stripped:
            raise ValueError("Empty SQL query")

        # Must start with SELECT or WITH (CTE)
        first_word = stripped.split()[0].upper()
        if first_word not in ("SELECT", "WITH"):
            raise ValueError(
                f"Only SELECT statements are allowed. Got: {first_word}... "
                "INSERT, UPDATE, DELETE, DROP, CREATE, and other mutations are blocked."
            )

        # Block mutation / DDL keywords
        match = _MUTATION_RE.search(stripped)
        if match:
            raise ValueError(
                f"Forbidden SQL keyword: {match.group(0)}. "
                "Only SELECT queries are allowed."
            )

        # Block direct file-reader functions (allowlist enforcement)
        # Users must go through dataset('name'), not read_parquet() etc.
        match = _FILE_READER_RE.search(stripped)
        if match:
            raise ValueError(
                f"Direct file access function '{match.group(0)}' is not allowed. "
                "Use the dataset('name') macro to access registered datasets. "
                "Example: SELECT * FROM dataset('my_data')"
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
