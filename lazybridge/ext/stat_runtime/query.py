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

Validation pipeline (audit Z2 — replaces regex-only enforcement):

1. Parse via :mod:`sqlglot` with the DuckDB dialect.  Multi-statement
   SQL, unparseable SQL, or SQL that sqlglot has to fall back to a
   ``Command`` node for (e.g. ``LOAD httpfs``) are all rejected.
2. Walk the AST for forbidden expression types (``Insert``, ``Update``,
   ``Drop``, ``Pragma``, …).  This catches mutations regardless of
   nesting (CTE bodies, subqueries) and immune to comment-injection
   bypasses that defeat regex scans.
3. Walk every ``Func`` for forbidden names (``read_parquet``,
   ``read_csv``, ``glob``, ``s3_*``, …).  Catches schema-qualified
   variants (``main.read_parquet(...)``) and typed reader nodes
   (``ReadParquet``) that sqlglot promotes from ``Anonymous``.
4. Walk every ``Table`` for path-literal replacement scans
   (``FROM '/etc/passwd'``).  AST gives us the literal contents
   directly, with no false positives on identically-named columns
   or in-string references.
5. Regex layer kept as a last-line defence and as a graceful fallback
   when ``sqlglot`` is missing (extension installed without ``[stats]``).
"""

from __future__ import annotations

import hashlib
import logging
import re

from lazybridge.ext.stat_runtime.catalog import DatasetCatalog
from lazybridge.ext.stat_runtime.schemas import QueryResult

_logger = logging.getLogger(__name__)

# Pattern to match dataset('name') or dataset("name")
_DATASET_MACRO_RE = re.compile(
    r"""dataset\(\s*['"]([^'"]+)['"]\s*\)""",
    re.IGNORECASE,
)

# DuckDB file-reader functions that must be blocked in USER sql.
# These are only allowed in the internally-generated expanded SQL.
# Used both by the AST walker (exact name match) and by the regex
# fallback when sqlglot is unavailable.
_FORBIDDEN_FUNC_NAMES: frozenset[str] = frozenset({
    "read_parquet", "read_csv", "read_csv_auto",
    "read_json", "read_json_auto",
    "read_text", "read_blob",
    "st_read",
    "iceberg_scan", "delta_scan",
    "parquet_scan", "parquet_metadata", "parquet_schema",
    "glob",
    "http_get",
})

# Function-name prefixes that should always be rejected (covers
# httpfs_*, s3_*, gcs_* helper families that we don't want to
# enumerate exhaustively).
_FORBIDDEN_FUNC_PREFIXES: tuple[str, ...] = ("httpfs_", "s3_", "gcs_", "azure_")

_FILE_READER_RE = re.compile(
    r"\b(read_parquet|read_csv|read_csv_auto|read_json|read_json_auto"
    r"|read_text|read_blob|st_read|iceberg_scan|delta_scan"
    r"|parquet_scan|parquet_metadata|parquet_schema"
    r"|glob|httpfs_|http_get|s3_)\b",
    re.IGNORECASE,
)

# DuckDB replacement scans: FROM 'path.parquet' / JOIN 'path.csv'
# Matches a path-like quoted string after FROM or JOIN keywords.
# Path indicators: / \ . .. or drive letter (C:)
_PATH_LITERAL_RE = re.compile(
    r"""(?:FROM|JOIN)\s+['"]"""  # FROM or JOIN followed by quote
    r"""(?:"""
    r"""[/\\]"""  # starts with / or \
    r"""|\.\.?[/\\]"""  # starts with ./ or ../
    r"""|[A-Za-z]:[/\\]"""  # Windows drive letter (C:/ C:\)
    r"""|[^'"]*\."""  # contains a dot (file extension)
    r""")""",
    re.IGNORECASE,
)

# Mutation / DDL keywords (regex fallback path only)
_MUTATION_RE = re.compile(
    r"""\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH"""
    r"""|COPY|EXPORT|IMPORT|LOAD|VACUUM|PRAGMA|SET|RESET|CALL"""
    r"""|INSTALL|EXECUTE|PREPARE)\b""",
    re.IGNORECASE,
)


def _path_like(value: str) -> bool:
    """Return True if ``value`` looks like a filesystem / URI path that
    DuckDB would treat as a replacement-scan target.

    Centralised so the AST walker and the regex fallback share the
    same heuristic — the previous regex-only path missed a few edge
    cases (no leading slash but explicit extension at the end of the
    string with no other path separators).
    """
    if not value:
        return False
    # Absolute / relative POSIX paths, Windows drive letters, dot-prefix.
    if value[0] in ("/", "\\") or value.startswith(("./", "../", ".\\", "..\\")):
        return True
    if len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\"):
        return True
    # URI schemes DuckDB knows about.
    lowered = value.lower()
    if any(lowered.startswith(p) for p in (
        "http://", "https://", "s3://", "gcs://", "azure://", "file://",
        "hdfs://", "abfs://", "abfss://", "r2://",
    )):
        return True
    # Filename-with-extension heuristic — DuckDB auto-detects
    # parquet/csv/json/etc. by suffix.
    if "." in value and not value.startswith("'"):
        suffix = value.rsplit(".", 1)[-1].lower()
        if suffix in {
            "parquet", "csv", "tsv", "json", "jsonl", "ndjson",
            "txt", "log", "gz", "zst", "bz2", "arrow", "ipc",
        }:
            return True
    return False


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

        original_sql = sql.strip().rstrip(";")

        # Validate BEFORE macro expansion (checks run on user-provided SQL)
        self._validate(original_sql)

        expanded_sql = self._expand_macros(original_sql)
        normalized_sql = self._normalize(expanded_sql)
        query_hash = self._hash(normalized_sql)

        _logger.debug("Executing query (hash=%s): %s", query_hash[:8], normalized_sql[:200])

        conn = duckdb.connect()
        try:
            # Fetch max_rows+1 to detect truncation without a separate COUNT query
            limited_sql = f"SELECT * FROM ({expanded_sql}) AS _q LIMIT {max_rows + 1}"
            result = conn.execute(limited_sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            truncated = len(rows) > max_rows
            if truncated:
                rows = rows[:max_rows]
            total_rows = len(rows)

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

        Pipeline: AST validation via sqlglot first (audit Z2); regex
        layer kept for defence-in-depth and as a fallback when sqlglot
        isn't installed.
        """
        stripped = sql.strip().rstrip(";").strip()
        if not stripped:
            raise ValueError("Empty SQL query")

        # AST validation — primary line of defence.
        if not _validate_with_sqlglot(stripped):
            # sqlglot unavailable — fall back to regex-only mode and warn.
            import warnings as _warnings

            _warnings.warn(
                "sqlglot is not installed — SQL validation degraded to "
                "regex-only mode (less robust against bypass attempts). "
                "Install with: pip install 'lazybridge[stats]'",
                UserWarning,
                stacklevel=3,
            )

        # Regex layer — defence in depth.  These checks are kept even
        # after AST validation succeeds so a future sqlglot regression
        # or unknown-dialect quirk still gets blocked.

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
            raise ValueError(f"Forbidden SQL keyword: {match.group(0)}. Only SELECT queries are allowed.")

        # Block direct file-reader functions (allowlist enforcement)
        # Users must go through dataset('name'), not read_parquet() etc.
        match = _FILE_READER_RE.search(stripped)
        if match:
            raise ValueError(
                f"Direct file access function '{match.group(0)}' is not allowed. "
                "Use the dataset('name') macro to access registered datasets. "
                "Example: SELECT * FROM dataset('my_data')"
            )

        # Block path-literal replacement scans: FROM '/path/file.parquet'
        # DuckDB auto-detects file extensions and reads them as tables.
        match = _PATH_LITERAL_RE.search(stripped)
        if match:
            raise ValueError(
                "Direct file path in FROM/JOIN clause is not allowed. "
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
            safe_uri = meta.uri.replace("'", "''")
            if meta.file_format == "csv":
                return f"read_csv_auto('{safe_uri}')"
            return f"read_parquet('{safe_uri}')"

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
                        ds_name,
                        meta.time_column,
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


def _validate_with_sqlglot(sql: str) -> bool:
    """Run AST-based validation.  Returns True when sqlglot is
    available and the SQL passed; raises :class:`ValueError` when the
    SQL is rejected; returns False (no raise) when sqlglot isn't
    installed so the caller can degrade to the regex-only path with
    a warning.
    """
    try:
        import sqlglot
        from sqlglot import expressions as exp
        from sqlglot.errors import ParseError, TokenError
    except ImportError:
        return False

    # ── 1. Parse ──────────────────────────────────────────────────────
    # Multi-statement SQL collapses to ``len(statements) > 1`` here, so
    # the classic ``SELECT 1; DROP TABLE x`` smuggling vector dies up
    # front.  Unparseable SQL is rejected outright instead of silently
    # falling through to regex (which would miss novel syntax).
    try:
        statements = sqlglot.parse(sql, dialect="duckdb")
    except (ParseError, TokenError) as exc:
        raise ValueError(f"SQL parse error: {exc}") from exc
    statements = [s for s in statements if s is not None]
    if not statements:
        raise ValueError("Empty SQL query")
    if len(statements) > 1:
        raise ValueError(
            f"Multi-statement SQL is not allowed (got {len(statements)} statements). "
            "Submit one SELECT query at a time."
        )
    root = statements[0]

    # ── 2. Forbidden statement types ──────────────────────────────────
    # Walking the whole tree (not just the root) catches mutations
    # nested inside CTE bodies — ``WITH t AS (SELECT 1) INSERT INTO ...``
    # has a ``Select`` root in some dialects but the ``Insert`` lives
    # one level down.
    _check_forbidden_types(root, exp)

    # ── 3. Allowed root shape ─────────────────────────────────────────
    # SELECT / WITH / set-ops only.  Subquery wrappers are allowed so
    # the validator survives sqlglot wrapping CTE roots in ``Subquery``
    # in some dialect / version combinations.
    allowed_roots = (exp.Select, exp.Subquery, exp.With, exp.Union, exp.Intersect, exp.Except)
    if not isinstance(root, allowed_roots):
        raise ValueError(
            f"Only SELECT / WITH (CTE) / set-operation statements are allowed. "
            f"Got: {type(root).__name__}."
        )

    # ── 4. Forbidden function calls ───────────────────────────────────
    # Walks every Func node in the tree.  Catches:
    #   - Anonymous calls:        ``read_csv_auto(...)``, ``glob(...)``
    #   - Typed reader nodes:     ``ReadParquet(...)``, ``ReadCSV(...)``
    #   - Schema-qualified calls: ``main.read_parquet(...)``
    #     (sqlglot resolves these to the same node types as bare calls)
    for func in root.find_all(exp.Func):
        name = _func_name(func).lower()
        if not name:
            continue
        if name in _FORBIDDEN_FUNC_NAMES or any(
            name.startswith(p) for p in _FORBIDDEN_FUNC_PREFIXES
        ):
            raise ValueError(
                f"Direct file access function '{name}' is not allowed. "
                "Use the dataset('name') macro to access registered datasets."
            )

    # ── 5. Path-literal replacement scans (FROM '/etc/passwd') ────────
    # In sqlglot's AST these surface as a Table whose ``this`` is a
    # quoted Identifier with the path as the literal text.  Fewer
    # false positives than the regex approach because in-string
    # references and column aliases never reach this branch.
    for table in root.find_all(exp.Table):
        ident = table.this
        if isinstance(ident, exp.Identifier) and ident.quoted:
            value = ident.name or ""
            if _path_like(value):
                raise ValueError(
                    "Direct file path in FROM/JOIN clause is not allowed. "
                    "Use the dataset('name') macro to access registered datasets."
                )

    return True


def _check_forbidden_types(root, exp_module) -> None:
    """Walk the AST raising ``ValueError`` on any forbidden node type.

    Forbidden families (kept as a tuple of (class, label) pairs so
    error messages name the offending construct, not just its parent):

    * Mutations: Insert / Update / Delete / Merge / TruncateTable
    * DDL:       Create / Drop / Alter / Detach / Attach
    * Dangerous side-effecting: Copy / Pragma / Set / Install / Use /
      Analyze / LoadData
    * sqlglot fallback: Command (covers VACUUM / LOAD httpfs / any
      syntax sqlglot couldn't parse into a typed node — these are
      always rejected because their semantics are opaque to us)
    """
    forbidden: list[tuple[type, str]] = []
    for cls_name, label in (
        ("Insert", "INSERT"),
        ("Update", "UPDATE"),
        ("Delete", "DELETE"),
        ("Merge", "MERGE"),
        ("TruncateTable", "TRUNCATE"),
        ("Create", "CREATE"),
        ("Drop", "DROP"),
        ("Alter", "ALTER"),
        ("Attach", "ATTACH"),
        ("Detach", "DETACH"),
        ("Copy", "COPY"),
        ("Pragma", "PRAGMA"),
        ("Set", "SET"),
        ("Install", "INSTALL"),
        ("Use", "USE"),
        ("Analyze", "ANALYZE"),
        ("LoadData", "LOAD"),
        ("Command", "unsupported / unparsed statement"),
    ):
        cls = getattr(exp_module, cls_name, None)
        if cls is not None:
            forbidden.append((cls, label))

    forbidden_classes = tuple(c for c, _ in forbidden)
    label_for = {c: lbl for c, lbl in forbidden}

    for node in root.walk():
        if isinstance(node, forbidden_classes):
            label = label_for.get(type(node), type(node).__name__)
            raise ValueError(
                f"Forbidden SQL construct: {label}. Only SELECT queries are allowed."
            )


def _func_name(func) -> str:
    """Best-effort name extraction for a sqlglot ``Func`` node.

    ``Anonymous`` carries the name in ``.name``; typed functions
    (``ReadParquet`` / ``ReadCSV`` / …) expose their canonical
    name via ``sql_name()`` or fall back to the class name.
    """
    name = getattr(func, "name", "") or ""
    if name:
        return name
    sql_name = getattr(func, "sql_name", None)
    if callable(sql_name):
        try:
            return sql_name() or ""
        except Exception:
            pass
    return type(func).__name__


def _import_duckdb():
    from lazybridge.ext.stat_runtime._deps import require_duckdb

    return require_duckdb()
