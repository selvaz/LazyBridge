"""Regression tests for query.py security sandbox.

Tests that:
- Direct file-reader functions are blocked
- Only SELECT/WITH are allowed
- Mutation keywords are blocked
- The dataset() macro still works
"""

import pytest

from lazybridge.stat_runtime.query import QueryEngine, _FILE_READER_RE, _MUTATION_RE


class FakeCatalog:
    """Minimal catalog stub for validation tests (no DuckDB needed)."""
    def get(self, name):
        return None
    def list_datasets(self):
        return []


@pytest.fixture
def engine():
    return QueryEngine(FakeCatalog())


class TestFileReaderBlocking:
    """P0: query_data must reject direct file reader functions."""

    def test_read_parquet_blocked(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM read_parquet('/etc/passwd')")

    def test_read_csv_blocked(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM read_csv('/tmp/data.csv')")

    def test_read_csv_auto_blocked(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM read_csv_auto('/tmp/data.csv')")

    def test_read_json_blocked(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM read_json('/tmp/data.json')")

    def test_read_json_auto_blocked(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM read_json_auto('/tmp/data.json')")

    def test_parquet_scan_blocked(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM parquet_scan('/data/*.parquet')")

    def test_glob_blocked(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM glob('/data/*')")

    def test_read_parquet_in_subquery_blocked(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate(
                "SELECT * FROM (SELECT * FROM read_parquet('/secret.parquet')) t"
            )

    def test_case_insensitive_blocking(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM READ_PARQUET('/data.parquet')")


class TestMutationBlocking:
    """Mutation and DDL keywords must be blocked (either by first-word or regex)."""

    def test_insert_blocked(self, engine):
        with pytest.raises(ValueError):
            engine._validate("INSERT INTO t VALUES (1)")

    def test_drop_blocked(self, engine):
        with pytest.raises(ValueError):
            engine._validate("DROP TABLE runs")

    def test_create_blocked(self, engine):
        with pytest.raises(ValueError):
            engine._validate("CREATE TABLE t (a INT)")

    def test_attach_blocked(self, engine):
        with pytest.raises(ValueError):
            engine._validate("ATTACH '/tmp/evil.db'")

    def test_copy_blocked(self, engine):
        with pytest.raises(ValueError):
            engine._validate("COPY t TO '/tmp/out.csv'")

    def test_pragma_blocked(self, engine):
        with pytest.raises(ValueError):
            engine._validate("PRAGMA database_list")

    def test_install_blocked(self, engine):
        with pytest.raises(ValueError):
            engine._validate("INSTALL httpfs")

    def test_mutation_in_cte_blocked(self, engine):
        """Mutation keyword inside a CTE must still be caught by the regex."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            engine._validate("WITH t AS (SELECT 1) INSERT INTO x SELECT * FROM t")


class TestSelectAllowed:
    """SELECT and WITH (CTE) must still work."""

    def test_select_allowed(self, engine):
        # Should not raise
        engine._validate("SELECT 1")

    def test_with_cte_allowed(self, engine):
        engine._validate("WITH t AS (SELECT 1) SELECT * FROM t")

    def test_select_with_dataset_macro(self, engine):
        engine._validate("SELECT * FROM dataset('equities') ORDER BY date")

    def test_empty_sql_rejected(self, engine):
        with pytest.raises(ValueError, match="Empty SQL"):
            engine._validate("")

    def test_non_select_rejected(self, engine):
        with pytest.raises(ValueError, match="Only SELECT"):
            engine._validate("DESCRIBE datasets")
