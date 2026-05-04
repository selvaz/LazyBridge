"""Regression tests for query.py security sandbox.

Tests that:
- Direct file-reader functions are blocked
- Only SELECT/WITH are allowed
- Mutation keywords are blocked
- The dataset() macro still works
"""

import pytest

from lazybridge.external_tools.stat_runtime.query import QueryEngine


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
            engine._validate("SELECT * FROM (SELECT * FROM read_parquet('/secret.parquet')) t")

    def test_case_insensitive_blocking(self, engine):
        with pytest.raises(ValueError, match="Direct file access"):
            engine._validate("SELECT * FROM READ_PARQUET('/data.parquet')")


class TestPathLiteralBlocking:
    """P0: DuckDB replacement scans via path literals must be blocked."""

    def test_absolute_unix_path(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate("SELECT * FROM '/tmp/secret.parquet'")

    def test_absolute_unix_path_double_quotes(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate('SELECT * FROM "/tmp/secret.parquet"')

    def test_windows_path_forward_slash(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate("SELECT * FROM 'C:/temp/secret.csv'")

    def test_windows_path_backslash(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate(r"SELECT * FROM 'C:\data\file.parquet'")

    def test_relative_dot_slash(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate("SELECT * FROM './local.csv'")

    def test_relative_dot_dot_slash(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate("SELECT * FROM '../outside.csv'")

    def test_filename_with_extension(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate("SELECT * FROM 'data.parquet'")

    def test_join_path_literal(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate("SELECT * FROM dataset('ok') JOIN '/tmp/evil.csv' ON 1=1")

    def test_path_in_cte_body(self, engine):
        with pytest.raises(ValueError, match="Direct file path"):
            engine._validate("WITH t AS (SELECT * FROM '/tmp/secret.parquet') SELECT * FROM t")

    def test_dataset_macro_still_works(self, engine):
        # dataset('name') should NOT be blocked by path-literal check
        engine._validate("SELECT * FROM dataset('equities') ORDER BY date")

    def test_string_literal_in_predicate_allowed(self, engine):
        # Normal string literals in WHERE should not trigger false positive
        engine._validate("SELECT * FROM dataset('ok') WHERE symbol = 'SPY'")

    def test_string_literal_with_dots_in_predicate_allowed(self, engine):
        # Strings with dots in predicates should not be blocked
        engine._validate("SELECT * FROM dataset('ok') WHERE name = 'file.txt'")


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
        """Mutation inside a CTE must still be caught.

        Post-Z2 the AST walker rejects this as "Forbidden SQL
        construct: INSERT" before the regex layer ever runs; the
        regex layer would catch the same query as "Forbidden SQL
        keyword: INSERT" if the AST path were unavailable.  Match on
        the shared "Forbidden SQL" prefix.
        """
        with pytest.raises(ValueError, match="Forbidden SQL"):
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
