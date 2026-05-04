"""Adversarial / red-team tests for the SQL sandbox.

These tests exercise bypass attempts the regex-only validator either
missed or false-positived on.  The post-Z2 AST validator (sqlglot
DuckDB dialect) is the primary defence; the regex layer is kept as
defence-in-depth, so each test asserts that the full pipeline
rejects (or accepts) the input.

Keep this file separate from ``test_query_sandbox.py`` — that file
covers the original happy-path and obvious-block cases; this one
covers bypass classes.
"""

from __future__ import annotations

import pytest

from lazybridge.ext.stat_runtime.query import QueryEngine


class _Catalog:
    """Catalog stub: ``ok`` resolves, everything else does not."""

    def get(self, name):
        if name == "ok":

            class _Meta:
                uri = "/data/ok.parquet"
                file_format = "parquet"
                time_column = None

            return _Meta()
        return None

    def list_datasets(self):
        return []


@pytest.fixture
def engine():
    return QueryEngine(_Catalog())


# ---------------------------------------------------------------------------
# Multi-statement smuggling — the classic SQL-injection vector.
# ---------------------------------------------------------------------------


class TestMultiStatement:
    """Attacker hides a mutation behind a benign first statement."""

    def test_select_then_drop_rejected(self, engine):
        # Pre-Z2: regex caught DROP via _MUTATION_RE.  AST also catches
        # the multi-statement count.  Either rejection mode is fine.
        with pytest.raises(ValueError):
            engine._validate("SELECT 1; DROP TABLE runs")

    def test_select_then_insert_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("SELECT 1; INSERT INTO t VALUES (1)")

    def test_three_statements_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("SELECT 1; SELECT 2; ATTACH '/tmp/evil.db'")

    def test_select_then_select_rejected_as_multistatement(self, engine):
        """Even two SELECTs is rejected — multi-statement is the gate,
        not the contents.  Caller should issue one query at a time."""
        with pytest.raises(ValueError, match=r"(?i)multi-?statement|forbidden"):
            engine._validate("SELECT 1; SELECT 2")


# ---------------------------------------------------------------------------
# Comment / whitespace tricks
# ---------------------------------------------------------------------------


class TestCommentBypass:
    """Comments and unusual whitespace must not let mutations through."""

    def test_inline_comment_in_mutation_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("/* friendly */ DROP TABLE runs")

    def test_block_comment_around_keyword_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("DELE/* x */TE FROM runs")

    def test_line_comment_then_drop_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("-- safe?\nDROP TABLE runs")


# ---------------------------------------------------------------------------
# String-literal false positives — AST should ALLOW these.
# Pre-Z2 the regex matched substrings inside literals.
# ---------------------------------------------------------------------------


class TestStringLiteralFalsePositives:
    """The AST walker doesn't false-positive on forbidden tokens
    that appear inside string literals.

    Pre-Z2 a literal like ``'remember to read_parquet later'`` would
    match ``_FILE_READER_RE`` and falsely reject the query.  Post-Z2
    the AST sees this as a Literal child of a comparison, not a
    function call, and the regex layer still triggers — so as a
    pragmatic compromise we KEEP rejecting these (defence-in-depth)
    BUT document that the AST path itself does not.
    """

    def test_ast_alone_allows_forbidden_word_in_string_literal(self, engine):
        """The AST validator (called directly) does not reject a
        forbidden word that appears purely as string-literal content.
        """
        from lazybridge.ext.stat_runtime.query import _validate_with_sqlglot

        # AST in isolation accepts this (no actual file read happens).
        ok = _validate_with_sqlglot("SELECT * FROM dataset('ok') WHERE notes = 'remember to read_parquet later'")
        assert ok is True

    def test_full_pipeline_still_blocks_via_regex_defense_in_depth(self, engine):
        """Composite pipeline (AST + regex) still blocks the literal —
        intentional belt-and-braces.  Document that fix path here:
        if you want literal substrings allowed, drop the regex layer."""
        with pytest.raises(ValueError):
            engine._validate("SELECT * FROM dataset('ok') WHERE notes = 'remember to read_parquet later'")


# ---------------------------------------------------------------------------
# Schema-qualified function calls
# ---------------------------------------------------------------------------


class TestSchemaQualifiedReaders:
    """``main.read_parquet(...)`` must be rejected like the bare call."""

    def test_main_qualified_read_parquet_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("SELECT * FROM main.read_parquet('/etc/passwd')")

    def test_uppercase_schema_qualified_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("SELECT * FROM MAIN.READ_PARQUET('/etc/passwd')")


# ---------------------------------------------------------------------------
# Mutations hidden in CTEs / subqueries
# ---------------------------------------------------------------------------


class TestNestedMutations:
    """Nested INSERT / DROP nodes are detected by walking the tree."""

    def test_insert_inside_cte_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("WITH t AS (SELECT 1) INSERT INTO x SELECT * FROM t")

    def test_truncate_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("TRUNCATE runs")


# ---------------------------------------------------------------------------
# DuckDB-specific dangerous statements that fall through to ``Command``
# ---------------------------------------------------------------------------


class TestUnparsedCommands:
    """Statements sqlglot represents as the catch-all ``Command``
    node (because the dialect doesn't model them) MUST be rejected —
    we don't know their semantics, so we cannot trust them.
    """

    def test_load_extension_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("LOAD httpfs")

    def test_vacuum_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("VACUUM")

    def test_call_arbitrary_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("CALL pg_terminate_backend(1)")


# ---------------------------------------------------------------------------
# Forbidden helper-function families (httpfs_, s3_, gcs_, azure_)
# ---------------------------------------------------------------------------


class TestForbiddenFunctionFamilies:
    """Any function whose name starts with httpfs_/s3_/gcs_/azure_ is
    rejected by prefix — we don't enumerate every helper."""

    def test_httpfs_helper_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("SELECT httpfs_set_secret('x')")

    def test_s3_helper_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("SELECT s3_credentials('x')")


# ---------------------------------------------------------------------------
# URI-scheme path literals
# ---------------------------------------------------------------------------


class TestURIPathLiterals:
    """Path-style literals using URI schemes (s3://, https://, …)
    should be rejected by the AST table-walker."""

    def test_s3_uri_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("SELECT * FROM 's3://bucket/file.parquet'")

    def test_https_uri_rejected(self, engine):
        with pytest.raises(ValueError):
            engine._validate("SELECT * FROM 'https://evil.example/data.csv'")


# ---------------------------------------------------------------------------
# Negative regression — legitimate dataset() queries must still pass
# ---------------------------------------------------------------------------


class TestLegitimateQueriesStillPass:
    """Sanity: the new AST layer does not over-reject normal queries."""

    def test_dataset_select_passes(self, engine):
        engine._validate("SELECT * FROM dataset('ok') ORDER BY date")

    def test_dataset_with_cte_passes(self, engine):
        engine._validate("WITH t AS (SELECT * FROM dataset('ok')) SELECT count(*) FROM t")

    def test_dataset_with_join_passes(self, engine):
        engine._validate("SELECT a.x FROM dataset('ok') a JOIN dataset('ok') b ON a.id = b.id")

    def test_aggregations_pass(self, engine):
        engine._validate("SELECT date_trunc('day', ts) AS d, avg(x) FROM dataset('ok') GROUP BY 1 ORDER BY 1")

    def test_window_functions_pass(self, engine):
        engine._validate("SELECT x, lag(x) OVER (ORDER BY ts) AS prev FROM dataset('ok') ORDER BY ts")


# ---------------------------------------------------------------------------
# AST validator unit tests (exercised directly)
# ---------------------------------------------------------------------------


class TestValidateWithSqlglotDirectly:
    """Hit the AST validator in isolation so failures pinpoint the
    AST layer rather than the composite pipeline."""

    def test_unparseable_sql_raises(self):
        from lazybridge.ext.stat_runtime.query import _validate_with_sqlglot

        with pytest.raises(ValueError, match="parse error"):
            _validate_with_sqlglot("SELECT FROM FROM FROM")

    def test_returns_true_when_sqlglot_available(self):
        from lazybridge.ext.stat_runtime.query import _validate_with_sqlglot

        # Allowed query; should return True (sqlglot present in test env).
        assert _validate_with_sqlglot("SELECT 1") is True
