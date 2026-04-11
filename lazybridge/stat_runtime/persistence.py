"""Dual-backend metadata store for the statistical runtime.

Two backends, same API:
  - InMemory (default): fast, process-local, lost on process exit
  - DuckDB: persistent across runs (activated via ``MetaStore(db="path.duckdb")``)

Follows the same dual-backend pattern as ``LazyStore`` in ``lazy_store.py``.

Usage::

    store = MetaStore()                       # in-memory
    store = MetaStore(db="stat_runtime.duckdb")  # DuckDB-backed

    store.save_dataset(meta)
    store.save_run(record)
    store.save_artifact(artifact)
    runs = store.list_runs(dataset_name="equities")
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lazybridge.stat_runtime.schemas import (
    ArtifactRecord,
    DatasetMeta,
    RunRecord,
    RunStatus,
)

_logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# InMemory backend
# ---------------------------------------------------------------------------

class _InMemoryBackend:
    def __init__(self) -> None:
        self._datasets: dict[str, DatasetMeta] = {}
        self._runs: dict[str, RunRecord] = {}
        self._artifacts: list[ArtifactRecord] = []
        self._lock = threading.Lock()

    # -- datasets --
    def save_dataset(self, meta: DatasetMeta) -> None:
        with self._lock:
            self._datasets[meta.name] = meta

    def get_dataset(self, name: str) -> DatasetMeta | None:
        with self._lock:
            return self._datasets.get(name)

    def list_datasets(self) -> list[DatasetMeta]:
        with self._lock:
            return list(self._datasets.values())

    def delete_dataset(self, name: str) -> None:
        with self._lock:
            self._datasets.pop(name, None)

    # -- runs --
    def save_run(self, record: RunRecord) -> None:
        with self._lock:
            self._runs[record.run_id] = record

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(
        self,
        dataset_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        with self._lock:
            runs = list(self._runs.values())
        if dataset_name:
            runs = [r for r in runs if r.dataset_name == dataset_name]
        if status:
            runs = [r for r in runs if r.status == status]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    # -- artifacts --
    def save_artifact(self, record: ArtifactRecord) -> None:
        with self._lock:
            self._artifacts.append(record)

    def list_artifacts(
        self,
        run_id: str | None = None,
        artifact_type: str | None = None,
    ) -> list[ArtifactRecord]:
        with self._lock:
            arts = list(self._artifacts)
        if run_id:
            arts = [a for a in arts if a.run_id == run_id]
        if artifact_type:
            arts = [a for a in arts if a.artifact_type == artifact_type]
        return arts

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------

class _DuckDBBackend:
    _DDL = """
    CREATE TABLE IF NOT EXISTS _meta (
        key   VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    );

    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id     VARCHAR PRIMARY KEY,
        name           VARCHAR UNIQUE NOT NULL,
        version        VARCHAR DEFAULT '1',
        uri            VARCHAR NOT NULL,
        file_format    VARCHAR DEFAULT 'parquet',
        schema_json    JSON,
        frequency      VARCHAR DEFAULT 'daily',
        time_column    VARCHAR,
        entity_keys    JSON,
        semantic_roles JSON,
        profile_json   JSON,
        row_count      BIGINT,
        registered_at  TIMESTAMP DEFAULT current_timestamp
    );

    CREATE TABLE IF NOT EXISTS runs (
        run_id           VARCHAR PRIMARY KEY,
        dataset_name     VARCHAR DEFAULT '',
        query_hash       VARCHAR DEFAULT '',
        spec_json        JSON,
        status           VARCHAR DEFAULT 'pending',
        fit_summary      VARCHAR DEFAULT '',
        params_json      JSON,
        metrics_json     JSON,
        diagnostics_json JSON,
        artifact_paths   JSON,
        engine           VARCHAR DEFAULT '',
        created_at       TIMESTAMP DEFAULT current_timestamp,
        duration_secs    DOUBLE,
        error_message    VARCHAR
    );

    CREATE TABLE IF NOT EXISTS artifacts (
        run_id        VARCHAR NOT NULL,
        name          VARCHAR NOT NULL,
        artifact_type VARCHAR NOT NULL,
        file_format   VARCHAR NOT NULL,
        path          VARCHAR NOT NULL,
        description   VARCHAR DEFAULT '',
        created_at    TIMESTAMP DEFAULT current_timestamp,
        metadata_json JSON,
        PRIMARY KEY (run_id, name)
    );
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._init_db()

    def _conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            duckdb = _import_duckdb()
            self._local.conn = duckdb.connect(self._db_path)
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.execute(self._DDL)
        # Check / set schema version
        existing = conn.execute(
            "SELECT value FROM _meta WHERE key = 'schema_version'"
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO _meta (key, value) VALUES ('schema_version', ?)",
                [str(_SCHEMA_VERSION)],
            )
        else:
            ver = int(existing[0])
            if ver != _SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {ver} is not supported by this version "
                    f"of stat_runtime (expects {_SCHEMA_VERSION}). "
                    "Delete the database file and re-register your datasets."
                )

    # -- datasets --
    def save_dataset(self, meta: DatasetMeta) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO datasets
               (dataset_id, name, version, uri, file_format, schema_json,
                frequency, time_column, entity_keys, semantic_roles,
                profile_json, row_count, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                meta.dataset_id, meta.name, meta.version, meta.uri,
                meta.file_format, json.dumps(meta.columns_schema),
                str(meta.frequency), meta.time_column,
                json.dumps(meta.entity_keys), json.dumps(meta.semantic_roles),
                json.dumps(meta.profile_json), meta.row_count,
                meta.registered_at.isoformat(),
            ],
        )

    def get_dataset(self, name: str) -> DatasetMeta | None:
        row = self._conn().execute(
            "SELECT * FROM datasets WHERE name = ?", [name]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dataset(row)

    def list_datasets(self) -> list[DatasetMeta]:
        rows = self._conn().execute(
            "SELECT * FROM datasets ORDER BY registered_at DESC"
        ).fetchall()
        return [self._row_to_dataset(r) for r in rows]

    def delete_dataset(self, name: str) -> None:
        self._conn().execute("DELETE FROM datasets WHERE name = ?", [name])

    @staticmethod
    def _row_to_dataset(row: tuple) -> DatasetMeta:
        return DatasetMeta(
            dataset_id=row[0], name=row[1], version=row[2], uri=row[3],
            file_format=row[4],
            columns_schema=json.loads(row[5]) if row[5] else {},
            frequency=row[6], time_column=row[7],
            entity_keys=json.loads(row[8]) if row[8] else [],
            semantic_roles=json.loads(row[9]) if row[9] else {},
            profile_json=json.loads(row[10]) if row[10] else {},
            row_count=row[11],
            registered_at=datetime.fromisoformat(row[12]) if row[12] else datetime.now(UTC),
        )

    # -- runs --
    def save_run(self, record: RunRecord) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, dataset_name, query_hash, spec_json, status,
                fit_summary, params_json, metrics_json, diagnostics_json,
                artifact_paths, engine, created_at, duration_secs, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                record.run_id, record.dataset_name, record.query_hash,
                json.dumps(record.spec_json), str(record.status),
                record.fit_summary, json.dumps(record.params_json),
                json.dumps(record.metrics_json), json.dumps(record.diagnostics_json),
                json.dumps(record.artifact_paths), record.engine,
                record.created_at.isoformat(), record.duration_secs,
                record.error_message,
            ],
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn().execute(
            "SELECT * FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_runs(
        self,
        dataset_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        clauses = []
        params: list[Any] = []
        if dataset_name:
            clauses.append("dataset_name = ?")
            params.append(dataset_name)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._conn().execute(
            f"SELECT * FROM runs{where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    @staticmethod
    def _row_to_run(row: tuple) -> RunRecord:
        return RunRecord(
            run_id=row[0], dataset_name=row[1], query_hash=row[2],
            spec_json=json.loads(row[3]) if row[3] else {},
            status=RunStatus(row[4]) if row[4] else RunStatus.PENDING,
            fit_summary=row[5] or "",
            params_json=json.loads(row[6]) if row[6] else {},
            metrics_json=json.loads(row[7]) if row[7] else {},
            diagnostics_json=json.loads(row[8]) if row[8] else [],
            artifact_paths=json.loads(row[9]) if row[9] else [],
            engine=row[10] or "",
            created_at=datetime.fromisoformat(row[11]) if row[11] else datetime.now(UTC),
            duration_secs=row[12],
            error_message=row[13],
        )

    # -- artifacts --
    def save_artifact(self, record: ArtifactRecord) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO artifacts
               (run_id, name, artifact_type, file_format, path,
                description, created_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                record.run_id, record.name, record.artifact_type,
                record.file_format, record.path, record.description,
                record.created_at.isoformat(), json.dumps(record.metadata),
            ],
        )

    def list_artifacts(
        self,
        run_id: str | None = None,
        artifact_type: str | None = None,
    ) -> list[ArtifactRecord]:
        clauses = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if artifact_type:
            clauses.append("artifact_type = ?")
            params.append(artifact_type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn().execute(
            f"SELECT * FROM artifacts{where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        return [
            ArtifactRecord(
                run_id=r[0], name=r[1], artifact_type=r[2], file_format=r[3],
                path=r[4], description=r[5] or "",
                created_at=datetime.fromisoformat(r[6]) if r[6] else datetime.now(UTC),
                metadata=json.loads(r[7]) if r[7] else {},
            )
            for r in rows
        ]

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None


def _import_duckdb():
    """Lazy import of duckdb — only called when DuckDB backend is used."""
    from lazybridge.stat_runtime._deps import require_duckdb
    return require_duckdb()


# ---------------------------------------------------------------------------
# MetaStore (public facade)
# ---------------------------------------------------------------------------

class MetaStore:
    """Metadata store for the statistical runtime.

    Usage::

        store = MetaStore()                          # in-memory
        store = MetaStore(db="stat_runtime.duckdb")  # DuckDB-backed

        with MetaStore(db="my.duckdb") as store:
            store.save_dataset(meta)
    """

    def __init__(self, db: str | None = None) -> None:
        self._backend: _InMemoryBackend | _DuckDBBackend = (
            _DuckDBBackend(db) if db else _InMemoryBackend()
        )

    # -- datasets --
    def save_dataset(self, meta: DatasetMeta) -> None:
        self._backend.save_dataset(meta)

    def get_dataset(self, name: str) -> DatasetMeta | None:
        return self._backend.get_dataset(name)

    def list_datasets(self) -> list[DatasetMeta]:
        return self._backend.list_datasets()

    def delete_dataset(self, name: str) -> None:
        self._backend.delete_dataset(name)

    # -- runs --
    def save_run(self, record: RunRecord) -> None:
        self._backend.save_run(record)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._backend.get_run(run_id)

    def list_runs(
        self,
        dataset_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        return self._backend.list_runs(dataset_name=dataset_name, status=status, limit=limit)

    # -- artifacts --
    def save_artifact(self, record: ArtifactRecord) -> None:
        self._backend.save_artifact(record)

    def list_artifacts(
        self,
        run_id: str | None = None,
        artifact_type: str | None = None,
    ) -> list[ArtifactRecord]:
        return self._backend.list_artifacts(run_id=run_id, artifact_type=artifact_type)

    # -- lifecycle --
    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> MetaStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        kind = "DuckDB" if isinstance(self._backend, _DuckDBBackend) else "InMemory"
        return f"MetaStore(backend={kind})"
