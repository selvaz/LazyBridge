"""Dataset registration, schema introspection, and profiling.

All data enters the runtime through this catalog.  Datasets must be registered
before any model tool can use them.

Usage::

    catalog = DatasetCatalog(meta_store)
    meta = catalog.register_parquet("equities", "/data/returns.parquet",
                                     time_column="date", entity_keys=["symbol"])
    profile = catalog.profile("equities")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lazybridge.ext.stat_runtime.persistence import MetaStore
from lazybridge.ext.stat_runtime.schemas import (
    ColumnProfile,
    DatasetMeta,
    Frequency,
    ProfileResult,
)

_logger = logging.getLogger(__name__)


class DatasetCatalog:
    """Registry for datasets used by the statistical runtime."""

    def __init__(self, meta_store: MetaStore) -> None:
        self._store = meta_store

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_parquet(
        self,
        name: str,
        uri: str,
        *,
        frequency: Frequency | str = Frequency.DAILY,
        time_column: str | None = None,
        entity_keys: list[str] | None = None,
        semantic_roles: dict[str, str] | None = None,
        business_description: str | None = None,
        canonical_target: str | None = None,
        identifiers_to_ignore: list[str] | None = None,
        version: str = "1",
    ) -> DatasetMeta:
        """Register a Parquet file or directory as a named dataset.

        Auto-detects schema and row count via Polars lazy scan.
        """
        pl = _import_polars()
        path = Path(uri).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Parquet path does not exist: {path}")

        # Lazy scan — reads only metadata, not the full file
        lf = pl.scan_parquet(str(path))
        schema = {col: str(dtype) for col, dtype in lf.collect_schema().items()}
        row_count = lf.select(pl.len()).collect().item()

        self._validate_columns(schema, time_column, canonical_target, identifiers_to_ignore, semantic_roles)

        freq = Frequency(frequency) if isinstance(frequency, str) else frequency
        meta = DatasetMeta(
            name=name,
            version=version,
            uri=str(path),
            file_format="parquet",
            columns_schema=schema,
            frequency=freq,
            time_column=time_column,
            entity_keys=entity_keys or [],
            semantic_roles=semantic_roles or {},
            business_description=business_description,
            canonical_target=canonical_target,
            identifiers_to_ignore=identifiers_to_ignore or [],
            row_count=row_count,
        )
        self._store.save_dataset(meta)
        _logger.info("Registered dataset '%s' (%d rows, %d cols)", name, row_count, len(schema))
        return meta

    def register_csv(
        self,
        name: str,
        uri: str,
        *,
        frequency: Frequency | str = Frequency.DAILY,
        time_column: str | None = None,
        entity_keys: list[str] | None = None,
        semantic_roles: dict[str, str] | None = None,
        business_description: str | None = None,
        canonical_target: str | None = None,
        identifiers_to_ignore: list[str] | None = None,
        version: str = "1",
    ) -> DatasetMeta:
        """Register a CSV file as a named dataset."""
        pl = _import_polars()
        path = Path(uri).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV path does not exist: {path}")

        lf = pl.scan_csv(str(path))
        schema = {col: str(dtype) for col, dtype in lf.collect_schema().items()}
        row_count = lf.select(pl.len()).collect().item()

        self._validate_columns(schema, time_column, canonical_target, identifiers_to_ignore, semantic_roles)

        freq = Frequency(frequency) if isinstance(frequency, str) else frequency
        meta = DatasetMeta(
            name=name,
            version=version,
            uri=str(path),
            file_format="csv",
            columns_schema=schema,
            frequency=freq,
            time_column=time_column,
            entity_keys=entity_keys or [],
            semantic_roles=semantic_roles or {},
            business_description=business_description,
            canonical_target=canonical_target,
            identifiers_to_ignore=identifiers_to_ignore or [],
            row_count=row_count,
        )
        self._store.save_dataset(meta)
        _logger.info("Registered CSV dataset '%s' (%d rows, %d cols)", name, row_count, len(schema))
        return meta

    @staticmethod
    def _validate_columns(
        schema: dict[str, str],
        time_column: str | None,
        canonical_target: str | None,
        identifiers_to_ignore: list[str] | None,
        semantic_roles: dict[str, str] | None,
    ) -> None:
        """Validate that referenced columns exist in the schema."""
        cols = list(schema.keys())
        if time_column and time_column not in schema:
            raise ValueError(f"time_column '{time_column}' not found in schema. Available columns: {cols}")
        if canonical_target and canonical_target not in schema:
            raise ValueError(f"canonical_target '{canonical_target}' not found in schema. Available columns: {cols}")
        if identifiers_to_ignore:
            bad = [c for c in identifiers_to_ignore if c not in schema]
            if bad:
                raise ValueError(f"identifiers_to_ignore contains unknown columns: {bad}. Available columns: {cols}")
        if semantic_roles:
            bad = [c for c in semantic_roles if c not in schema]
            if bad:
                raise ValueError(f"semantic_roles references unknown columns: {bad}. Available columns: {cols}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> DatasetMeta | None:
        """Get metadata for a registered dataset."""
        return self._store.get_dataset(name)

    def list_datasets(self) -> list[DatasetMeta]:
        """List all registered datasets."""
        return self._store.list_datasets()

    def deregister(self, name: str) -> None:
        """Remove a dataset from the catalog."""
        self._store.delete_dataset(name)
        _logger.info("Deregistered dataset '%s'", name)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(self, name: str) -> ProfileResult:
        """Compute column-level statistics for a registered dataset."""
        pl = _import_polars()
        meta = self._store.get_dataset(name)
        if meta is None:
            raise ValueError(f"Dataset '{name}' is not registered")

        df = self._load_df(meta, pl)
        columns: dict[str, ColumnProfile] = {}

        for col in df.columns:
            series = df[col]
            null_count = series.null_count()
            total = len(series)
            profile = ColumnProfile(
                dtype=str(series.dtype),
                null_count=null_count,
                null_pct=round(null_count / total * 100, 2) if total > 0 else 0.0,
            )
            # Numeric stats
            if series.dtype.is_numeric():
                non_null = series.drop_nulls()
                if len(non_null) > 0:
                    profile.min_val = non_null.min()
                    profile.max_val = non_null.max()
                    profile.mean = round(float(non_null.mean()), 6)
                    profile.std = round(float(non_null.std()), 6)
                    profile.unique_count = non_null.n_unique()
            elif series.dtype == pl.Utf8 or series.dtype == pl.String:
                profile.unique_count = series.n_unique()

            columns[col] = profile

        result = ProfileResult(
            dataset_name=name,
            row_count=len(df),
            columns=columns,
        )
        # Cache profile in metadata
        meta.profile_json = result.model_dump()
        self._store.save_dataset(meta)
        return result

    # ------------------------------------------------------------------
    # Data loading (internal)
    # ------------------------------------------------------------------

    def load_df(self, name: str):
        """Load a registered dataset as a Polars DataFrame."""
        pl = _import_polars()
        meta = self._store.get_dataset(name)
        if meta is None:
            raise ValueError(f"Dataset '{name}' is not registered")
        return self._load_df(meta, pl)

    @staticmethod
    def _load_df(meta: DatasetMeta, pl: Any):
        """Load dataset based on its format."""
        if meta.file_format == "csv":
            return pl.read_csv(meta.uri)
        return pl.read_parquet(meta.uri)


def _import_polars():
    from lazybridge.ext.stat_runtime._deps import require_polars

    return require_polars()
