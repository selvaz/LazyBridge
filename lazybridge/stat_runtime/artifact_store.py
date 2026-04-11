"""Filesystem-based artifact storage for the statistical runtime.

Artifacts (plots, data exports, summaries) are stored on disk organized by
run_id.  Each artifact is also registered in MetaStore for catalog queries.

Directory layout::

    {root}/
        {run_id}/
            plots/
                residuals.png
                acf_pacf.png
                volatility.png
            data/
                residuals.parquet
                fitted_values.parquet
            summaries/
                spec.json
                fit_summary.json
                diagnostics.json

Usage::

    store = ArtifactStore("artifacts", meta_store=meta)
    path = store.write_json(run_id, "spec", data_dict, artifact_type="summary")
    path = store.write_plot(run_id, "residuals", fig)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lazybridge.stat_runtime.schemas import ArtifactRecord

_logger = logging.getLogger(__name__)

# Subdirectory by artifact type
_TYPE_DIRS = {
    "plot": "plots",
    "data": "data",
    "summary": "summaries",
    "forecast": "data",
}


class ArtifactStore:
    """Filesystem artifact manager with optional MetaStore registration."""

    def __init__(
        self,
        root: str = "artifacts",
        meta_store: Any = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._meta_store = meta_store

    @property
    def root(self) -> Path:
        return self._root

    def path_for(
        self, run_id: str, name: str, artifact_type: str = "data", ext: str = ""
    ) -> Path:
        """Return the target path for an artifact (creates parent dirs)."""
        subdir = _TYPE_DIRS.get(artifact_type, artifact_type)
        fname = f"{name}{ext}" if ext else name
        path = self._root / run_id / subdir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def write_bytes(
        self,
        run_id: str,
        name: str,
        data: bytes,
        *,
        artifact_type: str = "data",
        file_format: str = "bin",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write raw bytes and register the artifact.  Returns the file path."""
        ext = f".{file_format}" if not name.endswith(f".{file_format}") else ""
        path = self.path_for(run_id, name, artifact_type, ext)
        path.write_bytes(data)
        self._register(run_id, name, artifact_type, file_format, str(path),
                        description, metadata)
        return str(path)

    def write_text(
        self,
        run_id: str,
        name: str,
        text: str,
        *,
        artifact_type: str = "summary",
        file_format: str = "txt",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write text and register.  Returns the file path."""
        ext = f".{file_format}" if not name.endswith(f".{file_format}") else ""
        path = self.path_for(run_id, name, artifact_type, ext)
        path.write_text(text, encoding="utf-8")
        self._register(run_id, name, artifact_type, file_format, str(path),
                        description, metadata)
        return str(path)

    def write_json(
        self,
        run_id: str,
        name: str,
        data: Any,
        *,
        artifact_type: str = "summary",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Serialize data as JSON and register.  Returns the file path."""
        path = self.path_for(run_id, name, artifact_type, ".json")
        path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        self._register(run_id, name, artifact_type, "json", str(path),
                        description, metadata)
        return str(path)

    def write_plot(
        self,
        run_id: str,
        name: str,
        fig: Any,
        *,
        file_format: str = "png",
        dpi: int = 150,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a matplotlib figure and register.  Returns the file path.

        Closes the figure after saving to free memory.
        """
        ext = f".{file_format}" if not name.endswith(f".{file_format}") else ""
        path = self.path_for(run_id, name, "plot", ext)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        # Close figure to free memory
        try:
            plt = fig.get_figure() if hasattr(fig, "get_figure") else fig
            import matplotlib.pyplot as _plt
            _plt.close(plt)
        except Exception:
            pass
        self._register(run_id, name, "plot", file_format, str(path),
                        description, metadata)
        return str(path)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def read_bytes(self, run_id: str, name: str, artifact_type: str = "data") -> bytes:
        """Read raw bytes for an artifact."""
        path = self._find_path(run_id, name, artifact_type)
        return path.read_bytes()

    def read_json(self, run_id: str, name: str, artifact_type: str = "summary") -> Any:
        """Read and parse a JSON artifact."""
        path = self._find_path(run_id, name, artifact_type)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_files(self, run_id: str) -> list[str]:
        """List all artifact files for a run."""
        run_dir = self._root / run_id
        if not run_dir.exists():
            return []
        return sorted(str(p) for p in run_dir.rglob("*") if p.is_file())

    def exists(self, run_id: str, name: str, artifact_type: str = "data") -> bool:
        """Check if an artifact file exists."""
        try:
            self._find_path(run_id, name, artifact_type)
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_path(self, run_id: str, name: str, artifact_type: str) -> Path:
        """Locate an artifact file, trying with and without common extensions."""
        subdir = _TYPE_DIRS.get(artifact_type, artifact_type)
        base_dir = self._root / run_id / subdir
        # Try exact name
        exact = base_dir / name
        if exact.exists():
            return exact
        # Try with common extensions
        for ext in (".json", ".png", ".svg", ".csv", ".parquet", ".txt"):
            candidate = base_dir / f"{name}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Artifact '{name}' (type={artifact_type}) not found for run {run_id} "
            f"in {base_dir}"
        )

    def _register(
        self,
        run_id: str,
        name: str,
        artifact_type: str,
        file_format: str,
        path: str,
        description: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Register an artifact in MetaStore if available."""
        if self._meta_store is None:
            return
        record = ArtifactRecord(
            run_id=run_id,
            name=name,
            artifact_type=artifact_type,
            file_format=file_format,
            path=path,
            description=description,
            created_at=datetime.now(UTC),
            metadata=metadata or {},
        )
        try:
            self._meta_store.save_artifact(record)
        except Exception as exc:
            _logger.warning("Failed to register artifact %s: %s", name, exc)
