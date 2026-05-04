"""FragmentBus — the thread-safe, resume-aware fragment collector.

A :class:`FragmentBus` is the runtime side of the parallel-assembly pipeline.
Steps in a Plan share one Bus and call :meth:`append` (directly from Python
or via :func:`fragment_tools` in an LLM agent's tools list).  Concurrent
``parallel=True`` Steps in the same band write through the bus
simultaneously; the bus serialises them via :meth:`Store.compare_and_swap`.

When the pipeline ends — typically a final Step that calls
:meth:`bus.export` — the configured :class:`Assembler` reduces the fragment
list to a structured :class:`AssembledReport` and the chosen
:class:`Exporter` writes it out.

State layout
------------

The full fragment list is kept under a single Store key,
``__report_fragments__:{report_id}``, as a JSON-serialised list of
:meth:`Fragment.model_dump` dicts.  Single-key + CAS keeps the contention
window narrow without needing a separate lock service; on resume the same
key reappears so already-emitted fragments survive a crash.

We deliberately do NOT key by fragment-id under different rows because:

1. ``Store`` doesn't expose an enumeration API guaranteed to be stable
   under concurrent writes;
2. typical fragment volumes are O(10s–100s) per report, which fits in
   memory comfortably and keeps the assembler's input contiguous.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lazybridge.ext.report_builder.fragments import Fragment
from lazybridge.store import Store

if TYPE_CHECKING:
    from lazybridge.ext.report_builder.assemblers import AssembledReport, Assembler


_STORE_KEY_PREFIX = "__report_fragments__"


class FragmentBus:
    """Thread-safe, resume-aware collection of report fragments.

    Backed by :class:`lazybridge.store.Store` for persistence — defaults to
    an in-memory Store so the user doesn't need SQLite for ad-hoc pipelines.
    Pass an existing ``Store`` (typically the same one a ``Plan`` is using
    for checkpointing) to share state with the rest of the pipeline.

    Usage::

        bus = FragmentBus("daily-news")
        bus.append(Fragment(kind="text", heading="Hello", body_md="World"))
        report = bus.export(["html"], "./out", title="Daily News")
    """

    def __init__(
        self,
        report_id: str,
        *,
        store: Store | None = None,
        assembler: Assembler | None = None,
    ) -> None:
        self.report_id = report_id
        self._store = store if store is not None else Store(db=None)
        self._owns_store = store is None  # for cleanup decisions
        # Default assembler is the free-form Blackboard one — works for any
        # pipeline shape without configuration.  Outline-shaped pipelines
        # pass an OutlineAssembler explicitly.
        if assembler is None:
            from lazybridge.ext.report_builder.assemblers import BlackboardAssembler

            assembler = BlackboardAssembler()
        self._assembler = assembler
        # Local lock guards the read-modify-write loop that wraps the
        # Store CAS — without it, two Python-side appends on the *same*
        # thread would each spin retrying the CAS in lockstep.  CAS still
        # protects against cross-thread contention; the local lock is a
        # cheap tail-end fast path.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Storage primitives
    # ------------------------------------------------------------------

    @property
    def _key(self) -> str:
        return f"{_STORE_KEY_PREFIX}:{self.report_id}"

    def _read_raw(self) -> list[dict]:
        """Read the stored list, normalising 'never written' to ``[]``.

        Used by the public API.  Internal CAS callers use :meth:`_read_for_cas`
        which preserves the ``None`` vs ``[]`` distinction the Store cares about.
        """
        raw = self._store.read(self._key, default=None)
        if raw is None:
            return []
        return list(raw)

    def _read_for_cas(self) -> list[dict] | None:
        """Read for compare-and-swap — returns ``None`` when the key is unset.

        :meth:`Store.compare_and_swap` distinguishes "missing key" from "key
        exists with empty list".  ``_read_raw`` collapses both to ``[]`` for
        callers who only care about contents; CAS callers must be precise.
        """
        raw = self._store.read(self._key, default=None)
        if raw is None:
            return None
        return list(raw)

    # ------------------------------------------------------------------
    # Public API — Python side
    # ------------------------------------------------------------------

    def append(self, fragment: Fragment) -> str:
        """Append a fragment and return its id.

        Concurrent appends from sibling parallel steps serialise on the
        Store's compare-and-swap.  If we lose a CAS race we re-read and
        retry — bounded by ``max_retries`` so a misbehaving Store can't
        deadlock the pipeline.
        """
        max_retries = 32
        with self._lock:
            for _ in range(max_retries):
                # Use the CAS-precise read so a never-written key compares as
                # ``None`` (Store.compare_and_swap requires this distinction).
                current = self._read_for_cas()
                view = current if current is not None else []
                # Idempotency: if the same fragment id is already present
                # (e.g. resumed-pipeline replay), skip the append silently
                # and return the existing id.
                if any(item.get("id") == fragment.id for item in view):
                    return fragment.id
                new = view + [fragment.model_dump(mode="json")]
                if self._store.compare_and_swap(self._key, expected=current, new=new):
                    return fragment.id
            raise RuntimeError(
                f"FragmentBus.append: lost {max_retries} CAS races for report_id={self.report_id!r}; "
                "either the Store is misbehaving or contention is pathological."
            )

    def fragments(self) -> list[Fragment]:
        """Return all fragments currently in the bus, ordered by ``created_at``."""
        items = [Fragment.model_validate(d) for d in self._read_raw()]
        items.sort(key=lambda f: f.created_at)
        return items

    def by_section(self, section: str) -> list[Fragment]:
        """Return fragments matching ``section`` — convenience for synthesis steps."""
        return [f for f in self.fragments() if f.section == section]

    def clear(self) -> None:
        """Remove all fragments for this report.  Mostly useful for tests."""
        with self._lock:
            self._store.delete(self._key)

    def __len__(self) -> int:
        return len(self._read_raw())

    # ------------------------------------------------------------------
    # Convenience export
    # ------------------------------------------------------------------

    def assemble(self, *, title: str = "Report") -> AssembledReport:
        """Run the configured assembler and return the structured report."""
        return self._assembler.assemble(self.fragments(), title=title)

    def export(
        self,
        formats: Sequence[Literal["html", "pdf", "docx", "revealjs"]],
        output_dir: str | Path,
        *,
        title: str,
        theme: str = "cosmo",
        backend: Literal["quarto", "weasyprint", "auto"] = "auto",
        assembler: Assembler | None = None,
        author: str | None = None,
    ) -> dict[str, Path]:
        """Render and write out the report in every requested format.

        ``backend="auto"`` (the default) picks ``quarto`` if the CLI is on
        ``$PATH``, otherwise falls back to ``weasyprint``.  Pass an explicit
        backend to disable auto-detection.

        Returns a mapping ``{format: output_path}`` for every successfully
        produced artifact.
        """
        from lazybridge.ext.report_builder.exporters import resolve_exporter

        chosen_assembler = assembler or self._assembler
        report = chosen_assembler.assemble(self.fragments(), title=title)
        if author and report.metadata.get("author") is None:
            report.metadata["author"] = author

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        exporter = resolve_exporter(backend=backend, formats=list(formats))
        return exporter.export(report, output_dir, formats=list(formats), theme=theme)

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FragmentBus(report_id={self.report_id!r}, "
            f"fragments={len(self)}, assembler={type(self._assembler).__name__})"
        )

    def to_jsonl(self) -> str:
        """Serialise current fragments as JSON Lines — handy for debugging."""
        return "\n".join(json.dumps(d, default=str) for d in self._read_raw())


def store_key_for(report_id: str) -> str:
    """Return the Store key under which fragments for ``report_id`` live.

    Exposed so users who want to introspect or hand-edit the Store can do so
    without depending on the private constant.
    """
    return f"{_STORE_KEY_PREFIX}:{report_id}"
