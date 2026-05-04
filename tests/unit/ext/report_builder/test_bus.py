"""Unit tests for FragmentBus — append, idempotency, persistence, threading."""

from __future__ import annotations

import threading

import pytest

from lazybridge.ext.report_builder import FragmentBus
from lazybridge.ext.report_builder.bus import store_key_for
from lazybridge.ext.report_builder.fragments import Fragment
from lazybridge.store import Store


class TestBasicAppend:
    def test_append_returns_id(self):
        bus = FragmentBus("r1")
        fid = bus.append(Fragment(kind="text", body_md="hi"))
        assert fid
        assert isinstance(fid, str)

    def test_count(self):
        bus = FragmentBus("r2")
        bus.append(Fragment(kind="text", body_md="a"))
        bus.append(Fragment(kind="text", body_md="b"))
        assert len(bus) == 2

    def test_clear(self):
        bus = FragmentBus("r3")
        bus.append(Fragment(kind="text", body_md="x"))
        assert len(bus) == 1
        bus.clear()
        assert len(bus) == 0


class TestOrdering:
    def test_fragments_returned_in_created_order(self):
        bus = FragmentBus("r4")
        a = bus.append(Fragment(kind="text", body_md="a"))
        b = bus.append(Fragment(kind="text", body_md="b"))
        c = bus.append(Fragment(kind="text", body_md="c"))
        ids = [f.id for f in bus.fragments()]
        assert ids == [a, b, c]

    def test_by_section_filters(self):
        bus = FragmentBus("r5")
        bus.append(Fragment(kind="text", body_md="a", section="intro"))
        bus.append(Fragment(kind="text", body_md="b", section="outro"))
        bus.append(Fragment(kind="text", body_md="c", section="intro"))
        intro = bus.by_section("intro")
        assert len(intro) == 2
        assert all(f.section == "intro" for f in intro)


class TestIdempotency:
    def test_dup_fragment_id_skipped(self):
        bus = FragmentBus("r6")
        f = Fragment(kind="text", body_md="once")
        bus.append(f)
        # Re-appending the same Fragment id should not double-write — this
        # is the resume-after-crash scenario where Plan replays a Step.
        result_id = bus.append(f)
        assert result_id == f.id
        assert len(bus) == 1


class TestPersistence:
    def test_sqlite_roundtrip_via_shared_store(self, tmp_path):
        db = str(tmp_path / "store.db")
        store = Store(db=db)
        try:
            bus_a = FragmentBus("rep", store=store)
            bus_a.append(Fragment(kind="text", body_md="alpha"))
            bus_a.append(Fragment(kind="text", body_md="beta"))

            # New bus pointed at the same store + report_id reads back the
            # full fragment list — the resume-after-crash use case.
            bus_b = FragmentBus("rep", store=store)
            assert len(bus_b) == 2
            bodies = {f.body_md for f in bus_b.fragments()}
            assert bodies == {"alpha", "beta"}
        finally:
            store.close()

    def test_store_key_for_helper(self):
        assert store_key_for("xyz").endswith(":xyz")


class TestThreadSafety:
    def test_concurrent_appends_preserve_all_writes(self):
        bus = FragmentBus("threaded")
        N = 50
        errors: list[Exception] = []

        def worker(i: int):
            try:
                bus.append(Fragment(kind="text", body_md=f"frag-{i}"))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        assert len(bus) == N
        bodies = {f.body_md for f in bus.fragments()}
        # Every distinct body should be present — no dupes, no losses.
        assert bodies == {f"frag-{i}" for i in range(N)}


class TestAssembleConvenience:
    def test_assemble_uses_default_blackboard(self):
        bus = FragmentBus("plain")
        bus.append(Fragment(kind="text", body_md="x", section="a"))
        bus.append(Fragment(kind="text", body_md="y", section="b"))
        report = bus.assemble(title="Title")
        assert report.title == "Title"
        assert report.metadata["assembler"] == "blackboard"
        assert {s.heading for s in report.sections} == {"a", "b"}


class TestRepr:
    def test_repr_includes_count_and_assembler(self):
        bus = FragmentBus("r")
        bus.append(Fragment(kind="text", body_md="x"))
        s = repr(bus)
        assert "fragments=1" in s
        assert "BlackboardAssembler" in s
