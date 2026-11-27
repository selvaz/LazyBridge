"""Versioned benchmark contracts; never call subscription models."""

import json

import pytest

from examples.cross_review_benchmark import harness


def test_versioned_inventory_and_freeze():
    from examples.cross_review_benchmark.suites import load_suite, task_hash

    v1, v2 = load_suite("v1"), load_suite("v2")
    assert len(v1.tasks) == 25
    assert len(v2.tasks) == 45
    assert sum(t.category == "coding" for t in v2.tasks) == 20
    assert sum(t.category == "api" for t in v2.tasks) == 25
    assert len({t.id for t in v2.tasks}) == 45
    assert task_hash("v1") == json.loads((v1.directory / "freeze.json").read_text())["task_sha256"]
    assert v1.tasks == list(harness.TASKS)
    assert all(v1.test_source(t) == harness.test_source(t) for t in v1.tasks)
    assert [t.spec for t in v1.tasks if t.category == "api"] == [t.spec for t in v2.tasks if t.category == "api"][:5]
    assert task_hash("v2") == json.loads((v2.directory / "freeze.json").read_text())["task_sha256"]


def test_v2_references_and_wrong_implementations(tmp_path):
    from examples.cross_review_benchmark.suites import load_suite

    suite = load_suite("v2")
    for task in suite.tasks:
        result = harness.grade(task, suite.solution(task), tmp_path, 20, suite=suite)
        assert result["passed"], f"{task.id}: {result['output']}"
    for category in ("coding", "api"):
        task = next(t for t in suite.tasks if t.category == category)
        assert not harness.grade(task, "raise RuntimeError('incorrect')", tmp_path, 20, suite=suite)["passed"]


@pytest.mark.asyncio
async def test_v2_mock_pipeline_and_version_resume(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("No live engine permitted")

    monkeypatch.setattr(harness, "live_engine", forbidden)
    path = tmp_path / "v2.jsonl"
    args = harness.parse_args(["--suite", "v2", "--mock", "--output", str(path)])
    rows = await harness.run(args)
    assert len(rows) == 180
    assert all(r["suite"] == "v2" and r["task_sha256"] for r in rows)
    assert harness.paired(rows, "A", "C", "coding")["fixed"]
    assert harness.paired(rows, "A", "C", "coding")["broken"]
    before = path.read_bytes()
    assert await harness.run(args) == rows
    assert path.read_bytes() == before
    args.suite = "v1"
    with pytest.raises(ValueError, match="incompatible"):
        await harness.run(args)


def test_changed_frozen_set_rejected(tmp_path):
    from examples.cross_review_benchmark.suites import verify_freeze

    for name in ("tasks.json", "hidden_tests.json", "mock_solutions.json"):
        (tmp_path / name).write_text("[]")
    (tmp_path / "freeze.json").write_text(json.dumps({"task_sha256": "wrong"}))
    with pytest.raises(ValueError, match="frozen"):
        verify_freeze(tmp_path)


@pytest.mark.asyncio
async def test_calibration_never_runs_review_arms(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("Must reject before constructing any engine")

    monkeypatch.setattr(harness, "live_engine", forbidden)
    args = harness.parse_args(
        ["--suite", "v2", "--phase", "calibration", "--arms", "C", "--output", str(tmp_path / "x.jsonl")]
    )
    with pytest.raises(ValueError, match="arm A only"):
        await harness.run(args)


def test_freeze_requires_complete_calibration_and_is_write_once(tmp_path, monkeypatch):
    import shutil
    import zipfile

    from examples.cross_review_benchmark import freeze
    from examples.cross_review_benchmark.suites import TASK_FILES, directory_hash, load_suite

    suite = load_suite("v2")
    for name in TASK_FILES:
        shutil.copyfile(suite.directory / name, tmp_path / name)
    suite.directory = tmp_path
    monkeypatch.setattr(freeze, "load_suite", lambda name: suite)
    monkeypatch.setattr(freeze, "HERE", tmp_path)
    path = tmp_path / "calibration.jsonl"
    rows = []
    counts = {"coding": 0, "api": 0}
    for task in suite.tasks:
        counts[task.category] += 1
        rows.append(
            {
                "task_id": task.id,
                "arm": "A",
                "category": task.category,
                "mock": False,
                "suite": "v2",
                "phase": "calibration",
                "passed": counts[task.category] <= {"coding": 12, "api": 15}[task.category],
                "protocol_hash": "protocol",
                "task_sha256": directory_hash(tmp_path),
                "started_at": "2026-10-11T00:00:00+00:00",
                "finished_at": "2026-10-11T00:01:00+00:00",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows[:1]) + "\n")
    with pytest.raises(ValueError, match="all 45"):
        freeze.freeze([path])
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    path.with_suffix(".manifest.json").write_text(
        json.dumps({"task_sha256": directory_hash(tmp_path), "protocol_hash": "protocol"})
    )
    with zipfile.ZipFile(path.with_suffix(".tasks.zip"), "w") as archive:
        for name in TASK_FILES:
            archive.write(tmp_path / name, name)
    record = freeze.freeze([path])
    assert record["calibration_runs"][0]["passes"] == {"coding": 12, "api": 15}
    assert record["cross_provider_arms_run"] == []
    with pytest.raises(ValueError, match="already frozen"):
        freeze.freeze([path])
