"""Contract tests written before the benchmark implementation; no provider calls."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from examples.cross_review_benchmark import harness
from examples.cross_review_benchmark.tasks import TASKS


def test_inventory_and_selection():
    assert len(TASKS) == 25
    assert len({t.id for t in TASKS}) == 25
    assert sum(t.category == "coding" for t in TASKS) == 20
    assert sum(t.category == "api" for t in TASKS) == 5
    first = harness.select_tasks("all", 3, 17)
    assert first == harness.select_tasks("all", 3, 17)
    assert len(first) == 3
    assert len(harness.select_tasks("api", None, 17)) == 5
    with pytest.raises(ValueError):
        harness.select_tasks("nonexistent", None, 17)


def test_statistics_and_pairing():
    lo, hi = harness.wilson(0, 20)
    assert lo == pytest.approx(0)
    assert hi == pytest.approx(0.161125, abs=1e-5)
    assert harness.mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert harness.mcnemar_exact(0, 0) == 1
    rows = [
        {"arm": "A", "task_id": "x", "category": "coding", "passed": False},
        {"arm": "C", "task_id": "x", "category": "coding", "passed": True},
        {"arm": "A", "task_id": "y", "category": "coding", "passed": True},
        {"arm": "C", "task_id": "y", "category": "coding", "passed": False},
        {"arm": "C", "task_id": "unpaired", "category": "coding", "passed": True},
    ]
    pair = harness.paired(rows, "A", "C", "coding")
    assert pair == {"n": 2, "fixed": ["x"], "broken": ["y"], "p": 1.0}


@pytest.mark.asyncio
async def test_whole_mock_pipeline_and_resume(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("mock must never construct a live engine")

    monkeypatch.setattr(harness, "live_engine", forbidden)
    path = tmp_path / "mock.jsonl"
    args = harness.parse_args(["--mock", "--output", str(path)])
    rows = await harness.run(args)
    assert len(rows) == 100
    assert all(r["mock"] and isinstance(r["passed"], bool) for r in rows)
    assert all(r["attempts"] == 1 for r in rows if r["arm"] in "AB")
    assert any(r["attempts"] == 2 and r["passed"] for r in rows)
    assert any(not r["passed"] and r["reviewer_verdicts"] for r in rows)
    assert harness.paired(rows, "A", "C", "coding")["fixed"]
    assert harness.paired(rows, "A", "C", "coding")["broken"]
    for arm in harness.ARMS:
        assert sum(r["passed"] for r in rows if r["arm"] == arm and r["category"] == "coding") == 10
        assert sum(r["passed"] for r in rows if r["arm"] == arm and r["category"] == "api") == 3
    assert all(
        r["writer_calls"][0]["prompt"] == harness.writer_prompt(next(t for t in TASKS if t.id == r["task_id"]))
        for r in rows
    )
    before = path.read_bytes()
    assert await harness.run(args) == rows
    assert path.read_bytes() == before
    assert path.with_suffix(".md").exists()
    # Mode, prompt, task/test content and model changes must not silently mix.
    changed = harness.parse_args(["--output", str(path), "--arms", "A"])
    with pytest.raises(ValueError, match="incompatible"):
        await harness.run(changed)
    data = json.loads(path.read_text().splitlines()[0])
    assert data["protocol_hash"] and data["candidate_code"]


def test_grader_syntax_timeout_and_hidden_tests(tmp_path):
    task = TASKS[0]
    assert harness.grade(task, "def broken(:", tmp_path, 5)["passed"] is False
    result = harness.grade(task, "while True: pass", tmp_path, 0.2)
    assert result["passed"] is False and "timeout" in result["error"].lower()
    prompt = harness.writer_prompt(task)
    assert task.spec in prompt
    assert "test_hidden" not in prompt
    assert "REFERENCE" not in prompt


@pytest.mark.asyncio
async def test_quota_only_selected_engines_and_threshold(monkeypatch):
    seen = []

    async def reader(engine, **kwargs):
        seen.append(engine)
        return SimpleNamespace(error=None, windows=[SimpleNamespace(used_percent=91.0)])

    gate = harness.QuotaGate(90, reader=reader)
    with pytest.raises(harness.QuotaStop):
        await gate.check("codex")
    assert seen == ["codex"]


def test_bad_resume_is_not_silently_ignored(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"truncated":', encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL"):
        harness.load_rows(path)


def test_all_hidden_tests_accept_reference_solutions(tmp_path):
    from examples.cross_review_benchmark.mock_fixtures import SOLUTIONS

    for task in TASKS:
        result = harness.grade(task, SOLUTIONS[task.id.split("_", 2)[2]], tmp_path, 20)
        assert result["passed"], f"{task.id}: {result['output']}"


@pytest.mark.asyncio
async def test_quota_stop_during_verify_does_not_complete_task(tmp_path, monkeypatch):
    count = 0

    async def reader(engine, **kwargs):
        nonlocal count
        count += 1
        return SimpleNamespace(error=None, windows=[SimpleNamespace(used_percent=10 if count <= 3 else 99)])

    gate = harness.QuotaGate(90, reader=reader)
    monkeypatch.setattr(harness, "QuotaGate", lambda *a, **kw: gate)
    path = tmp_path / "stop.jsonl"
    args = harness.parse_args(["--mock", "--arms", "C", "--max-tasks", "1", "--output", str(path)])
    rows = await harness.run(args)
    assert rows == [] and not path.exists()
    partial = json.loads(path.with_suffix(".partial.json").read_text())
    assert len(partial["writer_calls"]) == 1
    assert partial["reviewer_calls"] == []
    assert "quota" in partial["stop_reason"]


def test_grader_import_namespace_and_future_check(tmp_path):
    from examples.cross_review_benchmark.mock_fixtures import SOLUTIONS

    # A correct candidate may use module-qualified datetime names. Its imports
    # must not replace the grader's own datetime class and invalidate its oracle.
    source = SOLUTIONS["local_midnights"].replace(
        "from datetime import datetime, time, timedelta, timezone", "import datetime"
    )
    source = source.replace("datetime.combine", "datetime.datetime.combine")
    source = source.replace("timezone.utc", "datetime.timezone.utc")
    source = source.replace("time()", "datetime.time()")
    source = source.replace("timedelta(days=1)", "datetime.timedelta(days=1)")
    task = next(t for t in TASKS if t.id == "coding_19_local_midnights")
    result = harness.grade(task, source, tmp_path, 20)
    assert result["passed"], result["output"]
    tool = harness.make_check_tool(tmp_path, harness.parse_args(["--mock"]), [])
    output = asyncio.run(tool.run(code="from __future__ import annotations\nassert 1+1 == 2\nprint('ok')"))
    assert output.startswith("exit=0") and "ok" in output


def test_regrading_preserves_provenance_without_model_calls(tmp_path, monkeypatch):
    from examples.cross_review_benchmark.mock_fixtures import SOLUTIONS
    from examples.cross_review_benchmark.regrade import regrade

    def forbidden(*args, **kwargs):
        raise AssertionError("regrading must not call a model")

    monkeypatch.setattr(harness, "live_engine", forbidden)
    task = TASKS[0]
    code = SOLUTIONS["gap_runs"]
    original = {
        "task_id": task.id,
        "arm": "A",
        "category": "coding",
        "mock": True,
        "seed": 20261011,
        "model_config": {"codex": "gpt-6.1-sol", "claude": "sonnet", "codex_effort": "high"},
        "protocol_hash": "previous",
        "passed": False,
        "error": "old grader failure",
        "grade": {"passed": False},
        "candidate_code": code,
        "writer_calls": [{"prompt": harness.writer_prompt(task), "text": code}],
    }
    source, target = tmp_path / "original.jsonl", tmp_path / "regraded.jsonl"
    source.write_text(json.dumps(original) + "\n", encoding="utf-8")
    before = source.read_bytes()
    rows = regrade(source, target, "grader isolation fix")
    assert rows[0]["passed"] is True
    assert rows[0]["generation_protocol_hash"] == "previous"
    assert rows[0]["regrade"]["previous_passed"] is False
    assert rows[0]["writer_calls"] == original["writer_calls"]
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="new output"):
        regrade(source, target, "never overwrite")
