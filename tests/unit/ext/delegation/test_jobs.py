from __future__ import annotations

import os
from pathlib import Path

import pytest

from lazybridge import Store
from lazybridge.ext.delegation import jobs as jobs_module
from lazybridge.ext.delegation.jobs import (
    DEFAULT_JOB_PREFIX,
    JobRegistry,
    current_owner_fields,
    session_id_key,
)


def test_session_id_key_is_stable_scoped_and_configurable() -> None:
    first = Path("C:/work/one")
    second = Path("C:/work/two")

    assert session_id_key(first) == session_id_key(first)
    assert session_id_key(first) != session_id_key(second)
    assert session_id_key(first).startswith("delegation:session:")
    assert session_id_key(first, prefix="legacy:").startswith("legacy:")


def test_session_id_key_canonicalizes_relative_and_absolute_spellings() -> None:
    """An unresolved relative root scopes the key to its literal spelling
    rather than the actual workspace: Path("project") launched from two
    different parent directories would otherwise hash to the SAME key for
    two different working directories, and a relative vs. absolute
    spelling of the same directory would hash to two DIFFERENT keys for
    the same one. Found by Codex review before this ever shipped."""
    workspace = Path.cwd() / "some-project"

    assert session_id_key(workspace) == session_id_key(Path("some-project"))
    assert session_id_key(workspace) == session_id_key(workspace.resolve())


def test_write_uses_one_compatible_record_shape_and_omits_none() -> None:
    store = Store()
    registry = JobRegistry(store, prefix="custom:")

    registry.write(
        "job-a",
        "do it",
        tool_name="delegate",
        status="done",
        plan_id="plan-a",
        task_index=2,
        plan_task_text="task text",
        result="finished",
    )

    assert store.read("custom:job-a") == {
        "job_id": "job-a",
        "kind": "delegate",
        "objective": "do it",
        "status": "done",
        "plan_id": "plan-a",
        "task_index": 2,
        "plan_task_text": "task text",
        "result": "finished",
    }


def test_reclaim_interrupted_marks_only_orphanable_states() -> None:
    store = Store()
    registry = JobRegistry(store)
    registry.write("running", "x", tool_name="delegate", status="running")
    registry.write("waiting", "y", tool_name="delegate", status="awaiting_approval")
    registry.write("done", "z", tool_name="delegate", status="done", result="ok")

    assert registry.reclaim_interrupted() == ["running", "waiting"]
    assert registry.find("running")["status"] == "interrupted"
    assert registry.find("waiting")["status"] == "interrupted"
    assert registry.find("done")["status"] == "done"


def test_reclaim_interrupted_does_not_clobber_concurrent_completion() -> None:
    store = Store()
    registry = JobRegistry(store)
    registry.write("job-a", "x", tool_name="delegate", status="running")
    key = f"{DEFAULT_JOB_PREFIX}job-a"
    real_items = store.items

    def items_that_race(*, prefix: str):
        result = list(real_items(prefix=prefix))
        registry.write("job-a", "x", tool_name="delegate", status="done", result="finished for real")
        return result

    store.items = items_that_race  # type: ignore[method-assign]

    assert registry.reclaim_interrupted() == []
    assert store.read(key)["status"] == "done"
    assert store.read(key)["result"] == "finished for real"


def test_find_checker_and_result_tools_support_short_ids_and_previews() -> None:
    store = Store()
    registry = JobRegistry(store)
    full_id = "12345678-abcd-abcd-abcd-1234567890ab"
    long_objective = "objective-" + "o" * 300
    long_result = "result-" + "r" * 5000
    registry.write(
        full_id,
        long_objective,
        tool_name="delegate_plan_tasks",
        status="done",
        plan_id="plan-a",
        task_index=4,
        plan_task_text="linked task text",
        result=long_result,
    )
    registry.write("failed-job", "break it", tool_name="delegate", status="failed", error="e" * 300)

    assert registry.find(full_id)["result"] == long_result
    assert registry.find("12345678")["result"] == long_result
    assert registry.find("unknown") is None

    checker = registry.checker_tool(doc="checker docs")
    report = checker.func()
    assert checker.name == "check_jobs"
    assert checker.func.__doc__ == "checker docs"
    assert "12345678 (delegate_plan_tasks) [done]" in report
    assert "plan-a" in report and "task 4" in report and "linked task text" in report
    assert long_result[:200] in report
    assert long_result not in report
    assert "error:" in report

    getter = registry.result_tool(doc="result docs")
    full = getter.func(job_id="12345678")
    assert getter.name == "get_job_result"
    assert getter.func.__doc__ == "result docs"
    assert long_objective in full
    assert long_result in full
    assert "task_index: 4" in full
    assert "no job found" in getter.func(job_id="unknown").lower()


def test_checker_reports_no_jobs() -> None:
    assert JobRegistry(Store()).checker_tool().func() == "no jobs yet"


def test_write_accepts_richer_fields_and_omits_none() -> None:
    """engine/model/effort/execution_started/cost_usd/cost_unknown/
    created_at/finished_at are all optional fields written the same way the
    original set always was -- omitted from the record when left at their
    ``None`` default, present verbatim when given."""
    store = Store()
    registry = JobRegistry(store)

    registry.write(
        "job-a",
        "do it",
        tool_name="delegate",
        status="running",
        engine="ClaudeCodeEngine",
        model="sonnet",
        effort="high",
        execution_started=True,
        cost_usd=0.0,
        cost_unknown=True,
        created_at="2026-01-01T00:00:00+00:00",
    )

    record = store.read("delegation:job:job-a")
    assert record["engine"] == "ClaudeCodeEngine"
    assert record["model"] == "sonnet"
    assert record["effort"] == "high"
    assert record["execution_started"] is True
    assert record["cost_usd"] == 0.0
    assert record["cost_unknown"] is True
    assert record["created_at"] == "2026-01-01T00:00:00+00:00"
    assert "finished_at" not in record
    assert "owner_pid" not in record


def test_write_execution_started_false_is_not_treated_as_unset() -> None:
    """False is a legitimate, meaningful value here (an admission refusal
    recorded with execution_started=False) -- only None means "omit this
    field", so False must still land in the record, not be dropped the
    way a falsy-but-not-None value could be if this used `if value:`
    instead of `if value is not None:`."""
    store = Store()
    registry = JobRegistry(store)
    registry.write("job-a", "do it", tool_name="delegate", status="failed", execution_started=False)
    assert store.read("delegation:job:job-a")["execution_started"] is False


def test_write_rejects_extra_fields_that_collide_with_reserved_names() -> None:
    registry = JobRegistry(Store())
    with pytest.raises(ValueError, match="reserved"):
        registry.write("job-a", "do it", tool_name="delegate", status="running", extra={"status": "done", "custom": 1})


def test_write_passes_through_open_extra_fields() -> None:
    """A caller's own fields this module has never heard of -- a project
    id, a routing decision, anything -- without forking write()'s own
    signature for every new need."""
    store = Store()
    registry = JobRegistry(store)
    registry.write("job-a", "do it", tool_name="delegate", status="done", extra={"project_id": "lazy-x", "attempt": 2})
    record = store.read("delegation:job:job-a")
    assert record["project_id"] == "lazy-x"
    assert record["attempt"] == 2


def test_result_tool_shows_engine_model_cost_and_timestamps_when_present() -> None:
    registry = JobRegistry(Store())
    registry.write(
        "job-a",
        "do it",
        tool_name="delegate",
        status="done",
        result="ok",
        engine="CodexEngine",
        model="gpt-5",
        effort="medium",
        created_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:05:00+00:00",
        cost_usd=0.42,
    )
    report = registry.result_tool().func(job_id="job-a")
    assert "engine: CodexEngine" in report
    assert "model: gpt-5" in report
    assert "effort: medium" in report
    assert "created_at: 2026-01-01T00:00:00+00:00" in report
    assert "finished_at: 2026-01-01T00:05:00+00:00" in report
    assert "cost_usd: 0.42" in report


def test_result_tool_shows_cost_unknown_when_cost_usd_is_absent() -> None:
    registry = JobRegistry(Store())
    registry.write("job-a", "do it", tool_name="delegate", status="running", cost_unknown=True)
    report = registry.result_tool().func(job_id="job-a")
    assert "cost_usd: unknown" in report


def test_current_owner_fields_pairs_pid_with_a_stable_boot_id() -> None:
    first = current_owner_fields()
    second = current_owner_fields()
    assert first["owner_pid"] == os.getpid() == second["owner_pid"]
    assert first["owner_boot_id"] == second["owner_boot_id"]
    assert isinstance(first["owner_boot_id"], str) and first["owner_boot_id"]


def test_reclaim_interrupted_skips_a_job_owned_by_a_still_alive_process() -> None:
    """The whole point of owner-liveness: more than one process can share a
    Store and be concurrently ALIVE at once -- the un-ownership-aware
    version of this method could not tell that apart from a genuinely dead
    writer and would wrongly interrupt a live job. Stamping THIS process's
    own owner fields and asking reclaim_interrupted to look is the
    simplest way to exercise "owner is alive" without mocking the real
    process table."""
    store = Store()
    registry = JobRegistry(store)
    registry.write("alive", "x", tool_name="delegate", status="running", **current_owner_fields())
    registry.write("orphan", "y", tool_name="delegate", status="running")  # no owner stamp at all

    assert registry.reclaim_interrupted() == ["orphan"]
    assert registry.find("alive")["status"] == "running"
    assert registry.find("orphan")["status"] == "interrupted"


def test_reclaim_interrupted_reclaims_a_job_owned_by_a_dead_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A record naming a DIFFERENT boot id (a process that no longer
    exists, or was replaced) must be checked against the real process
    table, not assumed alive just because *a* pid is present."""
    monkeypatch.setattr(jobs_module, "_pid_is_running", lambda pid, **kwargs: False)
    store = Store()
    registry = JobRegistry(store)
    registry.write(
        "dead-owner", "x", tool_name="delegate", status="running", owner_pid=999999, owner_boot_id="stale-boot-id"
    )

    assert registry.reclaim_interrupted() == ["dead-owner"]
    assert registry.find("dead-owner")["status"] == "interrupted"


def test_reclaim_interrupted_treats_an_unknown_foreign_pid_as_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Liveness for a foreign pid is best-effort and assumes ALIVE on any
    failure/timeout/unexpected output -- wrongly treating a live writer as
    dead risks a second worker starting on top of it, worse than leaving a
    job alone for one more sweep."""
    monkeypatch.setattr(jobs_module, "_pid_is_running", lambda pid, **kwargs: True)
    store = Store()
    registry = JobRegistry(store)
    registry.write(
        "foreign", "x", tool_name="delegate", status="running", owner_pid=424242, owner_boot_id="some-other-boot"
    )

    assert registry.reclaim_interrupted() == []
    assert registry.find("foreign")["status"] == "running"


def test_job_owner_is_alive_rejects_malformed_owner_fields() -> None:
    from lazybridge.ext.delegation.jobs import _job_owner_is_alive

    assert _job_owner_is_alive({}) is False
    assert _job_owner_is_alive({"owner_pid": "not-an-int", "owner_boot_id": "x"}) is False
    assert _job_owner_is_alive({"owner_pid": 1, "owner_boot_id": None}) is False
