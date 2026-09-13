from __future__ import annotations

from pathlib import Path

from lazybridge import Store
from lazybridge.ext.delegation.jobs import (
    DEFAULT_JOB_PREFIX,
    JobRegistry,
    session_id_key,
)


def test_session_id_key_is_stable_scoped_and_configurable() -> None:
    first = Path("C:/work/one")
    second = Path("C:/work/two")

    assert session_id_key(first) == session_id_key(first)
    assert session_id_key(first) != session_id_key(second)
    assert session_id_key(first).startswith("delegation:session:")
    assert session_id_key(first, prefix="legacy:").startswith("legacy:")


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
