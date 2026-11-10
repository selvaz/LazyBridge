import json

import pytest

from examples.cross_review_benchmark import cost_report
from examples.cross_review_benchmark.harness import load_rows


def test_report_synthetic_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_report, "_PRICE_TABLE", {"gpt-test": (2.0, 0.5, 10.0)})
    codex = {"provider": "codex", "metadata": {"model": "gpt-test", "input_tokens": 1000,
                                                "cached_input_tokens": 400, "output_tokens": 100}}
    claude = {"provider": "claude_code", "metadata": {"cost_usd": 0.02}}
    rows = []
    for task, outcomes in (("one", (True, False, True, False, True, False)),
                           ("two", (False, True, False, True, False, True)),
                           ("three", (False, False, True, False, False, False))):
        for arm, passed in zip("ABCDEF", outcomes):
            calls = [codex] if arm in "ACE" else [claude]
            if arm in "CDEF":
                calls.append(claude if arm in "CF" else codex)
            rows.append({"task_id": task, "arm": arm, "passed": passed,
                         "writer_calls": calls[:1], "reviewer_calls": calls[1:]})
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = cost_report.report(load_rows(path))
    assert "| A | 1/3 | 33.3%" in output
    assert "$0.002400" in output
    assert "| Oracle (A or B) | 2/3 | 66.7% |" in output
    assert "| max(A,B) on A/B pairs | 1/3 | 33.3% |" in output
    assert "| max(A,B), complete cohort | 1/3 | 33.3% |" in output
    assert "both fail 1; only A fails 1; only B fails 1" in output
    assert cost_report.row_cost(rows[2])[0] == pytest.approx(0.0224)


def test_missing_usage_and_unknown_price_are_not_free(monkeypatch):
    monkeypatch.setattr(cost_report, "_PRICE_TABLE", {"gpt-test": (2.0, 0.5, 10.0)})
    call = {"provider": "codex", "metadata": {"model": "gpt-test", "input_tokens": 100, "output_tokens": 10}}
    assert cost_report.call_cost(call) == pytest.approx((0.0003, True))
    row = {"arm": "A", "task_id": "t", "passed": True, "writer_calls": [call], "reviewer_calls": []}
    assert "upper bound" in cost_report.report([row])
    call["metadata"]["model"] = "gpt-unknown"
    assert cost_report.row_cost(row)[0] is None
    assert "| A | 1/1 | 100.0%" in cost_report.report([row])
    assert "unknown | unknown" in cost_report.report([row])
    assert cost_report.call_cost({"provider": "claude_code", "metadata": {}})[0] is None


def test_control_mock_scripts_match_cross_review():
    from examples.cross_review_benchmark.suites import load_suite

    suite = load_suite("v2")
    for task in suite.tasks[:4]:
        assert suite.scripts(task, "C") == suite.scripts(task, "E")
        assert suite.scripts(task, "D") == suite.scripts(task, "F")


def test_incomplete_pairs_and_no_passes():
    row = {"arm": "B", "task_id": "t", "passed": False,
           "writer_calls": [{"provider": "claude_code", "metadata": {"cost_usd": 0.1}}], "reviewer_calls": []}
    output = cost_report.report([row])
    assert "undefined (0 passes)" in output
    assert "Oracle (A or B) | 0/0 | pending" in output
