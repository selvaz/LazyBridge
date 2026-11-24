from lazybridge.policy._claude_release_check import _resume_policy_probe


def test_resume_policy_distinguishes_existing_gaps_from_regressions():
    start = {"egress": {"passed": False}, "reads": {"passed": True}, "writes": {"passed": True}}
    resumed = {"egress": {"passed": False}, "reads": {"passed": False}}

    probe = _resume_policy_probe(start, resumed)

    assert not probe["passed"]
    assert probe["detail"] == (
        "Pre-existing failures on resume: egress. New or missing failures on resume: reads, writes."
    )
    assert probe["probes"] == resumed


def test_resume_policy_passes_only_when_all_enforcement_probes_pass():
    passed = {"reads": {"passed": True}}
    assert _resume_policy_probe(passed, passed)["passed"]
    assert not _resume_policy_probe({"reads": {"passed": False}}, {"reads": {"passed": False}})["passed"]
