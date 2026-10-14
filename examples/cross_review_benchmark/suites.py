"""Versioned data, content hashes and evaluator-only reference implementations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from examples.cross_review_benchmark.tasks import Task

HERE = Path(__file__).resolve().parent
TASK_FILES = ("tasks.json", "hidden_tests.json", "mock_solutions.json")


def directory_hash(directory):
    digest = hashlib.sha256()
    for name in TASK_FILES:
        data = (directory / name).read_bytes().replace(b"\r\n", b"\n")
        digest.update(name.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return digest.hexdigest()


def task_hash(version):
    return directory_hash(HERE / version)


def verify_freeze(directory):
    record = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    if record["task_sha256"] != directory_hash(directory):
        raise ValueError("frozen task files changed; do not run cross-provider arms")
    return record


@dataclass
class Suite:
    name: str
    directory: Path
    tasks: list[Task]
    tests: dict[str, str]
    solutions: dict[str, str]

    def test_source(self, task):
        # Separate candidate and oracle namespaces, including wildcard imports.
        return self.tests[task.id]

    def solution(self, task):
        return self.solutions[task.id]

    def scripts(self, task, arm):
        good = self.solution(task)
        bad = "raise RuntimeError('scripted wrong implementation')\n"
        scenario = self.tasks.index(task) % 4
        drafts = [good if scenario in (0, 2) else bad]
        reviews = ["approved"]
        if arm in "CD" and scenario in (1, 2):
            drafts.append(good if scenario == 1 else bad)
            reviews = [
                "rejected: check all specified constraints",
                "approved" if scenario == 1 else "rejected: still wrong",
            ]
        return drafts, reviews


def load_suite(version):
    if version not in ("v1", "v2"):
        raise ValueError("suite must be v1 or v2")
    directory = HERE / version
    tasks = [Task(**item) for item in json.loads((directory / "tasks.json").read_text(encoding="utf-8"))]
    tests = json.loads((directory / "hidden_tests.json").read_text(encoding="utf-8"))
    solutions = json.loads((directory / "mock_solutions.json").read_text(encoding="utf-8"))
    if len({t.id for t in tasks}) != len(tasks) or set(tests) != {t.id for t in tasks} or set(solutions) != set(tests):
        raise ValueError("Invalid suite inventory")
    return Suite(version, directory, tasks, tests, solutions)
