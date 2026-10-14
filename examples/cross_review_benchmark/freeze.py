"""Freeze the latest complete calibration design, before cross-provider runs."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from examples.cross_review_benchmark.harness import load_rows
from examples.cross_review_benchmark.suites import HERE, TASK_FILES, directory_hash, load_suite


def freeze(paths):
    suite = load_suite("v2")
    destination = suite.directory / "freeze.json"
    if destination.exists():
        raise ValueError("v2 already frozen; never overwrite registration")
    if not 1 <= len(paths) <= 3:
        raise ValueError("one initial calibration and at most two revisions")
    # Count distinct runs, not arguments: the same calibration passed twice is not a revision.
    if len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError("duplicate calibration run; each revision must be a distinct run")
    runs = []
    for path in paths:
        rows = load_rows(path)
        if (
            len(rows) != 45
            or {r["arm"] for r in rows} != {"A"}
            or any(r["mock"] or r["suite"] != "v2" or r["phase"] != "calibration" for r in rows)
        ):
            raise ValueError("every calibration must contain all 45 live arm A records")
        if {r["task_id"] for r in rows} != {t.id for t in suite.tasks}:
            raise ValueError("task IDs changed between calibration designs")
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if any(
            r["task_sha256"] != manifest["task_sha256"] or r["protocol_hash"] != manifest["protocol_hash"] for r in rows
        ):
            raise ValueError("mixed calibration provenance")
        # Snapshot contents, not just its filename, must match the registration.
        with zipfile.ZipFile(path.with_suffix(".tasks.zip")) as archive:
            import hashlib

            digest = hashlib.sha256()
            for name in TASK_FILES:
                data = archive.read(name).replace(b"\r\n", b"\n")
                digest.update(name.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
        if digest.hexdigest() != manifest["task_sha256"]:
            raise ValueError("calibration snapshot changed")
        counts = {
            category: sum(r["passed"] for r in rows if r["category"] == category) for category in ("coding", "api")
        }
        if runs and 8 <= runs[-1]["passes"]["coding"] <= 14 and 10 <= runs[-1]["passes"]["api"] <= 17:
            raise ValueError("calibration should have stopped when both targets were met")
        runs.append(
            {
                "file": path.relative_to(HERE).as_posix(),
                **manifest,
                "passes": counts,
                "started_at": min(r["started_at"] for r in rows),
                "finished_at": max(r["finished_at"] for r in rows),
            }
        )
    # A copied file under another name is still the same run.
    if len({(run["started_at"], run["finished_at"]) for run in runs}) != len(runs):
        raise ValueError("duplicate calibration run; each revision must be a distinct run")
    current = directory_hash(suite.directory)
    if runs[-1]["task_sha256"] != current:
        raise ValueError("latest complete calibration must use current design")
    if len(runs) < 3 and not (8 <= runs[-1]["passes"]["coding"] <= 14 and 10 <= runs[-1]["passes"]["api"] <= 17):
        raise ValueError("outside target: use remaining design revisions before freezing")
    record = {
        "task_sha256": current,
        "frozen_at": datetime.now(UTC).isoformat(),
        "selection_rule": "latest design; stop early only if both categories meet target; at most two revisions",
        "targets": {"coding": [8, 14], "api": [10, 17]},
        "calibration_runs": runs,
        "cross_provider_arms_run": [],
        "fresh_baseline": "v2/results/arm_a.jsonl",
    }
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibrations", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze([p.resolve() for p in args.calibrations]), indent=2))


if __name__ == "__main__":
    main()
