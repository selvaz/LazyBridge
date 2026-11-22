"""Run the delegation audit serially and collect reviewable evidence."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from _harness import PYTHON, WORK, environment

EVIDENCE = Path("D:/work/r2-live/evidence/lb-e2e-audit")
BASELINES = {
    "01": ("A4 live checkpoint takeover", "1.8.10 allowed resume to take over a live checkpoint."),
    "02": ("A4 legacy checkpoint", "1.8.10 had no owner-aware legacy reconciliation."),
    "03": ("Windows PID reuse", "1.8.10 did not stamp or compare process creation times."),
    "04": ("A6 startup recovery", "1.8.10 could interrupt live jobs during recovery."),
    "05": ("A5 blackboard lease", "1.8.10 could lose a lease while its worker was active."),
    "06": ("A7 uncertain outcome", "1.8.10 could automatically replay an uncertain operation."),
    "07": ("A3 failed / guarded child", "1.8.10 could expose failed or denied child output."),
    "08": ("A7 verify retry", "Ordinary-error retry behavior on 1.8.10 has not been verified."),
    "09": ("M3 reservations", "1.8.10 could strand reservations during cancellation."),
    "10": ("A1/A2 approvals", "1.8.10 could lose permission scope or reuse a mismatched confirmation."),
    "11": ("M4 Codex slots", "1.8.10 could leave detached workers uncounted."),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    destination = EVIDENCE / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination.mkdir(parents=True, exist_ok=False)
    scripts = sorted(Path(__file__).parent.glob("s[0-9][0-9]_*.py"))
    by_number = {p.name[1:3]: p for p in scripts}
    if len(by_number) != len(scripts):
        raise RuntimeError("Duplicate scenario numbers")
    rows = []
    failed = False
    run_id = destination.name
    env = environment()
    env["LB_E2E_RUN_ID"] = run_id
    for number, (finding, baseline) in BASELINES.items():
        script = by_number.get(number)
        if script is None:
            name = f"s{number}_not_implemented"
            target = destination / name
            target.mkdir()
            result = {
                "scenario": name,
                "finding": finding,
                "expected": "Scenario implemented and passed",
                "observed": "Not implemented",
                "status": "FAIL",
                "duration": 0,
            }
            (target / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            rows.append((name, result, baseline))
            failed = True
            print(f"s{number}: FAIL (not implemented)", flush=True)
            continue
        name = script.stem
        source = WORK / name / run_id
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [str(PYTHON), str(script)], env=env, capture_output=True, text=True, timeout=args.timeout
            )
            output = proc.stdout + proc.stderr
            code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            output = f"Timed out after {args.timeout} seconds: {exc}"
            code = 1
        target = destination / name
        if source.exists():
            shutil.copytree(source, target)
            old_runs = sorted(
                (
                    path
                    for path in source.parent.iterdir()
                    if path.is_dir() and not path.is_symlink() and re.fullmatch(r"\d{8}-\d{6}-\d{6}", path.name)
                ),
                key=lambda path: path.name,
                reverse=True,
            )
            for old in old_runs[5:]:
                shutil.rmtree(old)
        else:
            target.mkdir()
        (target / "process.log").write_text(output, encoding="utf-8")
        result_path = target / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {
                "finding": "Scenario did not produce a result",
                "status": "FAIL",
                "duration": round(time.monotonic() - started, 3),
            }
        )
        if code or result["status"] != "PASS":
            failed = True
            result["status"] = "FAIL"
        rows.append((name, result, baseline))
        print(f"{name}: {result['status']} ({result['duration']}s)", flush=True)
    report = [
        "# Delegation audit end-to-end report",
        "",
        "| Scenario | Finding | Status | Duration (s) | Evidence |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for name, result, _baseline in rows:
        links = " · ".join(
            f"[{label}]({name}/{label}.md)"
            for label in ("calls", "db-diff")
            if (destination / name / f"{label}.md").exists()
        )
        report.append(f"| {name} | {result['finding']} | {result['status']} | {result['duration']} | {links} |")
    report += [
        "",
        "## LazyBridge 1.8.10 baseline",
        "",
        "The descriptions below are derived from the audit; this run did not execute 1.8.10.",
    ]
    report += [f"- **{name}:** {baseline}" for name, _, baseline in rows]
    (destination / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Evidence: {destination}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
