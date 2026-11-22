"""Run the delegation audit serially and collect reviewable evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from _harness import PYTHON, WORK, environment

EVIDENCE = Path(os.environ.get("LB_E2E_EVIDENCE", WORK / "evidence"))
BASELINES = {
    "01": ("A4 live checkpoint takeover", "1.8.10 allowed resume to take over a live checkpoint."),
    "02": ("A4 legacy checkpoint", "1.8.10 had no owner-aware legacy reconciliation."),
    "03": ("Windows PID reuse", "1.8.10 did not stamp or compare process creation times."),
    "04": ("A6 startup recovery", "1.8.10 could interrupt live jobs during recovery."),
    "05": ("A5 blackboard lease", "1.8.10 could lose a lease while its worker was active."),
    "06": ("A7 uncertain outcome", "1.8.10 could automatically replay an uncertain operation."),
    "07": ("A3 failed / guarded child", "1.8.10 could expose failed or denied child output."),
    "08": ("A7 verify retry", "Ordinary-error retry may already work on 1.8.10; see the measured status above."),
    "09": ("M3 reservations", "1.8.10 could strand reservations during cancellation."),
    "10": ("A1/A2 approvals", "1.8.10 could lose permission scope or reuse a mismatched confirmation."),
    "11": ("M4 Codex slots", "1.8.10 could leave detached workers uncounted."),
}


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Stop a timed-out scenario and every process it started."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--baseline", type=Path, help="Evidence directory from a v1.8.10 run")
    parser.add_argument("--baseline-self", action="store_true", help="This run uses v1.8.10")
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
    # Scenario processes are Python: make their output UTF-8 regardless of the console code page.
    env["PYTHONIOENCODING"] = "utf-8"
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
        # Output goes to a file, not a pipe: a descendant that outlives a
        # timeout kill cannot block the runner, and nothing printed is lost.
        log_path = WORK / f"{name}-{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(
                [str(PYTHON), str(script)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            try:
                code = proc.wait(timeout=args.timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                code = 1
                timed_out = True
        # Read once the scenario has exited (or its tree was killed); output a
        # descendant writes after that point is not captured.
        output = log_path.read_text(encoding="utf-8", errors="replace")
        try:
            log_path.unlink()
        except OSError:
            pass  # still held by a surviving descendant; pruned with the run folders
        if timed_out:
            output = f"Timed out after {args.timeout} seconds" + chr(10) + output
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
                "scenario": name,
                "finding": finding,
                "expected": "Scenario completes successfully",
                "observed": "Scenario did not produce a result; see process.log",
                "status": "FAIL",
                "duration": round(time.monotonic() - started, 3),
                "failure_kind": "startup or timeout",
            }
        )
        if code or result["status"] not in ("PASS", "SKIP"):
            failed = True
            result["status"] = "FAIL"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.append((name, result, baseline))
        print(f"{name}: {result['status']} ({result['duration']}s)", flush=True)
    report = [
        "# Delegation audit end-to-end report",
        "",
        "| Scenario | Finding | Status | Duration (s) | 1.8.10 | Evidence |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for name, result, _baseline in rows:
        if args.baseline_self:
            baseline_result = result
        elif args.baseline and (args.baseline / name / "result.json").exists():
            baseline_result = json.loads((args.baseline / name / "result.json").read_text(encoding="utf-8"))
        else:
            baseline_result = None
        baseline_status = baseline_result["status"] if baseline_result else "NOT RUN"
        if baseline_result and baseline_result.get("failure_kind"):
            baseline_status += " (startup/timeout)"
        links = " · ".join(
            f"[{label}]({name}/{label}.md)"
            for label in ("calls", "db-diff")
            if (destination / name / f"{label}.md").exists()
        )
        report.append(
            f"| {name} | {result['finding']} | {result['status']} | {result['duration']} | {baseline_status} | {links} |"
        )
    report += [
        "",
        "## LazyBridge 1.8.10 baseline",
        "",
        "The descriptions below are derived from the audit. The 1.8.10 column reports the supplied baseline run; "
        "a PASS is a regression guard, not proof of a fix. Startup or timeout failures do not prove a behavioral regression. "
        "Inspect the baseline process logs to distinguish behavioral failures from API incompatibilities.",
    ]
    if args.baseline:
        report.append(f"Baseline evidence directory: `{args.baseline.resolve()}`.")
    report += [f"- **{name}:** {baseline}" for name, _, baseline in rows]
    (destination / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Evidence: {destination}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
