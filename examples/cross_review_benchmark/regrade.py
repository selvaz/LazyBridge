"""Re-evaluate saved candidates with explicit provenance, without generating new code."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from .harness import extract_code, grade, load_rows, parse_args, protocol_hash, summary, writer_prompt
from .tasks import TASKS


def regrade(input_path, output_path, reason, *, timeout=20):
    """Keep the original log immutable. New rows identify original generation protocol.

    Only compatible specs, mode, ordering seed and model configuration can be
    compared in the new file. Previous grades/errors are retained on every row.
    Changes to hidden-test expectations must be disclosed in the reason and docs.
    """
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError("Regrading requires a new output file; preserve the original JSONL")
    rows = load_rows(source)
    if not rows:
        raise ValueError("No completed records to regrade")
    first = rows[0]
    models = first["model_config"]
    argv = [
        "--seed",
        str(first["seed"]),
        "--codex-model",
        models["codex"],
        "--claude-model",
        models["claude"],
        "--codex-effort",
        models["codex_effort"],
        "--test-timeout",
        str(timeout),
    ]
    if first["mock"]:
        argv.append("--mock")
    args = parse_args(argv)
    fingerprint = protocol_hash(args)
    inventory = {task.id: task for task in TASKS}
    for row in rows:
        if row["model_config"] != models or row["seed"] != first["seed"] or row["mock"] != first["mock"]:
            raise ValueError("Incompatible model configuration, mode or seed in input")
        task = inventory[row["task_id"]]
        calls = row["writer_calls"]
        if calls and calls[0]["prompt"] != writer_prompt(task):
            raise ValueError(f"Specification/prompt changed for {task.id}; do not silently regrade")
        if row["candidate_code"] and (not calls or row["candidate_code"] != extract_code(calls[-1]["text"])):
            raise ValueError(f"Candidate differs from recorded model output for {task.id}")
    target.parent.mkdir(parents=True, exist_ok=True)
    new_rows = []
    with tempfile.TemporaryDirectory(prefix="cross-review-regrade-") as temp:
        for original in rows:
            row = dict(original)
            row["generation_protocol_hash"] = original.get("generation_protocol_hash", original["protocol_hash"])
            row["regrade"] = {
                "reason": reason,
                "at": datetime.now(UTC).isoformat(),
                "previous_protocol_hash": original["protocol_hash"],
                "previous_passed": original["passed"],
                "previous_error": original["error"],
                "previous_grade": original.get("grade"),
            }
            row["protocol_hash"] = fingerprint
            started = time.perf_counter()
            if original.get("grade") is not None:
                row["grade"] = grade(inventory[row["task_id"]], row["candidate_code"], Path(temp), timeout)
                row["passed"] = row["grade"]["passed"]
                row["error"] = row["grade"]["error"]
            # Engine errors remain failures: no fabricated candidate or new engine call.
            row["regrade"]["wall_seconds"] = time.perf_counter() - started
            with target.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            new_rows.append(row)
            print(f"Regrade {row['arm']} {row['task_id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    target.with_suffix(".md").write_text(summary(new_rows), encoding="utf-8")
    return new_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reason", required=True, help="Public explanation of the grader change")
    parser.add_argument("--test-timeout", type=float, default=20)
    args = parser.parse_args()
    if args.test_timeout <= 0:
        parser.error("test-timeout must be positive")
    regrade(args.input, args.output, args.reason, timeout=args.test_timeout)


if __name__ == "__main__":
    main()
