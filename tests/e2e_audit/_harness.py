"""Process tracing and SQLite evidence for the delegation audit scenarios."""

from __future__ import annotations

import collections
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(os.environ.get("LB_E2E_CODE_ROOT", ROOT)).resolve()
PYTHON = Path(sys.executable)
WORK = Path(os.environ.get("LB_E2E_WORK", Path(tempfile.gettempdir()) / "lb-e2e"))


class SkipScenario(Exception):
    pass


def scenario_dir(name):
    run_id = os.environ.get("LB_E2E_RUN_ID", "standalone")
    if Path(run_id).name != run_id or run_id in ("", ".", ".."):
        raise ValueError("Invalid run ID")
    return WORK / name / run_id


def environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def child(script, *args):
    return subprocess.Popen([str(PYTHON), str(script), *map(str, args)], env=environment())


def start_trace(directory):
    """Trace this process and subsequently created threads, with immediate flushes."""
    import lazybridge

    print(f"pid={os.getpid()} lazybridge.__file__={lazybridge.__file__}", flush=True)
    assert Path(lazybridge.__file__).resolve().is_relative_to(CODE_ROOT)
    path = Path(directory) / f"trace-{os.getpid()}.jsonl"
    stream = path.open("w", encoding="utf-8", buffering=1)
    depths = threading.local()

    def profile(frame, event, arg):
        if event not in ("call", "return"):
            return
        depth = getattr(depths, "value", 0)
        if event == "return":
            depths.value = max(0, depth - 1)
            return
        depths.value = depth + 1
        module = frame.f_globals.get("__name__", "")
        if module.startswith("lazybridge."):
            stream.write(
                json.dumps(
                    {
                        "process": os.getpid(),
                        "timestamp": time.time(),
                        "function": f"{module}.{frame.f_code.co_qualname}",
                        "depth": depth,
                    }
                )
                + "\n"
            )

    sys.setprofile(profile)
    threading.setprofile(profile)
    return stream


def snapshot(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".db", ".sqlite", ".sqlite3"):
            continue
        uri = path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            tables = {}
            for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                quoted = '"' + name.replace('"', '""') + '"'
                cursor = db.execute(f"SELECT * FROM {quoted}")
                columns = [c[0] for c in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                tables[name] = rows
            result[str(path.relative_to(directory))] = tables
    return result


def write_evidence(directory, before):
    directory = Path(directory)
    after = snapshot(directory)
    lines = ["# Database changes", ""]
    for db in sorted(set(before) | set(after)):
        lines += [f"## {db}", ""]
        for table in sorted(set(before.get(db, {})) | set(after.get(db, {}))):
            old = before.get(db, {}).get(table, [])
            new = after.get(db, {}).get(table, [])

            def encode(row):
                return json.dumps(row, sort_keys=True, default=str, ensure_ascii=False)

            removed = collections.Counter(map(encode, old)) - collections.Counter(map(encode, new))
            added = collections.Counter(map(encode, new)) - collections.Counter(map(encode, old))
            lines += [f"### {table}: {len(old)} → {len(new)} rows", "", "```json"]
            lines += [
                json.dumps(
                    {"removed": list(removed.elements()), "added": list(added.elements())}, ensure_ascii=False, indent=2
                ),
                "```",
                "",
            ]
    if len(lines) == 2:
        lines += ["No SQLite databases found.", ""]
    (directory / "db-diff.md").write_text("\n".join(lines), encoding="utf-8")
    events = []
    for path in directory.glob("trace-*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    events.sort(key=lambda item: item["timestamp"])
    (directory / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    counts = collections.Counter(e["function"] for e in events)
    (directory / "calls.md").write_text(
        "# LazyBridge calls\n\n| Function | Calls |\n| --- | ---: |\n"
        + "".join(f"| `{name}` | {count} |\n" for name, count in counts.items()),
        encoding="utf-8",
    )


def scenario(name, finding, expected, run):
    directory = scenario_dir(name)
    if directory.is_symlink():
        raise RuntimeError(f"Refusing to clean a symlink: {directory}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    before = snapshot(directory)
    started = time.monotonic()
    trace = start_trace(directory)
    status = "PASS"
    try:
        observed = run(directory)
    except SkipScenario as exc:
        status = "SKIP"
        observed = str(exc)
    except Exception:
        # KeyboardInterrupt/SystemExit propagate; the finally block still records the run.
        status = "FAIL"
        observed = traceback.format_exc()
        print(observed, file=sys.stderr, flush=True)
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        trace.close()
    result = dict(
        scenario=name,
        finding=finding,
        expected=expected,
        observed=observed,
        status=status,
        duration=round(time.monotonic() - started, 3),
    )
    write_evidence(directory, before)
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{name}: {status} ({result['duration']}s)", flush=True)
    return 0 if status in ("PASS", "SKIP") else 1
