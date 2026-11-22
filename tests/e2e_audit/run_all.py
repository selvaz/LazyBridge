"""Run the delegation audit serially and collect reviewable evidence."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from _harness import PYTHON, WORK, environment

EVIDENCE = Path('D:/work/r2-live/evidence/lb-e2e-audit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    destination = EVIDENCE / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    destination.mkdir(parents=True, exist_ok=False)
    scripts = sorted(Path(__file__).parent.glob('s[0-9][0-9]_*.py'))
    rows = []
    failed = not bool(scripts)
    for script in scripts:
        name = script.stem
        source = WORK / name
        if source.exists():
            raise RuntimeError(f'Scenario directory already exists: {source}; preserve evidence and remove it explicitly before rerunning')
        started = time.monotonic()
        try:
            proc = subprocess.run([str(PYTHON), str(script)], env=environment(), capture_output=True, text=True, timeout=args.timeout)
            output = proc.stdout + proc.stderr
            code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            output = f'Timed out after {args.timeout} seconds: {exc}'
            code = 1
        target = destination / name
        if source.exists():
            shutil.copytree(source, target)
        else:
            target.mkdir()
        (target / 'process.log').write_text(output, encoding='utf-8')
        result_path = target / 'result.json'
        result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'finding': 'Scenario did not produce a result', 'status': 'FAIL', 'duration': round(time.monotonic()-started, 3)}
        if code or result['status'] != 'PASS':
            failed = True
            result['status'] = 'FAIL'
        rows.append((name, result))
        print(f"{name}: {result['status']} ({result['duration']}s)", flush=True)
    report = ['# Delegation audit end-to-end report', '', '| Scenario | Finding | Status | Duration (s) | Evidence |', '| --- | --- | --- | ---: | --- |']
    for name, result in rows:
        links = ' · '.join(f'[{label}]({name}/{label}.md)' for label in ('calls', 'db-diff') if (destination / name / f'{label}.md').exists())
        report.append(f"| {name} | {result['finding']} | {result['status']} | {result['duration']} | {links} |")
    if not scripts:
        report += ['', 'No scenarios implemented.']
    (destination / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Evidence: {destination}')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
