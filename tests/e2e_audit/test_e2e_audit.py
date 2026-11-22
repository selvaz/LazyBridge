"""Opt-in subprocess execution of each offline audit scenario."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted(Path(__file__).parent.glob("s[0-9][0-9]_*.py"))


@pytest.mark.e2e
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.stem)
def test_scenario(script, tmp_path):
    env = os.environ.copy()
    env["LB_E2E_WORK"] = str(tmp_path)
    env["PYTHONPATH"] = env.get("LB_E2E_CODE_ROOT", str(Path(__file__).resolve().parents[2]))
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
    result_path = tmp_path / script.stem / "standalone" / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] == "SKIP":
            pytest.skip(result["observed"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
