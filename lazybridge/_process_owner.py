"""Process ownership for durable records shared on one host."""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Mapping
from typing import Any

_PROCESS_BOOT_ID = str(uuid.uuid4())


def current_owner_fields() -> dict[str, Any]:
    """Return this process's PID and identity for a durable claim."""
    return {"owner_pid": os.getpid(), "owner_boot_id": _PROCESS_BOOT_ID}


def _pid_is_running(pid: int) -> bool:
    """Probe a foreign PID and treat an inconclusive response as live."""
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=5.0
            )
        except (OSError, subprocess.SubprocessError, UnicodeError):
            return True
        if result.returncode != 0:
            return True
        return not result.stdout.strip().upper().startswith("INFO:")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, OverflowError):
        return True
    return True


def owner_is_alive(record: Mapping[str, Any]) -> bool | None:
    """Return liveness for a stamped claim, or None for an unknown owner."""
    pid = record.get("owner_pid")
    boot_id = record.get("owner_boot_id")
    if type(pid) is not int or pid <= 0 or not isinstance(boot_id, str) or not boot_id:
        return None
    if pid == os.getpid():
        return boot_id == _PROCESS_BOOT_ID
    return _pid_is_running(pid)
