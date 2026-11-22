"""Process ownership for durable records shared on one host."""

from __future__ import annotations

import ctypes
import os
import time
import uuid
from collections.abc import Mapping
from typing import Any

_PROCESS_BOOT_ID = str(uuid.uuid4())


def _process_start_time(pid: int) -> float | None:
    """Return a stable process creation stamp, or ``None`` if unavailable."""
    if os.name == "nt":
        try:
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            # PROCESS_QUERY_LIMITED_INFORMATION: also granted for processes of
            # other users, unlike PROCESS_QUERY_INFORMATION (0x0400).
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return None
            try:
                values = [wintypes.FILETIME() for _ in range(4)]
                if not kernel32.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
                    return None
                creation = values[0]
                return float((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return None
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as stat:
            return float(stat.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


_PROCESS_START_TIME = _process_start_time(os.getpid())


def current_owner_fields() -> dict[str, Any]:
    """Return this process's PID and identity for a durable claim."""
    fields: dict[str, Any] = {"owner_pid": os.getpid(), "owner_boot_id": _PROCESS_BOOT_ID}
    if _PROCESS_START_TIME is not None:
        fields["owner_start_time"] = _PROCESS_START_TIME
    return fields


def probe_pid(pid: int) -> bool | None:
    """Probe liveness without parsing localised command output."""
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                error = ctypes.get_last_error()  # type: ignore[attr-defined]
                # 87 ERROR_INVALID_PARAMETER: no such process. 5 ERROR_ACCESS_DENIED:
                # the process exists but this account may not open it -> alive, so a
                # live owner running under another account is never reclaimed.
                if error == 87:
                    return False
                return True if error == 5 else None
            try:
                code = ctypes.c_ulong()
                return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259)
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError):
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM means the process exists but belongs to another user.
        return True
    except (OverflowError, OSError):
        return None
    return True


def _pid_is_running(pid: int) -> bool | None:
    """Deprecated compatibility alias for the tri-state process probe."""
    return probe_pid(pid)


def owner_is_alive(record: Mapping[str, Any]) -> bool | None:
    """Return liveness for a stamped claim, including PID-reuse detection."""
    pid = record.get("owner_pid")
    boot_id = record.get("owner_boot_id")
    if type(pid) is not int or pid <= 0 or not isinstance(boot_id, str) or not boot_id:
        return None
    if pid == os.getpid():
        return boot_id == _PROCESS_BOOT_ID
    alive = _pid_is_running(pid)
    if alive is not True:
        return alive
    expected = record.get("owner_start_time")
    actual = _process_start_time(pid)
    if isinstance(expected, (int, float)) and actual is not None:
        return abs(float(expected) - actual) < 0.5
    return True


def record_age(record: Mapping[str, Any], *, now: float | None = None) -> float | None:
    """Best-effort age from numeric or ISO durable-record timestamps."""
    from datetime import datetime

    value = record.get("updated_at", record.get("created_at"))
    if isinstance(value, (int, float)):
        stamp = float(value)
    elif isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    else:
        return None
    return (now if now is not None else time.time()) - stamp
