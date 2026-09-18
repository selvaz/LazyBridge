"""Does this storage engine actually survive our concurrency?

A decision gate, not a test. The plan that led here says SQLite in WAL
mode is the CANDIDATE, kept only if it survives real contention on the
real filesystem -- and that if it does not, the answer is a server
database, never a retry loop that hides an unsuitable substrate.

So this uses separate PROCESSES, not threads. Threads in one interpreter
share a connection pool and a GIL and would happily pass while the thing
we actually deploy -- several agent processes on one Windows disk --
fails. Anything that only reproduces across processes is exactly what a
thread-based check cannot see.

    python -m lazybridge.control_plane.probe --store C:/tmp/probe.db \\
        --scenario concurrent-queue --workers 16 --items 10000

Exit code 0 only when every count is what it must be. A probe that
reports "mostly fine" has answered a question nobody asked.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .ledger import verify_ledger
from .store import ControlStore, FenceRejected


def _drain(store_path: str, worker_id: int, deadline: float, out: Any) -> None:
    """One worker process: claim, finish, repeat until the queue is dry."""
    store = ControlStore(store_path)
    claimed: list[tuple[str, int]] = []
    locked = 0
    try:
        while time.time() < deadline:
            try:
                item = store.claim(owner=f"worker-{worker_id}")
            except Exception as exc:
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    locked += 1
                    continue
                raise
            if item is None:
                break
            claimed.append((item.item_id, item.fence))
            store.finish(item.item_id, fence=item.fence, status="done")
    finally:
        store.close()
        out.put({"worker": worker_id, "claimed": claimed, "locked": locked})


def concurrent_queue(store_path: str, *, workers: int, items: int, timeout: float) -> int:
    store = ControlStore(store_path)
    store.create_project("probe", "probe", status="open")
    for n in range(items):
        store.enqueue("probe", {"n": n})
    store.close()

    queue: Any = mp.Queue()
    deadline = time.time() + timeout
    procs = [mp.Process(target=_drain, args=(store_path, w, deadline, queue), daemon=False) for w in range(workers)]
    started = time.time()
    for p in procs:
        p.start()

    results = [queue.get() for _ in procs]
    for p in procs:
        p.join(timeout=30)

    all_claims = [c for r in results for c in r["claimed"]]
    locked = sum(r["locked"] for r in results)
    counts = Counter(item_id for item_id, _fence in all_claims)
    duplicates = {k: v for k, v in counts.items() if v > 1}

    store = ControlStore(store_path)
    final = store.counts()
    problems = verify_ledger(store)
    store.close()

    lost = items - final.get("done", 0)
    elapsed = time.time() - started
    print(f"claimed={len(all_claims)} distinct={len(counts)} terminal={final.get('done', 0)}")
    print(f"duplicate_claims={len(duplicates)} lost={lost} database_locked_errors={locked}")
    print(f"ledger_problems={len(problems)} elapsed={elapsed:.1f}s workers={workers}")
    for problem in problems[:5]:
        print(f"  {problem}")

    ok = len(all_claims) == items and not duplicates and lost == 0 and locked == 0 and not problems
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _stall_then_report(store_path: str, out: Any) -> None:
    """Claim, then die without reporting -- the worker that goes away."""
    store = ControlStore(store_path, lease_seconds=0.5)
    item = store.claim(owner="the-one-that-stalls")
    out.put({"item_id": item.item_id, "fence": item.fence})
    store.close()
    os._exit(0)  # no atexit, no cleanup: as abrupt as a kill


def kill_and_reclaim(store_path: str) -> int:
    store = ControlStore(store_path, lease_seconds=0.5)
    store.create_project("probe", "probe", status="open")
    store.enqueue("probe", {"work": "the only item"})
    store.close()

    queue: Any = mp.Queue()
    victim = mp.Process(target=_stall_then_report, args=(store_path, queue))
    victim.start()
    stale = queue.get(timeout=30)
    victim.join(timeout=30)

    time.sleep(0.8)  # the lease lapses while nobody is holding it

    store = ControlStore(store_path, lease_seconds=60.0)
    fresh = store.claim(owner="the-one-that-takes-over")
    if fresh is None:
        print("FAIL: a lapsed lease was not reclaimable")
        store.close()
        return 1

    stale_refused = False
    try:
        store.finish(stale["item_id"], fence=stale["fence"], status="done")
    except FenceRejected:
        stale_refused = True

    store.finish(fresh.item_id, fence=fresh.fence, status="done")

    second_completion_refused = False
    try:
        store.finish(fresh.item_id, fence=fresh.fence, status="done")
    except FenceRejected:
        second_completion_refused = True

    events = store.events(fresh.item_id)
    terminal = [e for e in events if e["event_type"] in ("done", "failed")]
    problems = verify_ledger(store)
    store.close()

    print(f"stale_fence={stale['fence']} fresh_fence={fresh.fence}")
    print(f"stale_token_refused={stale_refused} double_completion_refused={second_completion_refused}")
    print(f"terminal_events={len(terminal)} ledger_problems={len(problems)}")
    ok = (
        stale_refused
        and second_completion_refused
        and fresh.fence > stale["fence"]
        and len(terminal) == 1
        and not problems
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--scenario", required=True, choices=("concurrent-queue", "kill-and-reclaim"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--items", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    path = Path(args.store)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)

    if args.scenario == "concurrent-queue":
        return concurrent_queue(str(path), workers=args.workers, items=args.items, timeout=args.timeout)
    return kill_and_reclaim(str(path))


if __name__ == "__main__":
    sys.exit(main())
