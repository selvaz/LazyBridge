"""Replay controller: pump recorded events into the hub with controls."""

from __future__ import annotations

import time

import pytest

from lazybridge.ext.viz.exporter import EventHub
from lazybridge.ext.viz.replay import ReplayController, reconstruct_graph


def _events(n: int) -> list[dict]:
    return [{"event_type": "loop_step", "agent_name": "a", "ts": float(i)} for i in range(n)]


def test_step_emits_one_event():
    hub = EventHub()
    q = hub.subscribe(replay_recent=False)
    rc = ReplayController(hub, _events(3), speed=1.0)
    assert rc.progress == (0, 3)
    rc.step()
    assert rc.progress == (1, 3)
    out = q.get(timeout=0.5)
    assert out["event_type"] == "loop_step"


def test_step_past_end_is_noop():
    hub = EventHub()
    rc = ReplayController(hub, _events(1), speed=1.0)
    rc.step()
    rc.step()  # no-op
    assert rc.progress == (1, 1)


def test_play_drives_thread():
    hub = EventHub()
    q = hub.subscribe(replay_recent=False)
    # Tight gaps → fast playback
    evs = [{"event_type": "loop_step", "ts": 0.0 + 0.01 * i} for i in range(5)]
    rc = ReplayController(hub, evs, speed=10.0)
    rc.start()
    rc.play()
    deadline = time.monotonic() + 3.0
    seen = 0
    while seen < 5 and time.monotonic() < deadline:
        try:
            q.get(timeout=0.5)
            seen += 1
        except Exception:
            pass
    rc.stop()
    assert seen == 5


def test_pause_holds_playback():
    hub = EventHub()
    q = hub.subscribe(replay_recent=False)
    rc = ReplayController(hub, _events(3), speed=10.0)
    rc.start()
    # Don't call play()  → controller is paused by default
    time.sleep(0.3)
    rc.stop()
    assert q.empty()


def test_reconstruct_graph_finds_unique_agents():
    events = [
        {"event_type": "agent_start", "agent_name": "alice"},
        {"event_type": "tool_call", "agent_name": "alice", "name": "search"},
        {"event_type": "agent_start", "agent_name": "bob"},
        {"event_type": "loop_step"},  # no agent_name
    ]
    g = reconstruct_graph(events)
    names = sorted(n["name"] for n in g["nodes"])
    assert names == ["alice", "bob"]
